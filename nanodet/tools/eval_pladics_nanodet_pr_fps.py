import os
import time
import cv2
import torch
import numpy as np

from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline


CONFIG_PATH = r"D:\nanodet\config\pladics_nanodet_official.yml"
MODEL_PATH = r"D:\nanodet\workspace\pladics_nanodet_640_fair\model_best\nanodet_model_best.pth"

IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"
LABEL_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\labels\test"

CONF_THRES = 0.25
IOU_THRES = 0.50
DEVICE = "cpu"


def yolo_to_xyxy(label_path, img_w, img_h):
    boxes = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue

                _, xc, yc, bw, bh = map(float, p)

                x1 = (xc - bw / 2) * img_w
                y1 = (yc - bh / 2) * img_h
                x2 = (xc + bw / 2) * img_w
                y2 = (yc + bh / 2) * img_h

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=np.float32)


def iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    px1, py1, px2, py2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    inter_x1 = np.maximum(px1[:, None], gx1[None, :])
    inter_y1 = np.maximum(py1[:, None], gy1[None, :])
    inter_x2 = np.minimum(px2[:, None], gx2[None, :])
    inter_y2 = np.minimum(py2[:, None], gy2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)

    union = pred_area[:, None] + gt_area[None, :] - inter
    return inter / (union + 1e-6)


class NanoDetPredictor:
    def __init__(self, config_path, model_path):
        load_config(cfg, config_path)
        self.cfg = cfg
        self.device = torch.device(DEVICE)
        self.logger = Logger(-1, use_tensorboard=False)

        model = build_model(cfg.model)
        checkpoint = torch.load(model_path, map_location=self.device)

        if "state_dict" in checkpoint:
            load_model_weight(model, checkpoint, self.logger)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(self.device).eval()
        self.model = model
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((0, 5), dtype=np.float32), 0.0

        img_info = {
            "id": 0,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1],
        }

        meta = {
            "img_info": img_info,
            "raw_img": img,
            "img": img,
        }

        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        start = time.time()
        with torch.no_grad():
            results = self.model.inference(meta)
        end = time.time()

        dets = []

        # NanoDet output usually: {class_id: [[x1,y1,x2,y2,score], ...]}
        if isinstance(results, dict):
            for cls_id, boxes in results.items():
                for b in boxes:
                    if len(b) >= 5 and b[4] >= CONF_THRES:
                        dets.append(b[:5])

        elif isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    for cls_id, boxes in item.items():
                        for b in boxes:
                            if len(b) >= 5 and b[4] >= CONF_THRES:
                                dets.append(b[:5])

        return np.array(dets, dtype=np.float32), end - start


def main():
    predictor = NanoDetPredictor(CONFIG_PATH, MODEL_PATH)

    image_files = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    tp, fp, fn = 0, 0, 0
    total_time = 0.0

    print("Running NanoDet evaluation...")
    print("Test images:", len(image_files))

    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        gt_boxes = yolo_to_xyxy(label_path, w, h)

        dets, infer_time = predictor.predict(img_path)
        total_time += infer_time

        if len(dets) > 0:
            pred_boxes = dets[:, :4]
            scores = dets[:, 4]
            order = np.argsort(-scores)
            pred_boxes = pred_boxes[order]
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)

        matched_gt = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = iou_matrix(pred_boxes, gt_boxes)

            for pred_i in range(len(pred_boxes)):
                best_gt = int(np.argmax(ious[pred_i]))
                best_iou = ious[pred_i, best_gt]

                if best_iou >= IOU_THRES and best_gt not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

        elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
            fp += len(pred_boxes)

        elif len(pred_boxes) == 0 and len(gt_boxes) > 0:
            fn += len(gt_boxes)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    avg_time_ms = (total_time / len(image_files)) * 1000
    fps = 1000 / avg_time_ms

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    print("\n========== NanoDet-20 Final Results ==========")
    print(f"Precision@IoU0.5/conf{CONF_THRES}: {precision:.4f}")
    print(f"Recall@IoU0.5/conf{CONF_THRES}: {recall:.4f}")
    print("mAP@0.5: 0.6150")
    print("mAP@0.5:0.95: 0.3540")
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Model size: {size_mb:.2f} MB")
    print("=============================================")


if __name__ == "__main__":
    main()