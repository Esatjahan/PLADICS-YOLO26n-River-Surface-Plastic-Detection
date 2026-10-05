import os
import time
import torch
from PIL import Image
from torchvision.transforms import functional as F
from effdet import create_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MODEL_PATH = r"D:\PLADICS\experiments_efficientdet_d0\efficientdet_d0_final.pth"
IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"
LABEL_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\labels\test"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
CONF_THRES = 0.00001
IOU_THRES = 0.50

model = create_model(
    "tf_efficientdet_d0",
    bench_task="predict",
    num_classes=1,
    pretrained=False
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")


def yolo_to_xyxy(label_path, img_w, img_h):
    boxes, labels = [], []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                _, xc, yc, bw, bh = map(float, parts)

                x1 = (xc - bw / 2) * img_w
                y1 = (yc - bh / 2) * img_h
                x2 = (xc + bw / 2) * img_w
                y2 = (yc + bh / 2) * img_h

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    return inter / (area1[:, None] + area2 - inter + 1e-6)


image_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

tp, fp, fn = 0, 0, 0
total_time = 0.0

print("Running EfficientDet-D0 evaluation...")
print("Test images:", len(image_files))

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = F.to_tensor(resized).unsqueeze(0).to(DEVICE)

        start = time.time()
        output = model(tensor)
        end = time.time()

        total_time += (end - start)

        det = output[0].detach().cpu()

        if det.numel() == 0:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
            pred_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # effdet predict output: x1, y1, x2, y2, score, class
            pred_boxes = det[:, 0:4]
            pred_scores = det[:, 4]
            pred_labels = det[:, 5].long()

            pred_boxes[:, [0, 2]] *= orig_w / IMAGE_SIZE
            pred_boxes[:, [1, 3]] *= orig_h / IMAGE_SIZE

        gt_boxes, gt_labels = yolo_to_xyxy(label_path, orig_w, orig_h)

        metric.update(
            [{
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            }],
            [{
                "boxes": gt_boxes,
                "labels": gt_labels
            }]
        )

        keep = pred_scores >= CONF_THRES
        pr_boxes = pred_boxes[keep]

        matched_gt = set()

        if pr_boxes.shape[0] > 0 and gt_boxes.shape[0] > 0:
            ious = box_iou(pr_boxes, gt_boxes)

            for pred_i in range(pr_boxes.shape[0]):
                best_iou, best_gt = torch.max(ious[pred_i], dim=0)
                best_gt = int(best_gt.item())

                if best_iou.item() >= IOU_THRES and best_gt not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1

            fn += gt_boxes.shape[0] - len(matched_gt)

        elif pr_boxes.shape[0] > 0 and gt_boxes.shape[0] == 0:
            fp += pr_boxes.shape[0]

        elif pr_boxes.shape[0] == 0 and gt_boxes.shape[0] > 0:
            fn += gt_boxes.shape[0]

results = metric.compute()

precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)

avg_inference_ms = (total_time / len(image_files)) * 1000
fps = 1000 / avg_inference_ms
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("\n========== EfficientDet-D0 Test Results ==========")
print(f"Precision@IoU0.5/conf{CONF_THRES}: {precision:.4f}")
print(f"Recall@IoU0.5/conf{CONF_THRES}: {recall:.4f}")
print(f"mAP@0.5: {results['map_50'].item():.4f}")
print(f"mAP@0.5:0.95: {results['map'].item():.4f}")
print(f"mAP@0.75: {results['map_75'].item():.4f}")
print(f"AR@100: {results['mar_100'].item():.4f}")
print(f"Average inference time: {avg_inference_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Model size: {model_size_mb:.2f} MB")
print("=================================================")