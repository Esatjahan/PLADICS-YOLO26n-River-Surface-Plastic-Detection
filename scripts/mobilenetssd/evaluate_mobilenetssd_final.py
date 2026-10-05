import os
import time
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MODEL_PATH = r"D:\PLADICS\experiments_mobilenetssd_correct\mobilenetssd_correct_final.pth"
IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"
LABEL_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\labels\test"

DEVICE = torch.device("cpu")
CONF_THRES = 0.25
IOU_THRES = 0.50

model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
    num_classes=2
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
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

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


image_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

tp, fp, fn = 0, 0, 0
total_time = 0.0

print("Running final MobileNet-SSD evaluation...")
print("Test images:", len(image_files))

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        tensor = F.to_tensor(img).to(DEVICE)

        start = time.time()
        output = model([tensor])[0]
        end = time.time()

        total_time += (end - start)

        gt_boxes, gt_labels = yolo_to_xyxy(label_path, img_w, img_h)

        pred_boxes = output["boxes"].cpu()
        pred_scores = output["scores"].cpu()
        pred_labels = output["labels"].cpu()

        # Keep only plastic class predictions
        class_keep = pred_labels == 1
        pred_boxes = pred_boxes[class_keep]
        pred_scores = pred_scores[class_keep]
        pred_labels = pred_labels[class_keep]

        # mAP uses score-ranked predictions
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

        # Precision/Recall at fixed confidence + IoU threshold
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

print("\n========== FINAL MobileNet-SSD Results ==========")
print(f"Precision@IoU0.5/conf{CONF_THRES}: {precision:.4f}")
print(f"Recall@IoU0.5/conf{CONF_THRES}: {recall:.4f}")
print(f"mAP@0.5: {results['map_50'].item():.4f}")
print(f"mAP@0.5:0.95: {results['map'].item():.4f}")
print(f"mAP@0.75: {results['map_75'].item():.4f}")
print(f"AR@100: {results['mar_100'].item():.4f}")
print(f"Average inference time: {avg_inference_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Model size: {model_size_mb:.2f} MB")
print("================================================")