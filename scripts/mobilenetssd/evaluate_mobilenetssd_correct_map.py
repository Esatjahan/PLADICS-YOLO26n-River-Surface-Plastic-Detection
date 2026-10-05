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

model = ssdlite320_mobilenet_v3_large(
    weights=None,
    weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
    num_classes=2
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)

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

                boxes.append([x1, y1, x2, y2])
                labels.append(1)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


image_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

total_time = 0.0

print("Running corrected MobileNet-SSD evaluation...")
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

        preds = [{
            "boxes": output["boxes"].cpu(),
            "scores": output["scores"].cpu(),
            "labels": output["labels"].cpu()
        }]

        targets = [{
            "boxes": gt_boxes,
            "labels": gt_labels
        }]

        metric.update(preds, targets)

results = metric.compute()

avg_inference_ms = (total_time / len(image_files)) * 1000
fps = 1000 / avg_inference_ms
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("\n========== Corrected MobileNet-SSD Test Results ==========")
print(f"mAP@0.5: {results['map_50'].item():.4f}")
print(f"mAP@0.5:0.95: {results['map'].item():.4f}")
print(f"mAP@0.75: {results['map_75'].item():.4f}")
print(f"Recall AR@100: {results['mar_100'].item():.4f}")
print(f"Average inference time: {avg_inference_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Model size: {model_size_mb:.2f} MB")
print("==========================================================")