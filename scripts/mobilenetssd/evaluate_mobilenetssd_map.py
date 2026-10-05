import os
import time
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MODEL_PATH = r"D:\PLADICS\experiments_mobilenetssd\mobilenetssd_final.pth"
IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"
LABEL_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\labels\test"

DEVICE = torch.device("cpu")
CONF_THRES = 0.25

model = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
)

model.head.classification_head.num_classes = 2
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox"
)

image_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

total_time = 0.0
total_images = 0

def yolo_to_xyxy(label_path, img_w, img_h):
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)

            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h

            boxes.append([x1, y1, x2, y2])
            labels.append(1)

    if len(boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

print("Running COCO-style MobileNet-SSD evaluation...")
print("Test images:", len(image_files))

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        tensor = F.to_tensor(img).to(DEVICE)

        start = time.time()
        output = model([tensor])[0]
        end = time.time()

        total_time += (end - start)
        total_images += 1

        scores = output["scores"].cpu()
        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()

        keep = scores >= CONF_THRES

        pred = {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": torch.ones((keep.sum(),), dtype=torch.int64)
        }

        gt_boxes, gt_labels = yolo_to_xyxy(label_path, img_w, img_h)

        target = {
            "boxes": gt_boxes,
            "labels": gt_labels
        }

        metric.update([pred], [target])

results = metric.compute()

avg_inference_ms = (total_time / total_images) * 1000
fps = 1000 / avg_inference_ms
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("\n========== MobileNet-SSD COCO-style Test Results ==========")
print(f"mAP@0.5:0.95: {results['map'].item():.4f}")
print(f"mAP@0.5: {results['map_50'].item():.4f}")
print(f"mAP@0.75: {results['map_75'].item():.4f}")
print(f"Recall AR@100: {results['mar_100'].item():.4f}")
print(f"Average inference time: {avg_inference_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Model size: {model_size_mb:.2f} MB")
print("===========================================================")