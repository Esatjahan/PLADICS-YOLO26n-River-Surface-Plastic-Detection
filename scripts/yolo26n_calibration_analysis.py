import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
DATASET_ROOT = r"D:\PLADICS\datasets_clean\merged_clean_dataset"

IMAGE_DIR = os.path.join(DATASET_ROOT, "images", "test")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels", "test")

OUTPUT_DIR = r"D:\PLADICS\confidence_analysis\calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.5
NUM_BINS = 10

model = YOLO(MODEL_PATH)


def yolo_to_xyxy(label, img_w, img_h):
    cls, xc, yc, bw, bh = map(float, label)
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def load_gt_boxes(label_path, img_w, img_h):
    gt_boxes = []

    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                gt_boxes.append(yolo_to_xyxy(parts, img_w, img_h))

    return gt_boxes


confidences = []
correctness = []

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Total test images:", len(image_files))

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    gt_boxes = load_gt_boxes(label_path, w, h)

    results = model.predict(
        source=img_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        continue

    pred_boxes = results.boxes.xyxy.cpu().numpy()
    pred_confs = results.boxes.conf.cpu().numpy()

    matched_gt = set()

    for pred_box, conf in zip(pred_boxes, pred_confs):
        best_iou = 0
        best_gt_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        is_correct = 0

        if best_iou >= IOU_THRES and best_gt_idx != -1:
            is_correct = 1
            matched_gt.add(best_gt_idx)

        confidences.append(float(conf))
        correctness.append(is_correct)


confidences = np.array(confidences)
correctness = np.array(correctness)

print("Total predictions:", len(confidences))

bin_edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
bin_centers = []
bin_confidences = []
bin_accuracies = []
bin_counts = []

ece = 0.0

for i in range(NUM_BINS):
    low = bin_edges[i]
    high = bin_edges[i + 1]

    if i == NUM_BINS - 1:
        mask = (confidences >= low) & (confidences <= high)
    else:
        mask = (confidences >= low) & (confidences < high)

    count = np.sum(mask)

    if count > 0:
        avg_conf = np.mean(confidences[mask])
        acc = np.mean(correctness[mask])
    else:
        avg_conf = 0
        acc = 0

    bin_centers.append((low + high) / 2)
    bin_confidences.append(avg_conf)
    bin_accuracies.append(acc)
    bin_counts.append(count)

    ece += (count / len(confidences)) * abs(acc - avg_conf)

bin_centers = np.array(bin_centers)
bin_confidences = np.array(bin_confidences)
bin_accuracies = np.array(bin_accuracies)
bin_counts = np.array(bin_counts)

print(f"Expected Calibration Error (ECE): {ece:.4f}")

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, "calibration_results.csv")
with open(csv_path, "w") as f:
    f.write("bin_start,bin_end,count,avg_confidence,accuracy\n")
    for i in range(NUM_BINS):
        f.write(
            f"{bin_edges[i]:.2f},{bin_edges[i+1]:.2f},"
            f"{bin_counts[i]},{bin_confidences[i]:.4f},{bin_accuracies[i]:.4f}\n"
        )
    f.write(f"\nECE,{ece:.4f}\n")

# Reliability diagram
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
plt.plot(bin_confidences, bin_accuracies, marker="o", label="YOLO26n")
plt.xlabel("Mean Confidence")
plt.ylabel("Detection Accuracy")
plt.title(f"YOLO26n Reliability Diagram (ECE = {ece:.3f})")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

fig_path = os.path.join(OUTPUT_DIR, "yolo26n_reliability_diagram.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved CSV:", csv_path)
print("Saved figure:", fig_path)