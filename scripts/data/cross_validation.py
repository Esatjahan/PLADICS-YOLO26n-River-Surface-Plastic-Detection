import os
import shutil
import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:/PLADICS/datasets_clean/merged_clean_dataset"
OUTPUT_PATH = r"D:/PLADICS/experiments_crossval_clean"

MODEL_PATH = r"D:/PLADICS/yolo26n.pt"

K = 4
EPOCHS = 40
IMGSZ = 640
BATCH = 8
DEVICE = "cpu"
WORKERS = 0

# =========================
# PATHS
# =========================
images_dir = os.path.join(DATASET_PATH, "images")
labels_dir = os.path.join(DATASET_PATH, "labels")

train_img_dir = os.path.join(images_dir, "train")
val_img_dir = os.path.join(images_dir, "val")

train_lbl_dir = os.path.join(labels_dir, "train")
val_lbl_dir = os.path.join(labels_dir, "val")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# COLLECT TRAIN + VAL ONLY
# test set will remain untouched
# =========================
samples = []

for split in ["train", "val"]:
    img_dir = os.path.join(images_dir, split)
    lbl_dir = os.path.join(labels_dir, split)

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        img_path = os.path.join(img_dir, img_name)
        label_name = Path(img_name).stem + ".txt"
        lbl_path = os.path.join(lbl_dir, label_name)

        if os.path.exists(lbl_path):
            samples.append({
                "image_name": img_name,
                "label_name": label_name,
                "image_path": img_path,
                "label_path": lbl_path,
                "original_split": split
            })

print(f"Total train+val samples for K-fold: {len(samples)}")

if len(samples) == 0:
    raise ValueError("No samples found. Check dataset path.")

# =========================
# K-FOLD
# =========================
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

results_summary = []

fold = 1

for train_idx, val_idx in kfold.split(samples):
    print(f"\n===== Fold {fold}/{K} =====")

    fold_root = os.path.join(OUTPUT_PATH, f"fold_{fold}")

    fold_img_train = os.path.join(fold_root, "images", "train")
    fold_img_val = os.path.join(fold_root, "images", "val")
    fold_lbl_train = os.path.join(fold_root, "labels", "train")
    fold_lbl_val = os.path.join(fold_root, "labels", "val")

    for d in [fold_img_train, fold_img_val, fold_lbl_train, fold_lbl_val]:
        os.makedirs(d, exist_ok=True)

    # =========================
    # COPY TRAIN FOLD DATA
    # =========================
    for i in train_idx:
        sample = samples[i]

        shutil.copy2(sample["image_path"], os.path.join(fold_img_train, sample["image_name"]))
        shutil.copy2(sample["label_path"], os.path.join(fold_lbl_train, sample["label_name"]))

    # =========================
    # COPY VAL FOLD DATA
    # =========================
    for i in val_idx:
        sample = samples[i]

        shutil.copy2(sample["image_path"], os.path.join(fold_img_val, sample["image_name"]))
        shutil.copy2(sample["label_path"], os.path.join(fold_lbl_val, sample["label_name"]))

    print(f"Train images: {len(train_idx)}")
    print(f"Val images  : {len(val_idx)}")

    # =========================
    # CREATE YAML
    # =========================
    fold_yaml = os.path.join(fold_root, "plastic_fold.yaml")

    with open(fold_yaml, "w", encoding="utf-8") as f:
        f.write(f"""path: {fold_root}
train: images/train
val: images/val

names:
  0: plastic
""")

    # =========================
    # TRAIN MODEL
    # =========================
    model = YOLO(MODEL_PATH)

    results = model.train(
        data=fold_yaml,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=fold_root,
        name=f"yolo26n_fold_{fold}",
        exist_ok=True
    )

    # =========================
    # EXTRACT METRICS
    # =========================
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)
    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    map5095 = results.results_dict.get("metrics/mAP50-95(B)", 0)

    results_summary.append({
        "fold": fold,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map5095
    })

    print(
        f"Fold {fold} Result | "
        f"P: {precision:.4f} | "
        f"R: {recall:.4f} | "
        f"mAP50: {map50:.4f} | "
        f"mAP50-95: {map5095:.4f}"
    )

    fold += 1

# =========================
# FINAL SUMMARY
# =========================
precisions = [r["precision"] for r in results_summary]
recalls = [r["recall"] for r in results_summary]
map50s = [r["mAP50"] for r in results_summary]
map5095s = [r["mAP50-95"] for r in results_summary]

summary_path = os.path.join(OUTPUT_PATH, "crossval_summary.csv")

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "precision", "recall", "mAP50", "mAP50-95"])

    for r in results_summary:
        writer.writerow([
            r["fold"],
            r["precision"],
            r["recall"],
            r["mAP50"],
            r["mAP50-95"]
        ])

    writer.writerow([])
    writer.writerow(["mean", np.mean(precisions), np.mean(recalls), np.mean(map50s), np.mean(map5095s)])
    writer.writerow(["std", np.std(precisions), np.std(recalls), np.std(map50s), np.std(map5095s)])

print("\n===== CLEAN CROSS-VALIDATION FINAL SUMMARY =====")
print(f"Precision : {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall    : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"mAP@0.5   : {np.mean(map50s):.4f} ± {np.std(map50s):.4f}")
print(f"mAP@0.5:95: {np.mean(map5095s):.4f} ± {np.std(map5095s):.4f}")

print("\nPer-fold results:")
for r in results_summary:
    print(
        f"Fold {r['fold']}: "
        f"P={r['precision']:.4f} | "
        f"R={r['recall']:.4f} | "
        f"mAP50={r['mAP50']:.4f} | "
        f"mAP50-95={r['mAP50-95']:.4f}"
    )

print("\nSaved summary to:", summary_path)