import os
import shutil
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
DATASET_PATH = r"D:/PLADICS/datasets_clean/merged_clean_dataset"
OUTPUT_PATH = r"D:/PLADICS/experiments_crossval_clean"

MODEL_PATH = r"D:/PLADICS/yolo26n.pt"

K = 4
TARGET_FOLD = 4          # Only Fold 4 will run
EPOCHS = 50
IMGSZ = 640
BATCH = 8
DEVICE = "cpu"
WORKERS = 0

# If Fold 4 has an incomplete last.pt, resume from it.
# If last.pt is missing, Fold 4 will be trained fresh.
RESUME_IF_LAST_PT_EXISTS = True

# If Fold 4 does not have last.pt and is incomplete, old Fold 4 folder will be deleted
# before starting fresh. Fold 1, Fold 2, Fold 3 will never be deleted by this script.
DELETE_INCOMPLETE_TARGET_FOLD_WHEN_NO_CHECKPOINT = True

# =========================
# PATHS
# =========================
images_dir = os.path.join(DATASET_PATH, "images")
labels_dir = os.path.join(DATASET_PATH, "labels")

os.makedirs(OUTPUT_PATH, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def normalize_row(row):
    """Strip spaces from CSV column names."""
    return {str(k).strip(): v for k, v in row.items()}


def read_results_rows(results_csv):
    if not os.path.exists(results_csv):
        return []

    with open(results_csv, "r", encoding="utf-8") as f:
        return [normalize_row(row) for row in csv.DictReader(f)]


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def get_last_metrics_from_results_csv(results_csv):
    rows = read_results_rows(results_csv)
    if not rows:
        return None

    last = rows[-1]
    return {
        "epoch_rows": len(rows),
        "last_epoch": last.get("epoch", ""),
        "precision": safe_float(last.get("metrics/precision(B)", 0)),
        "recall": safe_float(last.get("metrics/recall(B)", 0)),
        "mAP50": safe_float(last.get("metrics/mAP50(B)", 0)),
        "mAP50-95": safe_float(last.get("metrics/mAP50-95(B)", 0)),
    }


def is_fold_complete(fold):
    run_dir = os.path.join(OUTPUT_PATH, f"fold_{fold}", f"yolo26n_fold_{fold}")
    results_csv = os.path.join(run_dir, "results.csv")
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    rows = read_results_rows(results_csv)
    return len(rows) >= EPOCHS and os.path.exists(best_pt) and os.path.exists(last_pt)


def prepare_fold_data(fold, train_idx, val_idx, samples, delete_existing=False):
    fold_root = os.path.join(OUTPUT_PATH, f"fold_{fold}")

    if delete_existing and os.path.exists(fold_root):
        print(f"Deleting old incomplete Fold {fold} folder: {fold_root}")
        shutil.rmtree(fold_root)

    fold_img_train = os.path.join(fold_root, "images", "train")
    fold_img_val = os.path.join(fold_root, "images", "val")
    fold_lbl_train = os.path.join(fold_root, "labels", "train")
    fold_lbl_val = os.path.join(fold_root, "labels", "val")

    for d in [fold_img_train, fold_img_val, fold_lbl_train, fold_lbl_val]:
        os.makedirs(d, exist_ok=True)

    # Copy train fold data. Existing files will be overwritten safely.
    for i in train_idx:
        sample = samples[i]
        shutil.copy2(sample["image_path"], os.path.join(fold_img_train, sample["image_name"]))
        shutil.copy2(sample["label_path"], os.path.join(fold_lbl_train, sample["label_name"]))

    # Copy validation fold data. Existing files will be overwritten safely.
    for i in val_idx:
        sample = samples[i]
        shutil.copy2(sample["image_path"], os.path.join(fold_img_val, sample["image_name"]))
        shutil.copy2(sample["label_path"], os.path.join(fold_lbl_val, sample["label_name"]))

    fold_yaml = os.path.join(fold_root, "plastic_fold.yaml")
    with open(fold_yaml, "w", encoding="utf-8") as f:
        f.write(f"""path: {fold_root}
train: images/train
val: images/val

names:
  0: plastic
""")

    print(f"Train images: {len(train_idx)}")
    print(f"Val images  : {len(val_idx)}")
    print(f"Fold YAML   : {fold_yaml}")

    return fold_root, fold_yaml


# =========================
# COLLECT TRAIN + VAL ONLY
# TEST SET UNTOUCHED
# =========================
samples = []

for split in ["train", "val"]:
    img_dir = os.path.join(images_dir, split)
    lbl_dir = os.path.join(labels_dir, split)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"Label folder not found: {lbl_dir}")

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(IMAGE_EXTENSIONS):
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
                "original_split": split,
            })
        else:
            print(f"Warning: label missing for image, skipped: {img_path}")

print(f"Total train+val samples for K-fold: {len(samples)}")

if len(samples) == 0:
    raise ValueError("No samples found. Check DATASET_PATH.")

if TARGET_FOLD < 1 or TARGET_FOLD > K:
    raise ValueError(f"TARGET_FOLD must be between 1 and {K}. Current TARGET_FOLD={TARGET_FOLD}")

# =========================
# K-FOLD SETUP
# =========================
kfold = KFold(n_splits=K, shuffle=True, random_state=42)

# =========================
# RUN ONLY TARGET FOLD
# =========================
for fold, (train_idx, val_idx) in enumerate(kfold.split(samples), start=1):
    if fold != TARGET_FOLD:
        print(f"Skipping Fold {fold}/{K}. Existing folder will not be touched.")
        continue

    print(f"\n===== Fold {fold}/{K} =====")

    fold_root = os.path.join(OUTPUT_PATH, f"fold_{fold}")
    run_dir = os.path.join(fold_root, f"yolo26n_fold_{fold}")
    results_csv = os.path.join(run_dir, "results.csv")
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    best_pt = os.path.join(run_dir, "weights", "best.pt")

    if is_fold_complete(fold):
        print(f"Fold {fold} is already complete. Training skipped.")
        metrics = get_last_metrics_from_results_csv(results_csv)
        if metrics:
            print(
                f"Fold {fold} Final Result | "
                f"P: {metrics['precision']:.4f} | "
                f"R: {metrics['recall']:.4f} | "
                f"mAP50: {metrics['mAP50']:.4f} | "
                f"mAP50-95: {metrics['mAP50-95']:.4f}"
            )
        break

    resume_available = os.path.exists(last_pt)

    if resume_available and RESUME_IF_LAST_PT_EXISTS:
        print(f"Resume checkpoint found: {last_pt}")
        print("Preparing Fold 4 data/yaml without deleting the existing checkpoint...")
        prepare_fold_data(fold, train_idx, val_idx, samples, delete_existing=False)

        print("Resuming training from last.pt...")
        model = YOLO(last_pt)
        model.train(resume=True)

    else:
        if not resume_available:
            print(f"No resume checkpoint found at: {last_pt}")
            print("Fold 4 cannot be resumed. Starting Fold 4 fresh from the base model.")

        delete_existing = DELETE_INCOMPLETE_TARGET_FOLD_WHEN_NO_CHECKPOINT
        fold_root, fold_yaml = prepare_fold_data(
            fold,
            train_idx,
            val_idx,
            samples,
            delete_existing=delete_existing,
        )

        print("Starting fresh Fold 4 training...")
        model = YOLO(MODEL_PATH)
        model.train(
            data=fold_yaml,
            imgsz=IMGSZ,
            epochs=EPOCHS,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=fold_root,
            name=f"yolo26n_fold_{fold}",
            exist_ok=True,
        )

    # Print target fold result after training/resume
    if os.path.exists(results_csv):
        metrics = get_last_metrics_from_results_csv(results_csv)
        if metrics:
            print(
                f"\nFold {fold} Current/Final Result | "
                f"Epoch rows: {metrics['epoch_rows']} | "
                f"Last epoch: {metrics['last_epoch']} | "
                f"P: {metrics['precision']:.4f} | "
                f"R: {metrics['recall']:.4f} | "
                f"mAP50: {metrics['mAP50']:.4f} | "
                f"mAP50-95: {metrics['mAP50-95']:.4f}"
            )

    print(f"Best weight path: {best_pt}")
    print(f"Last weight path: {last_pt}")

# =========================
# REBUILD SUMMARY FROM ALL AVAILABLE FOLDS
# =========================
results_summary = []

for fold in range(1, K + 1):
    run_dir = os.path.join(OUTPUT_PATH, f"fold_{fold}", f"yolo26n_fold_{fold}")
    results_csv = os.path.join(run_dir, "results.csv")

    metrics = get_last_metrics_from_results_csv(results_csv)
    if metrics is None:
        print(f"Fold {fold}: results.csv not found or empty. Summary skipped for this fold.")
        continue

    status = "COMPLETE" if metrics["epoch_rows"] >= EPOCHS else "INCOMPLETE"
    results_summary.append({
        "fold": fold,
        "status": status,
        "epoch_rows": metrics["epoch_rows"],
        "last_epoch": metrics["last_epoch"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
    })

summary_path = os.path.join(OUTPUT_PATH, "crossval_summary.csv")

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "fold",
        "status",
        "epoch_rows",
        "last_epoch",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
    ])

    for r in results_summary:
        writer.writerow([
            r["fold"],
            r["status"],
            r["epoch_rows"],
            r["last_epoch"],
            r["precision"],
            r["recall"],
            r["mAP50"],
            r["mAP50-95"],
        ])

    completed_results = [r for r in results_summary if r["status"] == "COMPLETE"]

    if completed_results:
        precisions = [r["precision"] for r in completed_results]
        recalls = [r["recall"] for r in completed_results]
        map50s = [r["mAP50"] for r in completed_results]
        map5095s = [r["mAP50-95"] for r in completed_results]

        writer.writerow([])
        writer.writerow([
            "mean_completed_folds_only",
            "",
            "",
            "",
            np.mean(precisions),
            np.mean(recalls),
            np.mean(map50s),
            np.mean(map5095s),
        ])
        writer.writerow([
            "std_completed_folds_only",
            "",
            "",
            "",
            np.std(precisions),
            np.std(recalls),
            np.std(map50s),
            np.std(map5095s),
        ])

print("\n===== CROSS-VALIDATION SUMMARY FROM AVAILABLE FOLDS =====")

for r in results_summary:
    print(
        f"Fold {r['fold']} [{r['status']} | rows={r['epoch_rows']} | last_epoch={r['last_epoch']}]: "
        f"P={r['precision']:.4f} | "
        f"R={r['recall']:.4f} | "
        f"mAP50={r['mAP50']:.4f} | "
        f"mAP50-95={r['mAP50-95']:.4f}"
    )

completed_results = [r for r in results_summary if r["status"] == "COMPLETE"]
if completed_results:
    precisions = [r["precision"] for r in completed_results]
    recalls = [r["recall"] for r in completed_results]
    map50s = [r["mAP50"] for r in completed_results]
    map5095s = [r["mAP50-95"] for r in completed_results]

    print("\nMean/Std calculated from completed folds only:")
    print(f"Precision : {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall    : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"mAP@0.5   : {np.mean(map50s):.4f} ± {np.std(map50s):.4f}")
    print(f"mAP@0.5:95: {np.mean(map5095s):.4f} ± {np.std(map5095s):.4f}")
else:
    print("No completed folds found yet. Mean/std not calculated.")

print("\nSaved summary to:", summary_path)