"""
PLADICS Q1-oriented detection-confidence calibration analysis.

Corrections relative to the exploratory script:
1. writes to a unique timestamped run directory;
2. uses deterministic image ordering;
3. sorts predictions by descending confidence before greedy one-to-one matching;
4. records prediction-level matching results;
5. excludes empty bins from the reliability curve;
6. labels the y-axis as empirical precision;
7. saves run metadata and does not overwrite previous outputs.

Interpretation:
The reported ECE is threshold-conditioned because only predictions retained at
confidence >= CONF_THRES are included. Missed ground-truth objects are not part
of this prediction-level ECE.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = Path(
    r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
)
DATASET_ROOT = Path(r"D:\PLADICS\datasets_clean\merged_clean_dataset")

IMAGE_DIR = DATASET_ROOT / "images" / "test"
LABEL_DIR = DATASET_ROOT / "labels" / "test"

# A new timestamped run folder is created below this root.
OUTPUT_ROOT = Path(r"D:\PLADICS\confidence_analysis\calibration_q1_v2")
RUN_PREFIX = "calibration_q1"

IMG_SIZE = 640
CONF_THRES = 0.25
MATCH_IOU_THRES = 0.50
NUM_BINS = 10
DEVICE = "cpu"
SEED = 42

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==========================================================
# UTILITIES
# ==========================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_run_directory() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"{RUN_PREFIX}_{timestamp}"
    suffix = 1

    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"{RUN_PREFIX}_{timestamp}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def yolo_to_xyxy(parts: list[str], img_w: int, img_h: int) -> list[float]:
    _, xc, yc, bw, bh = map(float, parts)
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = max(0.0, float(box1[2] - box1[0])) * max(
        0.0, float(box1[3] - box1[1])
    )
    area2 = max(0.0, float(box2[2] - box2[0])) * max(
        0.0, float(box2[3] - box2[1])
    )

    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[np.ndarray]:
    gt_boxes: list[np.ndarray] = []

    if not label_path.exists():
        return gt_boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            gt_boxes.append(
                np.asarray(yolo_to_xyxy(parts, img_w, img_h), dtype=np.float32)
            )

    return gt_boxes


def list_test_images() -> list[Path]:
    return sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    np.random.seed(SEED)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Test image directory not found: {IMAGE_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Test label directory not found: {LABEL_DIR}")

    run_dir = make_run_directory()
    calibration_csv = run_dir / "calibration_results.csv"
    predictions_csv = run_dir / "prediction_level_results.csv"
    figure_path = run_dir / "yolo26n_reliability_diagram.png"
    metadata_path = run_dir / "run_metadata.json"

    print("Loading YOLO26n model...")
    model = YOLO(str(MODEL_PATH))

    image_files = list_test_images()
    print("Total test images:", len(image_files))

    confidences: list[float] = []
    correctness: list[int] = []
    prediction_rows: list[dict[str, object]] = []

    for image_index, image_path in enumerate(image_files, start=1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIP] Cannot read image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        gt_boxes = load_gt_boxes(label_path, width, height)

        result = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=DEVICE,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        pred_boxes = (
            result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        )
        pred_confs = (
            result.boxes.conf.detach().cpu().numpy().astype(np.float64)
        )

        # Deterministic confidence-descending greedy matching.
        order = np.argsort(-pred_confs, kind="stable")
        pred_boxes = pred_boxes[order]
        pred_confs = pred_confs[order]

        matched_gt: set[int] = set()

        for rank, (pred_box, confidence) in enumerate(
            zip(pred_boxes, pred_confs),
            start=1,
        ):
            best_iou = 0.0
            best_gt_index = -1

            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched_gt:
                    continue

                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            is_correct = 0
            matched_index_for_csv = -1

            if (
                best_gt_index >= 0
                and best_iou >= MATCH_IOU_THRES
            ):
                is_correct = 1
                matched_gt.add(best_gt_index)
                matched_index_for_csv = best_gt_index

            confidences.append(float(confidence))
            correctness.append(is_correct)

            prediction_rows.append(
                {
                    "image": image_path.name,
                    "prediction_rank": rank,
                    "confidence": float(confidence),
                    "correct": is_correct,
                    "best_unmatched_iou": float(best_iou),
                    "matched_gt_index": matched_index_for_csv,
                    "num_ground_truth_boxes": len(gt_boxes),
                    "x1": float(pred_box[0]),
                    "y1": float(pred_box[1]),
                    "x2": float(pred_box[2]),
                    "y2": float(pred_box[3]),
                }
            )

        if image_index % 100 == 0 or image_index == len(image_files):
            print(f"Processed {image_index}/{len(image_files)} images")

    confidence_array = np.asarray(confidences, dtype=np.float64)
    correctness_array = np.asarray(correctness, dtype=np.float64)

    if len(confidence_array) == 0:
        raise RuntimeError(
            "No retained predictions were found. Check the model, paths, "
            "and confidence threshold."
        )

    print("Total retained predictions:", len(confidence_array))

    bin_edges = np.linspace(0.0, 1.0, NUM_BINS + 1)
    bin_confidences = np.full(NUM_BINS, np.nan, dtype=np.float64)
    bin_precisions = np.full(NUM_BINS, np.nan, dtype=np.float64)
    bin_counts = np.zeros(NUM_BINS, dtype=np.int64)
    bin_gaps = np.full(NUM_BINS, np.nan, dtype=np.float64)
    bin_ece_contributions = np.zeros(NUM_BINS, dtype=np.float64)

    ece = 0.0

    for bin_index in range(NUM_BINS):
        low = bin_edges[bin_index]
        high = bin_edges[bin_index + 1]

        if bin_index == NUM_BINS - 1:
            mask = (
                (confidence_array >= low)
                & (confidence_array <= high)
            )
        else:
            mask = (
                (confidence_array >= low)
                & (confidence_array < high)
            )

        count = int(np.sum(mask))
        bin_counts[bin_index] = count

        if count == 0:
            continue

        mean_confidence = float(np.mean(confidence_array[mask]))
        empirical_precision = float(np.mean(correctness_array[mask]))
        gap = abs(empirical_precision - mean_confidence)
        contribution = (count / len(confidence_array)) * gap

        bin_confidences[bin_index] = mean_confidence
        bin_precisions[bin_index] = empirical_precision
        bin_gaps[bin_index] = gap
        bin_ece_contributions[bin_index] = contribution
        ece += contribution

    # Prediction-level audit trail.
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image",
            "prediction_rank",
            "confidence",
            "correct",
            "best_unmatched_iou",
            "matched_gt_index",
            "num_ground_truth_boxes",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    # Bin-level calibration summary.
    with calibration_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bin_start",
                "bin_end",
                "count",
                "mean_confidence",
                "empirical_precision",
                "absolute_gap",
                "ece_contribution",
            ]
        )
        for bin_index in range(NUM_BINS):
            writer.writerow(
                [
                    f"{bin_edges[bin_index]:.2f}",
                    f"{bin_edges[bin_index + 1]:.2f}",
                    int(bin_counts[bin_index]),
                    (
                        ""
                        if not np.isfinite(bin_confidences[bin_index])
                        else f"{bin_confidences[bin_index]:.6f}"
                    ),
                    (
                        ""
                        if not np.isfinite(bin_precisions[bin_index])
                        else f"{bin_precisions[bin_index]:.6f}"
                    ),
                    (
                        ""
                        if not np.isfinite(bin_gaps[bin_index])
                        else f"{bin_gaps[bin_index]:.6f}"
                    ),
                    f"{bin_ece_contributions[bin_index]:.8f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["ECE", f"{ece:.8f}"])
        writer.writerow(
            ["total_retained_predictions", len(confidence_array)]
        )

    populated = bin_counts > 0

    plt.figure(figsize=(6, 6))
    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Perfect calibration",
    )
    plt.plot(
        bin_confidences[populated],
        bin_precisions[populated],
        marker="o",
        label="YOLO26n-100",
    )
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical precision")
    plt.title(f"YOLO26n-100 Reliability Diagram (ECE = {ece:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    metadata = {
        "run_started_local": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "dataset_root": str(DATASET_ROOT),
        "image_directory": str(IMAGE_DIR),
        "label_directory": str(LABEL_DIR),
        "output_directory": str(run_dir),
        "number_of_test_images": len(image_files),
        "number_of_retained_predictions": len(confidence_array),
        "configuration": {
            "input_size": IMG_SIZE,
            "prediction_confidence_threshold": CONF_THRES,
            "matching_iou_threshold": MATCH_IOU_THRES,
            "number_of_equal_width_bins": NUM_BINS,
            "prediction_order": "confidence_descending_stable",
            "matching": (
                "greedy one-to-one matching to the highest-IoU unmatched "
                "ground-truth box"
            ),
            "device": DEVICE,
            "seed": SEED,
            "prediction_nms_iou": (
                "not explicitly overridden; inherited from Ultralytics "
                "version recorded below"
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "matplotlib": matplotlib.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "ece_interpretation": (
            "Threshold-conditioned prediction-level ECE. Only predictions "
            f"retained at confidence >= {CONF_THRES} are included; missed "
            "ground-truth objects are not included."
        ),
        "ece": ece,
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nExpected Calibration Error (ECE): {ece:.6f}")

    target_bin = 8  # [0.80, 0.90)
    if bin_counts[target_bin] > 0:
        print(
            "0.80-0.90 bin | "
            f"n={bin_counts[target_bin]} | "
            f"mean_confidence={bin_confidences[target_bin]:.6f} | "
            f"empirical_precision={bin_precisions[target_bin]:.6f}"
        )
    else:
        print("0.80-0.90 bin is empty.")

    print("\nSaved run directory:", run_dir)
    print("Saved bin-level CSV:", calibration_csv)
    print("Saved prediction-level CSV:", predictions_csv)
    print("Saved reliability figure:", figure_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()