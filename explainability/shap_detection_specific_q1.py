"""
PLADICS Q1-grade detection-specific SHAP analysis.

Purpose
-------
Explain the confidence of a fixed YOLO26n plastic detection by anchoring every
perturbed prediction to the original target box with IoU-based matching.

The script:
1. preserves the existing model and sample-image paths;
2. creates a timestamped output directory, so previous SHAP outputs are never
   overwritten;
3. uses deterministic image ordering and fixed seeds;
4. separates raw attribution values from visualization post-processing;
5. computes reproducible faithfulness and localization metrics;
6. saves raw SHAP arrays, per-image metrics, summary statistics, metadata, and
   publication-ready figures.

Important interpretation
------------------------
The output explains the confidence of the original highest-confidence plastic
box while a spatially matching detection remains available. It does not prove
causal reasoning, and deletion tests can be affected by perturbation artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import ultralytics
from ultralytics import YOLO


# ==========================================================
# REPRODUCIBLE CONFIGURATION
# ==========================================================

MODEL_PATH = Path(
    r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
)

# Keep the existing input path unchanged.
INPUT_DIR = Path(r"D:\PLADICS\explainability\sample_images")

# Keep the same SHAP root, but write every run into a new timestamped folder.
OUTPUT_ROOT = Path(r"D:\PLADICS\explainability\shap")
RUN_PREFIX = "q1_detection_specific_v2"

IMAGE_SIZE = 416
TARGET_SELECTION_CONF = 0.25
ATTRIBUTION_CONF_FLOOR = 0.01
TARGET_MATCH_IOU = 0.30
TARGET_CLASS_NAME = "plastic"

# Process all images in sample_images. Set an integer only for a documented
# pilot run; keep None for the final analysis.
MAX_IMAGES: int | None = None

# Official SHAP image examples use the image masker with max_evals and batch
# size. This final setting is intentionally higher than a quick pilot run.
MAX_EVALS = 1000
BATCH_SIZE = 2
EXPLAINER_ALGORITHM = "partition"
MASKER_BLUR = "blur(16,16)"

VISUAL_GAUSSIAN_SIGMA = 5.0
VISUAL_PERCENTILE = 98.0
OVERLAY_ALPHA = 0.32
BOX_EXPANSION_RATIO = 0.10
TOP_POSITIVE_FRACTION = 0.05
DELETION_FRACTIONS = (0.20, 0.40, 0.60)

BOOTSTRAP_ITERATIONS = 2000
SEED = 42
DEVICE = "cpu"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==========================================================
# DATA STRUCTURES
# ==========================================================

@dataclass(frozen=True)
class TargetDetection:
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class RunPaths:
    root: Path
    original: Path
    detection: Path
    shap_map: Path
    overlay: Path
    combined: Path
    deletion: Path
    raw: Path
    logs: Path


# ==========================================================
# GENERAL UTILITIES
# ==========================================================

def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_unique_run_directory(root: Path, prefix: str) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{prefix}_{timestamp}"
    suffix = 1

    while candidate.exists():
        candidate = root / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1

    paths = RunPaths(
        root=candidate,
        original=candidate / "original",
        detection=candidate / "detection",
        shap_map=candidate / "shap_map",
        overlay=candidate / "overlay",
        combined=candidate / "combined",
        deletion=candidate / "deletion",
        raw=candidate / "raw_arrays",
        logs=candidate / "logs",
    )

    for path in asdict(paths).values():
        Path(path).mkdir(parents=True, exist_ok=False)

    return paths


def list_images(directory: Path, limit: int | None) -> list[Path]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files if limit is None else files[:limit]


def letterbox_rgb(
    image_rgb: np.ndarray,
    size: int,
    fill_value: int = 114,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Resize while preserving aspect ratio, then pad to a square canvas."""
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full((size, size, 3), fill_value, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    metadata = {
        "original_width": width,
        "original_height": height,
        "scale": scale,
        "pad_left": left,
        "pad_top": top,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }
    return canvas, metadata


# ==========================================================
# DETECTION AND TARGET MATCHING
# ==========================================================

def model_class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def predict_detections(
    model: YOLO,
    image_rgb: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Ultralytics prediction.

    The analysis image is stored as RGB for SHAP and Matplotlib. Ultralytics is
    given a contiguous BGR NumPy array, matching OpenCV-based input usage.
    """
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])

    result = model.predict(
        source=image_bgr,
        imgsz=IMAGE_SIZE,
        conf=float(confidence),
        device=DEVICE,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    return boxes, confidences, classes


def select_reference_target(
    model: YOLO,
    image_rgb: np.ndarray,
) -> tuple[TargetDetection | None, np.ndarray, np.ndarray, np.ndarray]:
    boxes, confidences, classes = predict_detections(
        model,
        image_rgb,
        TARGET_SELECTION_CONF,
    )

    if len(confidences) == 0:
        return None, boxes, confidences, classes

    normalized_target = TARGET_CLASS_NAME.strip().lower()
    eligible_indices = [
        index
        for index, class_id in enumerate(classes)
        if model_class_name(model.names, int(class_id)).strip().lower()
        == normalized_target
    ]

    if not eligible_indices:
        return None, boxes, confidences, classes

    best_index = max(eligible_indices, key=lambda index: float(confidences[index]))
    class_id = int(classes[best_index])
    class_name = model_class_name(model.names, class_id)
    box = tuple(float(value) for value in boxes[best_index])

    target = TargetDetection(
        box_xyxy=box,
        confidence=float(confidences[best_index]),
        class_id=class_id,
        class_name=class_name,
    )
    return target, boxes, confidences, classes


def iou_with_reference(reference_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(reference_box[0], boxes[:, 0])
    y1 = np.maximum(reference_box[1], boxes[:, 1])
    x2 = np.minimum(reference_box[2], boxes[:, 2])
    y2 = np.minimum(reference_box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    reference_area = max(
        0.0,
        float(reference_box[2] - reference_box[0]),
    ) * max(0.0, float(reference_box[3] - reference_box[1]))
    box_areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )

    union = reference_area + box_areas - intersection
    return intersection / np.maximum(union, 1e-9)


class FixedDetectionScorer:
    """Return confidence for the detection matching one fixed reference box."""

    def __init__(self, model: YOLO, target: TargetDetection) -> None:
        self.model = model
        self.target = target
        self.reference_box = np.asarray(target.box_xyxy, dtype=np.float32)

    def score_image(self, image_rgb: np.ndarray) -> tuple[float, float]:
        boxes, confidences, classes = predict_detections(
            self.model,
            image_rgb,
            ATTRIBUTION_CONF_FLOOR,
        )

        class_mask = classes == self.target.class_id
        boxes = boxes[class_mask]
        confidences = confidences[class_mask]

        if len(confidences) == 0:
            return 0.0, 0.0

        ious = iou_with_reference(self.reference_box, boxes)
        best_index = int(np.argmax(ious))
        best_iou = float(ious[best_index])

        if best_iou < TARGET_MATCH_IOU:
            return 0.0, best_iou

        return float(confidences[best_index]), best_iou

    def __call__(self, batch_images: np.ndarray) -> np.ndarray:
        outputs: list[list[float]] = []
        for image in batch_images:
            score, _ = self.score_image(image)
            outputs.append([score])
        return np.asarray(outputs, dtype=np.float64)


# ==========================================================
# SHAP EXTRACTION AND VISUALIZATION
# ==========================================================

def extract_signed_spatial_map(explanation: shap.Explanation) -> np.ndarray:
    values = np.asarray(explanation.values)

    if values.shape[0] != 1:
        raise ValueError(f"Expected one explained image, received shape {values.shape}")

    values = values[0]

    # Remove output dimensions until spatial RGB dimensions remain.
    while values.ndim > 3:
        values = values[..., 0]

    if values.ndim == 3:
        values = np.mean(values, axis=2)

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape after reduction: {values.shape}")

    return np.nan_to_num(values.astype(np.float64))


def visual_map_from_raw(raw_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(raw_map, (IMAGE_SIZE, IMAGE_SIZE))
    smoothed = cv2.GaussianBlur(
        resized,
        (0, 0),
        sigmaX=VISUAL_GAUSSIAN_SIGMA,
        sigmaY=VISUAL_GAUSSIAN_SIGMA,
    )

    limit = float(np.percentile(np.abs(smoothed), VISUAL_PERCENTILE))
    if limit <= 0:
        normalized = np.zeros_like(smoothed, dtype=np.float64)
    else:
        normalized = np.clip(smoothed, -limit, limit) / limit

    return smoothed, normalized


def create_signed_overlay(image_rgb: np.ndarray, normalized_map: np.ndarray) -> np.ndarray:
    color_map = plt.get_cmap("bwr")
    heat_rgb = color_map((normalized_map + 1.0) / 2.0)[:, :, :3]
    heat_rgb = (heat_rgb * 255.0).astype(np.uint8)

    strength = np.abs(normalized_map)[..., None]
    blended = (
        image_rgb.astype(np.float32) * (1.0 - OVERLAY_ALPHA * strength)
        + heat_rgb.astype(np.float32) * (OVERLAY_ALPHA * strength)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_detections(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    classes: np.ndarray,
    target: TargetDetection,
    names: dict | list,
) -> np.ndarray:
    output = image_rgb.copy()
    reference = np.asarray(target.box_xyxy, dtype=np.float32)
    ious = iou_with_reference(reference, boxes)
    target_index = int(np.argmax(ious)) if len(ious) else -1

    for index, (box, confidence, class_id) in enumerate(
        zip(boxes, confidences, classes)
    ):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        is_target = index == target_index and float(ious[index]) >= 0.99
        color = (255, 215, 0) if is_target else (0, 200, 0)
        thickness = 3 if is_target else 2
        prefix = "TARGET" if is_target else model_class_name(names, int(class_id))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            output,
            f"{prefix} {float(confidence):.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return output


def save_shap_figure(
    image_rgb: np.ndarray,
    normalized_map: np.ndarray,
    save_path: Path,
) -> None:
    overlay = create_signed_overlay(image_rgb, normalized_map)
    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    axis.imshow(overlay)
    axis.axis("off")

    scalar_map = plt.cm.ScalarMappable(
        cmap="bwr",
        norm=plt.Normalize(vmin=-1, vmax=1),
    )
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized signed SHAP attribution", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def save_combined_figure(
    original: np.ndarray,
    detection: np.ndarray,
    raw_map: np.ndarray,
    normalized_visual_map: np.ndarray,
    overlay_with_target: np.ndarray,
    save_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    axes[0].imshow(original)
    axes[0].set_title("(a) Letterboxed test image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(detection)
    axes[1].set_title("(b) Reference detection", fontsize=11)
    axes[1].axis("off")

    maximum = float(np.percentile(np.abs(raw_map), VISUAL_PERCENTILE))
    maximum = maximum if maximum > 0 else 1.0
    image_handle = axes[2].imshow(raw_map, cmap="bwr", vmin=-maximum, vmax=maximum)
    axes[2].set_title("(c) Raw signed attribution", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(overlay_with_target)
    axes[3].set_title("(d) Smoothed overlay", fontsize=11)
    axes[3].axis("off")

    colorbar = figure.colorbar(image_handle, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("Raw signed SHAP value", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


# ==========================================================
# QUANTITATIVE ATTRIBUTION METRICS
# ==========================================================

def box_to_mask(
    box_xyxy: Iterable[float],
    height: int,
    width: int,
    expansion_ratio: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)

    x1 -= box_width * expansion_ratio
    x2 += box_width * expansion_ratio
    y1 -= box_height * expansion_ratio
    y2 += box_height * expansion_ratio

    x1_i = max(0, min(width - 1, int(math.floor(x1))))
    y1_i = max(0, min(height - 1, int(math.floor(y1))))
    x2_i = max(x1_i + 1, min(width, int(math.ceil(x2))))
    y2_i = max(y1_i + 1, min(height, int(math.ceil(y2))))

    mask = np.zeros((height, width), dtype=bool)
    mask[y1_i:y2_i, x1_i:x2_i] = True
    return mask, (x1_i, y1_i, x2_i, y2_i)


def top_positive_mask(positive_map: np.ndarray, fraction: float) -> np.ndarray:
    flat = positive_map.ravel()
    positive_indices = np.flatnonzero(flat > 0)
    mask = np.zeros_like(flat, dtype=bool)

    if len(positive_indices) == 0:
        return mask.reshape(positive_map.shape)

    count = max(1, int(math.ceil(len(positive_indices) * fraction)))
    positive_values = flat[positive_indices]
    selected_local = np.argpartition(positive_values, -count)[-count:]
    selected_global = positive_indices[selected_local]
    mask[selected_global] = True
    return mask.reshape(positive_map.shape)


def fraction_inside(region_mask: np.ndarray, box_mask: np.ndarray) -> float:
    denominator = int(region_mask.sum())
    if denominator == 0:
        return float("nan")
    return float(np.logical_and(region_mask, box_mask).sum() / denominator)


def attribution_mass_inside(attribution: np.ndarray, box_mask: np.ndarray) -> float:
    total = float(attribution.sum())
    if total <= 0:
        return float("nan")
    return float(attribution[box_mask].sum() / total)


def weighted_centroid_inside(
    positive_map: np.ndarray,
    box_mask: np.ndarray,
) -> tuple[float, float, float]:
    total = float(positive_map.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")

    y_coordinates, x_coordinates = np.indices(positive_map.shape)
    centroid_x = float((x_coordinates * positive_map).sum() / total)
    centroid_y = float((y_coordinates * positive_map).sum() / total)

    x_index = int(np.clip(round(centroid_x), 0, positive_map.shape[1] - 1))
    y_index = int(np.clip(round(centroid_y), 0, positive_map.shape[0] - 1))
    inside = float(box_mask[y_index, x_index])
    return centroid_x, centroid_y, inside


def pointing_game(positive_map: np.ndarray, box_mask: np.ndarray) -> float:
    if float(positive_map.max()) <= 0:
        return float("nan")
    y_index, x_index = np.unravel_index(np.argmax(positive_map), positive_map.shape)
    return float(box_mask[y_index, x_index])


def deletion_metrics(
    image_rgb: np.ndarray,
    raw_map: np.ndarray,
    scorer: FixedDetectionScorer,
    base_score: float,
    save_prefix: Path,
) -> dict[str, float]:
    positive_map = np.clip(raw_map, 0.0, None)
    blur_baseline = cv2.blur(image_rgb, (16, 16))
    metrics: dict[str, float] = {}
    drops: list[float] = []

    for fraction in DELETION_FRACTIONS:
        percentage = int(round(fraction * 100))
        mask = top_positive_mask(positive_map, fraction)

        if not mask.any():
            masked_score = float("nan")
            drop = float("nan")
            relative_drop = float("nan")
        else:
            perturbed = image_rgb.copy()
            perturbed[mask] = blur_baseline[mask]
            masked_score, _ = scorer.score_image(perturbed)
            drop = float(base_score - masked_score)
            relative_drop = (
                float(drop / base_score) if base_score > 0 else float("nan")
            )

            cv2.imwrite(
                str(save_prefix.with_name(f"{save_prefix.name}_top{percentage}.jpg")),
                cv2.cvtColor(perturbed, cv2.COLOR_RGB2BGR),
            )

        metrics[f"masked_score_top{percentage}"] = masked_score
        metrics[f"confidence_drop_top{percentage}"] = drop
        metrics[f"relative_drop_top{percentage}"] = relative_drop
        drops.append(drop)

    valid_drops = [value for value in drops if np.isfinite(value)]
    metrics["deletion_monotonic"] = (
        float(all(a <= b + 1e-12 for a, b in zip(valid_drops, valid_drops[1:])))
        if len(valid_drops) == len(DELETION_FRACTIONS)
        else float("nan")
    )
    return metrics


def compute_spatial_metrics(
    raw_map: np.ndarray,
    target_box: tuple[float, float, float, float],
) -> dict[str, float]:
    raw_map = cv2.resize(raw_map, (IMAGE_SIZE, IMAGE_SIZE))
    positive_map = np.clip(raw_map, 0.0, None)
    absolute_map = np.abs(raw_map)

    original_box_mask, _ = box_to_mask(
        target_box,
        IMAGE_SIZE,
        IMAGE_SIZE,
        expansion_ratio=0.0,
    )
    expanded_box_mask, _ = box_to_mask(
        target_box,
        IMAGE_SIZE,
        IMAGE_SIZE,
        expansion_ratio=BOX_EXPANSION_RATIO,
    )

    top_mask = top_positive_mask(positive_map, TOP_POSITIVE_FRACTION)
    centroid_x, centroid_y, centroid_inside_original = weighted_centroid_inside(
        positive_map,
        original_box_mask,
    )
    _, _, centroid_inside_expanded = weighted_centroid_inside(
        positive_map,
        expanded_box_mask,
    )

    return {
        "positive_mass_inside_box": attribution_mass_inside(
            positive_map,
            original_box_mask,
        ),
        "positive_mass_inside_expanded_box": attribution_mass_inside(
            positive_map,
            expanded_box_mask,
        ),
        "absolute_mass_inside_box": attribution_mass_inside(
            absolute_map,
            original_box_mask,
        ),
        "absolute_mass_inside_expanded_box": attribution_mass_inside(
            absolute_map,
            expanded_box_mask,
        ),
        "top5_positive_inside_box": fraction_inside(top_mask, original_box_mask),
        "top5_positive_inside_expanded_box": fraction_inside(
            top_mask,
            expanded_box_mask,
        ),
        "pointing_game_inside_box": pointing_game(positive_map, original_box_mask),
        "pointing_game_inside_expanded_box": pointing_game(
            positive_map,
            expanded_box_mask,
        ),
        "positive_centroid_x": centroid_x,
        "positive_centroid_y": centroid_y,
        "centroid_inside_box": centroid_inside_original,
        "centroid_inside_expanded_box": centroid_inside_expanded,
    }


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

def bootstrap_mean_interval(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value

    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = generator.choice(values, size=len(values), replace=True)
        means[index] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")

    p = successes / total
    denominator = 1.0 + (z**2 / total)

    centre = (
        p + (z**2 / (2.0 * total))
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (p * (1.0 - p) / total)
            + (z**2 / (4.0 * total**2))
        )
        / denominator
    )

    return centre - half_width, centre + half_width

def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    metric_names = [
        "confidence_drop_top20",
        "confidence_drop_top40",
        "confidence_drop_top60",
        "relative_drop_top20",
        "relative_drop_top40",
        "relative_drop_top60",
        "deletion_monotonic",
        "positive_mass_inside_box",
        "positive_mass_inside_expanded_box",
        "absolute_mass_inside_box",
        "absolute_mass_inside_expanded_box",
        "top5_positive_inside_box",
        "top5_positive_inside_expanded_box",
        "pointing_game_inside_box",
        "pointing_game_inside_expanded_box",
        "centroid_inside_box",
        "centroid_inside_expanded_box",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "n",
                "mean",
                "population_std",
                "median",
                "q1",
                "q3",
                "bootstrap_95ci_low",
                "bootstrap_95ci_high",
            ]
        )

        for metric_index, metric_name in enumerate(metric_names):
            values = np.asarray(
                [
                    float(row[metric_name])
                    for row in rows
                    if metric_name in row
                    and row[metric_name] not in (None, "")
                    and np.isfinite(float(row[metric_name]))
                ],
                dtype=np.float64,
            )

            if len(values) == 0:
                writer.writerow([metric_name, 0, "", "", "", "", "", "", ""])
                continue

            binary_metrics = {
                "deletion_monotonic",
                "pointing_game_inside_box",
                "pointing_game_inside_expanded_box",
                "centroid_inside_box",
                "centroid_inside_expanded_box",
            }
            if metric_name in binary_metrics:
                successes = int(np.sum(values))
                ci_low, ci_high = wilson_interval(
                    successes,
                    len(values),
                )
            else:
                ci_low, ci_high = bootstrap_mean_interval(
                    values,
                    BOOTSTRAP_ITERATIONS,
                    SEED + metric_index,
                )        
            writer.writerow(
                [
                    metric_name,
                    len(values),
                    float(np.mean(values)),
                    float(np.std(values, ddof=0)),
                    float(np.median(values)),
                    float(np.percentile(values, 25)),
                    float(np.percentile(values, 75)),
                    ci_low,
                    ci_high,
                ]
            )


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def main() -> None:
    set_reproducible_seeds(SEED)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    paths = make_unique_run_directory(OUTPUT_ROOT, RUN_PREFIX)
    images = list_images(INPUT_DIR, MAX_IMAGES)

    if not images:
        raise RuntimeError(f"No supported images found in {INPUT_DIR}")

    print("Loading YOLO26n model...")
    model = YOLO(str(MODEL_PATH))
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()

    metadata = {
        "run_started_local": datetime.now().isoformat(timespec="seconds"),
        "script_name": Path(__file__).name,
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "input_directory": str(INPUT_DIR),
        "output_directory": str(paths.root),
        "number_of_candidate_images": len(images),
        "configuration": {
            "image_size": IMAGE_SIZE,
            "target_selection_confidence": TARGET_SELECTION_CONF,
            "attribution_confidence_floor": ATTRIBUTION_CONF_FLOOR,
            "target_match_iou": TARGET_MATCH_IOU,
            "max_images": MAX_IMAGES,
            "max_evals": MAX_EVALS,
            "batch_size": BATCH_SIZE,
            "explainer_algorithm": EXPLAINER_ALGORITHM,
            "masker_blur": MASKER_BLUR,
            "visual_gaussian_sigma": VISUAL_GAUSSIAN_SIGMA,
            "visual_percentile": VISUAL_PERCENTILE,
            "overlay_alpha": OVERLAY_ALPHA,
            "box_expansion_ratio": BOX_EXPANSION_RATIO,
            "top_positive_fraction": TOP_POSITIVE_FRACTION,
            "deletion_fractions": list(DELETION_FRACTIONS),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "seed": SEED,
            "device": DEVICE,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "matplotlib": matplotlib.__version__,
            "shap": shap.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "interpretation": (
            "Detection-specific attribution anchored to the original highest-"
            "confidence plastic box through same-class IoU matching."
        ),
    }

    with (paths.root / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    metric_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, str]] = []

    for image_index, image_path in enumerate(images, start=1):
        base_name = image_path.stem
        print(f"[{image_index}/{len(images)}] {image_path.name}")

        try:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("OpenCV could not read the image")

            original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image_rgb, letterbox_metadata = letterbox_rgb(original_rgb, IMAGE_SIZE)

            target, boxes, confidences, classes = select_reference_target(
                model,
                image_rgb,
            )

            cv2.imwrite(
                str(paths.original / f"{base_name}_original.jpg"),
                cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            )

            if target is None:
                failure_rows.append(
                    {
                        "image": image_path.name,
                        "image_sha256": sha256_file(image_path),
                        "status": "no_target_detection",
                        "message": (
                            f"No {TARGET_CLASS_NAME} detection at confidence "
                            f">= {TARGET_SELECTION_CONF}"
                        ),
                    }
                )
                print("  SKIP: no eligible reference detection")
                continue

            scorer = FixedDetectionScorer(model, target)
            base_score, base_match_iou = scorer.score_image(image_rgb)

            detection_image = draw_detections(
                image_rgb,
                boxes,
                confidences,
                classes,
                target,
                model.names,
            )
            cv2.imwrite(
                str(paths.detection / f"{base_name}_detection.jpg"),
                cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR),
            )

            masker = shap.maskers.Image(
                MASKER_BLUR,
                (IMAGE_SIZE, IMAGE_SIZE, 3),
            )
            explainer = shap.Explainer(
                scorer,
                masker,
                algorithm=EXPLAINER_ALGORITHM,
                output_names=["matched_target_confidence"],
                seed=SEED,
            )

            explanation = explainer(
                np.asarray([image_rgb], dtype=np.uint8),
                max_evals=MAX_EVALS,
                batch_size=BATCH_SIZE,
            )

            raw_map = extract_signed_spatial_map(explanation)
            raw_map = cv2.resize(raw_map, (IMAGE_SIZE, IMAGE_SIZE))
            smoothed_map, normalized_visual_map = visual_map_from_raw(raw_map)

            overlay = create_signed_overlay(image_rgb, normalized_visual_map)
            overlay_with_target = draw_detections(
                overlay,
                boxes,
                confidences,
                classes,
                target,
                model.names,
            )

            cv2.imwrite(
                str(paths.overlay / f"{base_name}_overlay.jpg"),
                cv2.cvtColor(overlay_with_target, cv2.COLOR_RGB2BGR),
            )
            save_shap_figure(
                image_rgb,
                normalized_visual_map,
                paths.shap_map / f"{base_name}_shap.jpg",
            )
            save_combined_figure(
                image_rgb,
                detection_image,
                raw_map,
                normalized_visual_map,
                overlay_with_target,
                paths.combined / f"{base_name}_combined.jpg",
            )

            np.savez_compressed(
                paths.raw / f"{base_name}_shap_raw.npz",
                raw_signed_map=raw_map,
                smoothed_signed_map=smoothed_map,
                normalized_visual_map=normalized_visual_map,
                target_box_xyxy=np.asarray(target.box_xyxy, dtype=np.float32),
                target_confidence=np.asarray(target.confidence, dtype=np.float32),
                base_matched_score=np.asarray(base_score, dtype=np.float32),
                shap_base_values=np.asarray(explanation.base_values),
                shap_values=np.asarray(explanation.values),
            )

            spatial = compute_spatial_metrics(raw_map, target.box_xyxy)
            deletion = deletion_metrics(
                image_rgb,
                raw_map,
                scorer,
                base_score,
                paths.deletion / base_name,
            )

            row: dict[str, object] = {
                "image": image_path.name,
                "image_sha256": sha256_file(image_path),
                "status": "success",
                "number_of_initial_detections": len(boxes),
                "target_class_id": target.class_id,
                "target_class_name": target.class_name,
                "target_confidence": target.confidence,
                "target_x1": target.box_xyxy[0],
                "target_y1": target.box_xyxy[1],
                "target_x2": target.box_xyxy[2],
                "target_y2": target.box_xyxy[3],
                "base_matched_score": base_score,
                "base_match_iou": base_match_iou,
                **letterbox_metadata,
                **spatial,
                **deletion,
            }
            metric_rows.append(row)
            print(
                "  DONE: "
                f"target_conf={target.confidence:.4f}, "
                f"base_matched_score={base_score:.4f}"
            )

        except Exception as error:  # continue remaining images, but preserve error
            failure_rows.append(
                {
                    "image": image_path.name,
                    "image_sha256": sha256_file(image_path),
                    "status": "failed",
                    "message": repr(error),
                }
            )
            print(f"  FAILED: {error}")

    metrics_path = paths.root / "image_metrics.csv"
    if metric_rows:
        all_fields = list(metric_rows[0].keys())
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(metric_rows)
        write_summary_csv(metric_rows, paths.root / "summary_metrics.csv")
    else:
        metrics_path.write_text("status\nno_successful_explanations\n", encoding="utf-8")

    failures_path = paths.root / "failed_or_skipped_images.csv"
    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "image_sha256", "status", "message"],
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    metadata["run_finished_local"] = datetime.now().isoformat(timespec="seconds")
    metadata["successful_explanations"] = len(metric_rows)
    metadata["failed_or_skipped_images"] = len(failure_rows)
    with (paths.root / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nQ1 SHAP analysis completed.")
    print("Output directory:", paths.root)
    print("Successful explanations:", len(metric_rows))
    print("Failed or skipped:", len(failure_rows))
    print("Per-image metrics:", metrics_path)
    print("Summary metrics:", paths.root / "summary_metrics.csv")


if __name__ == "__main__":
    main()