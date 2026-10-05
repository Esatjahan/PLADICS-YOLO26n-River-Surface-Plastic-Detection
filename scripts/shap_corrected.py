import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import shap
from ultralytics import YOLO

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"

INPUT_DIR = r"D:\PLADICS\explainability\shap\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\shap"

ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "original")
DETECTION_DIR = os.path.join(OUTPUT_DIR, "detection")
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap_map")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "shap_summary.csv")

# =========================
# SETTINGS
# =========================
SHAP_IMAGE_SIZE = 416
CONF_THRES = 0.25

MAX_IMAGES = 21

# Trial: 200
# Final paper figure: 500 or 800 if time allows
MAX_EVALS = 500
BATCH_SIZE = 2

MASKER_BLUR = "blur(16,16)"   # better than blur(32,32); less blocky
OVERLAY_ALPHA = 0.32          # lower alpha = original image more visible

print("Loading YOLO26n model...")
model = YOLO(MODEL_PATH)


def draw_boxes(img_rgb, boxes, confs):
    out = img_rgb.copy()

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"plastic {conf:.2f}",
            (x1, max(y1 - 8, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return out


def get_detection(img_rgb):
    result = model.predict(
        source=img_rgb,
        imgsz=SHAP_IMAGE_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.array([]), np.array([])

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    return boxes, confs


def yolo_confidence_function(batch_images):
    """
    SHAP target:
    maximum detected plastic confidence score.

    This explains which image regions increase or decrease
    the final plastic-confidence response of YOLO26n.
    """
    outputs = []

    for img in batch_images:
        img = np.clip(img, 0, 255).astype(np.uint8)

        boxes, confs = get_detection(img)

        if len(confs) == 0:
            outputs.append([0.0])
        else:
            outputs.append([float(np.max(confs))])

    return np.array(outputs)


def extract_signed_shap_map(shap_values):
    """
    Extract signed SHAP map.
    Positive values increase plastic confidence.
    Negative values decrease plastic confidence.
    """
    values = shap_values.values

    # Remove batch dimension
    values = values[0]

    # Remove output dimension if exists
    if values.ndim == 4:
        values = values[..., 0]

    # RGB channels -> spatial attribution map
    if values.ndim == 3:
        values = np.mean(values, axis=2)

    values = np.nan_to_num(values)
    return values


def normalize_signed_map(signed_map):
    """
    Normalize SHAP map symmetrically to [-1, 1].
    """
    vmax = np.percentile(np.abs(signed_map), 98)

    if vmax <= 0:
        return np.zeros_like(signed_map)

    signed_norm = np.clip(signed_map, -vmax, vmax) / vmax
    return signed_norm


def smooth_shap_map(signed_map):
    """
    Smooth for publication-quality visualization.
    """
    signed_map = cv2.resize(signed_map, (SHAP_IMAGE_SIZE, SHAP_IMAGE_SIZE))
    signed_map = cv2.GaussianBlur(signed_map, (0, 0), sigmaX=5, sigmaY=5)
    return signed_map


def create_signed_shap_overlay(img_rgb, signed_norm):
    """
    Create red-blue SHAP overlay.
    Red = positive contribution.
    Blue = negative contribution.
    """
    cmap = plt.get_cmap("bwr")
    heat_rgb = cmap((signed_norm + 1) / 2.0)[:, :, :3]
    heat_rgb = (heat_rgb * 255).astype(np.uint8)

    strength = np.abs(signed_norm)
    strength = np.expand_dims(strength, axis=2)

    blended = img_rgb.astype(np.float32) * (1 - OVERLAY_ALPHA * strength) + heat_rgb.astype(np.float32) * (OVERLAY_ALPHA * strength)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def save_shap_only(img_rgb, signed_norm, save_path):
    """
    Save signed SHAP map with image background and colorbar.
    """
    overlay = create_signed_shap_overlay(img_rgb, signed_norm)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SHAP contribution", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def make_combined(original, detection, signed_norm, overlay, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    axes[0].imshow(original)
    axes[0].set_title("(a) Original Test Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(detection)
    axes[1].set_title("(b) YOLO26n Detection", fontsize=12)
    axes[1].axis("off")

    im = axes[2].imshow(signed_norm, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("(c) Signed SHAP Map\nRed: increases confidence, Blue: decreases", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("(d) SHAP Overlay on Image", fontsize=12)
    axes[3].axis("off")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP contribution", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close()


# =========================
# SHAP EXPLAINER
# =========================
masker = shap.maskers.Image(
    MASKER_BLUR,
    (SHAP_IMAGE_SIZE, SHAP_IMAGE_SIZE, 3)
)

explainer = shap.Explainer(
    yolo_confidence_function,
    masker,
    output_names=["plastic_confidence"]
)

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_files = image_files[:MAX_IMAGES]

print(f"Processing {len(image_files)} images...")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "detections", "max_confidence", "status"])

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        bgr = cv2.imread(img_path)

        if bgr is None:
            print("[ERROR] Cannot read:", img_name)
            writer.writerow([img_name, 0, 0, "read_error"])
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (SHAP_IMAGE_SIZE, SHAP_IMAGE_SIZE))

        boxes, confs = get_detection(rgb)

        cv2.imwrite(
            os.path.join(ORIGINAL_DIR, f"{base}_original.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

        if len(boxes) == 0:
            print("[SKIP] No detection:", img_name)
            writer.writerow([img_name, 0, 0, "no_detection"])
            continue

        max_conf = float(np.max(confs))
        detection_img = draw_boxes(rgb, boxes, confs)

        cv2.imwrite(
            os.path.join(DETECTION_DIR, f"{base}_detection.jpg"),
            cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
        )

        print(f"[SHAP RUNNING] {img_name} | max_conf={max_conf:.3f}")

        try:
            shap_values = explainer(
                np.array([rgb]),
                max_evals=MAX_EVALS,
                batch_size=BATCH_SIZE
            )

            signed_map = extract_signed_shap_map(shap_values)
            signed_map = smooth_shap_map(signed_map)
            signed_norm = normalize_signed_map(signed_map)

            overlay = create_signed_shap_overlay(rgb, signed_norm)
            overlay_with_boxes = draw_boxes(overlay, boxes, confs)

            save_shap_only(
                rgb,
                signed_norm,
                os.path.join(SHAP_DIR, f"{base}_shap.jpg")
            )

            make_combined(
                rgb,
                detection_img,
                signed_norm,
                overlay_with_boxes,
                os.path.join(COMBINED_DIR, f"{base}_combined.jpg")
            )

            writer.writerow([img_name, len(boxes), max_conf, "success"])
            print(f"[DONE] {img_name}")

        except Exception as e:
            writer.writerow([img_name, len(boxes), max_conf, f"failed: {e}"])
            print(f"[FAILED] {img_name} | {e}")

print("\nDone.")
print("Saved to:", OUTPUT_DIR)
print("Summary:", CSV_PATH)