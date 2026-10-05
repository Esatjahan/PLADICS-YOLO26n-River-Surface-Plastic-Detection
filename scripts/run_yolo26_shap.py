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
INPUT_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\shap_final"

ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "original")
DETECTION_DIR = os.path.join(OUTPUT_DIR, "detection")
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap_map")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlay")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")

for d in [OUTPUT_DIR, ORIGINAL_DIR, DETECTION_DIR, SHAP_DIR, OVERLAY_DIR, COMBINED_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "shap_summary.csv")

# =========================
# SETTINGS
# =========================
IMG_SIZE = 416
CONF_THRES = 0.25
MIN_EXPLAIN_CONF = 0.60
MAX_IMAGES = 21
MAX_EVALS = 1200
BATCH_SIZE = 2

MASKER_BLUR = "blur(8,8)"
OVERLAY_STRENGTH = 0.30
GAUSSIAN_SIGMA = 2.5
DPI = 600
BOX_PAD = 3

print("=" * 80)
print("FINAL YOLO26n DETECTION-SPECIFIC SHAP SCRIPT")
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("INPUT_EXISTS:", os.path.exists(INPUT_DIR))
print("=" * 80)

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(INPUT_DIR)

model = YOLO(MODEL_PATH)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance_score(ref_box, box):
    rx = (ref_box[0] + ref_box[2]) / 2
    ry = (ref_box[1] + ref_box[3]) / 2
    bx = (box[0] + box[2]) / 2
    by = (box[1] + box[3]) / 2

    dist = np.sqrt((rx - bx) ** 2 + (ry - by) ** 2)
    diag = np.sqrt(IMG_SIZE ** 2 + IMG_SIZE ** 2)

    return max(0.0, 1.0 - dist / diag)


def get_detections(img_rgb):
    result = model.predict(
        source=img_rgb,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.array([]), np.array([])

    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


def choose_target(boxes, confs):
    best_idx = None
    best_score = -1.0

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < MIN_EXPLAIN_CONF:
            continue

        x1, y1, x2, y2 = box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        rel_area = area / (IMG_SIZE * IMG_SIZE)

        score = float(conf) * np.sqrt(rel_area)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def padded_box(box, pad=BOX_PAD):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(IMG_SIZE - 1, x2 + pad)
    y2 = min(IMG_SIZE - 1, y2 + pad)
    return x1, y1, x2, y2


def draw_boxes(img_rgb, boxes, confs, target_idx=None, only_target=False):
    out = img_rgb.copy()

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if only_target and i != target_idx:
            continue

        x1, y1, x2, y2 = padded_box(box, pad=BOX_PAD)

        selected = (i == target_idx)
        color = (0, 255, 0) if selected else (0, 210, 0)
        thickness = 3 if selected else 2

        label = f"plastic {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        font_scale = 0.55
        font_thickness = 2
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )[0]

        tx = min(max(x1, 8), IMG_SIZE - text_size[0] - 8)
        ty = y1 - 10 if y1 > 24 else y2 + 18
        ty = min(max(ty, 18), IMG_SIZE - 8)

        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA
        )

    return out


def make_target_score_function(ref_box):
    def score_fn(batch_images):
        scores = []

        for img in batch_images:
            img = np.clip(img, 0, 255).astype(np.uint8)
            boxes, confs = get_detections(img)

            if len(boxes) == 0:
                scores.append([0.0])
                continue

            best = 0.0

            for box, conf in zip(boxes, confs):
                overlap = iou(ref_box, box)
                center_score = center_distance_score(ref_box, box)

                match_score = 0.75 * overlap + 0.25 * center_score

                if match_score > 0.10:
                    best = max(best, float(conf) * match_score)

            scores.append([best])

        return np.array(scores)

    return score_fn


def extract_map(shap_values):
    values = shap_values.values[0]

    if values.ndim == 4:
        values = values[..., 0]

    if values.ndim == 3:
        values = np.sum(values, axis=2)

    return np.nan_to_num(values)


def normalize_signed(x):
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
    x = cv2.GaussianBlur(x, (0, 0), sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)

    vmax = np.percentile(np.abs(x), 99)

    if vmax <= 0:
        return np.zeros_like(x)

    x = np.clip(x, -vmax, vmax) / vmax
    return x


def overlay_signed(img_rgb, signed_map):
    cmap = plt.get_cmap("RdBu_r")
    color = cmap((signed_map + 1.0) / 2.0)[:, :, :3]
    color = (color * 255).astype(np.uint8)

    strength = np.abs(signed_map)
    alpha = np.expand_dims(OVERLAY_STRENGTH * strength, axis=2)

    out = img_rgb * (1 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_shap_map(signed_map, save_path):
    plt.figure(figsize=(5.2, 5.2))
    im = plt.imshow(signed_map, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.axis("off")

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("SHAP contribution", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_combined(original, detection, signed_map, overlay, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(original)
    axes[0].set_title("(a) Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(detection)
    axes[1].set_title("(b) YOLO26n Detection", fontsize=11)
    axes[1].axis("off")

    im = axes[2].imshow(signed_map, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("(c) Detection-specific SHAP Attribution", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("(d) SHAP Attribution Overlay", fontsize=11)
    axes[3].axis("off")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("SHAP contribution", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close()


image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:MAX_IMAGES]

print(f"Processing {len(image_files)} images...")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "detections", "target_confidence", "target_box", "status"])

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        bgr = cv2.imread(img_path)

        if bgr is None:
            print("[ERROR] Cannot read:", img_name)
            writer.writerow([img_name, 0, 0, "", "read_error"])
            continue

        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        boxes, confs = get_detections(rgb)

        cv2.imwrite(
            os.path.join(ORIGINAL_DIR, f"{base}_original.png"),
            bgr
        )

        if len(boxes) == 0:
            print("[SKIP] No detection:", img_name)
            writer.writerow([img_name, 0, 0, "", "no_detection"])
            continue

        target_idx = choose_target(boxes, confs)

        if target_idx is None:
            print("[SKIP] No confident target:", img_name)
            writer.writerow([
                img_name,
                len(boxes),
                float(np.max(confs)),
                "",
                "no_confident_target"
            ])
            continue

        target_box = boxes[target_idx]
        target_conf = float(confs[target_idx])

        detection_img = draw_boxes(
            rgb,
            boxes,
            confs,
            target_idx=target_idx,
            only_target=False
        )

        cv2.imwrite(
            os.path.join(DETECTION_DIR, f"{base}_detection.png"),
            cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
        )

        print(f"[SHAP RUNNING] {img_name} | plastic_conf={target_conf:.3f}")

        try:
            masker = shap.maskers.Image(
                MASKER_BLUR,
                (IMG_SIZE, IMG_SIZE, 3)
            )

            explainer = shap.Explainer(
                make_target_score_function(target_box),
                masker,
                output_names=["plastic_confidence"]
            )

            shap_values = explainer(
                np.array([rgb]),
                max_evals=MAX_EVALS,
                batch_size=BATCH_SIZE
            )

            raw_map = extract_map(shap_values)
            signed_map = normalize_signed(raw_map)
            
            np.save(
               os.path.join(SHAP_DIR, f"{base}_raw_map.npy"),
               raw_map
            )
            
            np.save(
               os.path.join(SHAP_DIR, f"{base}_signed_map.npy"),
               signed_map
            )

            overlay = overlay_signed(rgb, signed_map)
            overlay = draw_boxes(
                overlay,
                boxes,
                confs,
                target_idx=target_idx,
                only_target=True
            )

            save_shap_map(
                signed_map,
                os.path.join(SHAP_DIR, f"{base}_shap.png")
            )

            cv2.imwrite(
                os.path.join(OVERLAY_DIR, f"{base}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

            save_combined(
                rgb,
                detection_img,
                signed_map,
                overlay,
                os.path.join(COMBINED_DIR, f"{base}_combined.png")
            )

            writer.writerow([
                img_name,
                len(boxes),
                target_conf,
                target_box.tolist(),
                "success"
            ])

            print("[DONE]", img_name)

        except Exception as e:
            writer.writerow([
                img_name,
                len(boxes),
                target_conf,
                target_box.tolist(),
                f"failed: {e}"
            ])

            print("[FAILED]", img_name, "|", e)

print("\nDone.")
print("Saved to:", OUTPUT_DIR)
print("Summary:", CSV_PATH)