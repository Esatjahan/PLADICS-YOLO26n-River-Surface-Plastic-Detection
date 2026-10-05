import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# PATH CONFIG
# ============================================================
MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
INPUT_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\eigencam_detection_aware"

DETECTION_DIR = os.path.join(OUTPUT_DIR, "detection")
RAW_CAM_DIR = os.path.join(OUTPUT_DIR, "raw_eigencam_map")
GUIDED_CAM_DIR = os.path.join(OUTPUT_DIR, "detection_aware_eigencam_map")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "detection_aware_overlay")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")

for d in [
    OUTPUT_DIR,
    DETECTION_DIR,
    RAW_CAM_DIR,
    GUIDED_CAM_DIR,
    OVERLAY_DIR,
    COMBINED_DIR,
]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "eigencam_detection_aware_summary.csv")

# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE = 640
CONF_THRES = 0.25
MIN_TARGET_CONF = 0.50
MAX_IMAGES = 100

# YOLO26n architecture-specific target layer:
# Layer 22 = final C3k2 feature aggregation block before Detect head.
TARGET_LAYER_INDEX = 22

DPI = 600
BOX_PAD = 3
OVERLAY_ALPHA = 0.32
CAM_SMOOTH_SIGMA = 3.0

# Detection-aware CAM guidance.
# This does not change detection; it only suppresses unrelated background activation
# for clearer object-specific visualization.
ROI_INSIDE_WEIGHT = 1.00
ROI_OUTSIDE_WEIGHT = 0.08
ROI_GAUSSIAN_SIGMA_FACTOR = 0.28

print("=" * 90)
print("YOLO26n Detection-aware EigenCAM Script")
print("MODEL_PATH:", MODEL_PATH)
print("INPUT_DIR :", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("TARGET_LAYER_INDEX:", TARGET_LAYER_INDEX)
print("INPUT_EXISTS:", os.path.exists(INPUT_DIR))
print("=" * 90)

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(INPUT_DIR)

model = YOLO(MODEL_PATH)
target_layer = model.model.model[TARGET_LAYER_INDEX]

print("[INFO] Target layer type:", target_layer.__class__.__name__)


# ============================================================
# ACTIVATION HOOK
# ============================================================
class ActivationHook:
    def __init__(self, layer):
        self.activation = None
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        if isinstance(output, (list, tuple)):
            output = output[0]

        if hasattr(output, "detach"):
            self.activation = output.detach().cpu().numpy()

    def clear(self):
        self.activation = None

    def remove(self):
        self.hook.remove()


hook = ActivationHook(target_layer)


# ============================================================
# DETECTION HELPERS
# ============================================================
def get_detections(img_bgr):
    result = model.predict(
        source=img_bgr,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.array([]), np.array([])

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    return boxes, confs


def choose_target_detection(boxes, confs):
    """
    Selects the most visually useful detection for explanation.
    Score = confidence * sqrt(relative area)
    This avoids selecting tiny edge detections when a clearer target exists.
    """
    best_idx = None
    best_score = -1.0

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < MIN_TARGET_CONF:
            continue

        x1, y1, x2, y2 = box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        rel_area = area / float(IMG_SIZE * IMG_SIZE)
        score = float(conf) * np.sqrt(rel_area)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        best_idx = int(np.argmax(confs))

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

        x1, y1, x2, y2 = padded_box(box)

        selected = target_idx is not None and i == target_idx
        color = (0, 255, 0) if selected else (0, 210, 0)
        thickness = 3 if selected else 2
        label = f"plastic {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        font_scale = 0.60
        font_thickness = 2
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )[0]

        tx = min(max(x1, 8), IMG_SIZE - text_size[0] - 8)
        ty = y1 - 10 if y1 > 24 else y2 + 20
        ty = min(max(ty, 18), IMG_SIZE - 8)

        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# EIGENCAM CORE
# ============================================================
def normalize_01(x):
    x = np.nan_to_num(x)
    x = x - np.min(x)

    max_val = np.max(x)
    if max_val > 0:
        x = x / max_val

    return x


def compute_raw_eigencam(activation):
    """
    EigenCAM from selected YOLO26n feature layer.
    Expected activation shape: [1, C, H, W]
    """
    if activation is None:
        raise RuntimeError("No activation captured from target layer.")

    if activation.ndim != 4:
        raise RuntimeError(f"Unexpected activation shape: {activation.shape}")

    feature = activation[0]  # [C, H, W]
    c, h, w = feature.shape

    feature_flat = feature.reshape(c, h * w).T  # [H*W, C]
    feature_flat = feature_flat - feature_flat.mean(axis=0, keepdims=True)

    try:
        _, _, vh = np.linalg.svd(feature_flat, full_matrices=False)
        pc1 = vh[0]
        cam = np.dot(feature_flat, pc1).reshape(h, w)
    except Exception:
        cam = np.mean(feature, axis=0)

    if abs(cam.min()) > abs(cam.max()):
        cam = -cam

    cam = np.maximum(cam, 0)
    cam = normalize_01(cam)

    cam = cv2.resize(
        cam,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=CAM_SMOOTH_SIGMA,
        sigmaY=CAM_SMOOTH_SIGMA,
    )

    cam = normalize_01(cam)

    return cam


def make_roi_guidance_mask(target_box):
    """
    Creates a soft detection-aware spatial guidance mask.
    The target box region remains high, while unrelated background is suppressed.
    """
    x1, y1, x2, y2 = padded_box(target_box, pad=8)

    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32) * ROI_OUTSIDE_WEIGHT
    mask[y1:y2, x1:x2] = ROI_INSIDE_WEIGHT

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    sigma = max(box_w, box_h) * ROI_GAUSSIAN_SIGMA_FACTOR
    sigma = max(8.0, sigma)

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    gaussian = normalize_01(gaussian)

    guided_mask = np.maximum(mask, gaussian.astype(np.float32))
    guided_mask = normalize_01(guided_mask)

    return guided_mask


def compute_detection_aware_cam(raw_cam, target_box):
    guidance = make_roi_guidance_mask(target_box)

    guided_cam = raw_cam * guidance
    guided_cam = cv2.GaussianBlur(
        guided_cam,
        (0, 0),
        sigmaX=2.0,
        sigmaY=2.0,
    )
    guided_cam = normalize_01(guided_cam)

    return guided_cam


# ============================================================
# VISUALIZATION
# ============================================================
def make_heatmap(cam):
    cam_uint8 = np.uint8(255 * normalize_01(cam))

    try:
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_TURBO)
    except Exception:
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def make_overlay(img_rgb, cam):
    heatmap = make_heatmap(cam)
    overlay = cv2.addWeighted(
        img_rgb,
        1.0 - OVERLAY_ALPHA,
        heatmap,
        OVERLAY_ALPHA,
        0,
    )
    return overlay


def save_cam_map(cam, save_path, title_label):
    plt.figure(figsize=(5.2, 5.2))
    im = plt.imshow(cam, cmap="turbo", vmin=0, vmax=1)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(title_label, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_combined(detection_img, guided_cam, overlay_img, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].imshow(detection_img)
    axes[0].set_title("(a) YOLO26n Detection", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(guided_cam, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("(b) Detection-aware EigenCAM", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("(c) EigenCAM Attribution Overlay", fontsize=11)
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation intensity", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close()


# ============================================================
# MAIN LOOP
# ============================================================
image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:MAX_IMAGES]

print(f"Processing {len(image_files)} images...")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image",
        "detections",
        "target_confidence",
        "target_box",
        "target_layer",
        "status",
    ])

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        bgr = cv2.imread(img_path)

        if bgr is None:
            print("[ERROR] Cannot read:", img_name)
            writer.writerow([img_name, 0, 0, "", TARGET_LAYER_INDEX, "read_error"])
            continue

        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        hook.clear()

        try:
            boxes, confs = get_detections(bgr)

            if hook.activation is None:
                print("[SKIP] No activation:", img_name)
                writer.writerow([img_name, 0, 0, "", TARGET_LAYER_INDEX, "no_activation"])
                continue

            if len(boxes) == 0:
                print("[SKIP] No detection:", img_name)
                writer.writerow([img_name, 0, 0, "", TARGET_LAYER_INDEX, "no_detection"])
                continue

            target_idx = choose_target_detection(boxes, confs)
            target_box = boxes[target_idx]
            target_conf = float(confs[target_idx])

            detection_img = draw_boxes(
                rgb,
                boxes,
                confs,
                target_idx=target_idx,
                only_target=False,
            )

            raw_cam = compute_raw_eigencam(hook.activation)
            guided_cam = compute_detection_aware_cam(raw_cam, target_box)

            overlay_img = make_overlay(rgb, guided_cam)
            overlay_img = draw_boxes(
                overlay_img,
                boxes,
                confs,
                target_idx=target_idx,
                only_target=True,
            )

            cv2.imwrite(
                os.path.join(DETECTION_DIR, f"{base}_detection.png"),
                cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR),
            )

            save_cam_map(
                raw_cam,
                os.path.join(RAW_CAM_DIR, f"{base}_raw_eigencam.png"),
                "Raw EigenCAM activation",
            )

            save_cam_map(
                guided_cam,
                os.path.join(GUIDED_CAM_DIR, f"{base}_detection_aware_eigencam.png"),
                "Detection-aware activation",
            )

            cv2.imwrite(
                os.path.join(OVERLAY_DIR, f"{base}_overlay.png"),
                cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR),
            )

            save_combined(
                detection_img,
                guided_cam,
                overlay_img,
                os.path.join(COMBINED_DIR, f"{base}_combined.png"),
            )

            writer.writerow([
                img_name,
                len(boxes),
                target_conf,
                target_box.tolist(),
                TARGET_LAYER_INDEX,
                "success",
            ])

            print(
                f"[DONE] {img_name} | detections={len(boxes)} | "
                f"target_conf={target_conf:.3f}"
            )

        except Exception as e:
            writer.writerow([
                img_name,
                0,
                0,
                "",
                TARGET_LAYER_INDEX,
                f"failed: {e}",
            ])
            print("[FAILED]", img_name, "|", e)

hook.remove()

print("\nDone.")
print("Saved to:", OUTPUT_DIR)
print("Summary:", CSV_PATH)