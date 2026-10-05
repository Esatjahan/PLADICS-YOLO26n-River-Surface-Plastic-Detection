import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
INPUT_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\eigencam_final"

DETECTION_DIR = os.path.join(OUTPUT_DIR, "detection")
CAM_DIR = os.path.join(OUTPUT_DIR, "eigencam_map")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "eigencam_overlay")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")

for d in [OUTPUT_DIR, DETECTION_DIR, CAM_DIR, OVERLAY_DIR, COMBINED_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "eigencam_summary.csv")

# =========================
# SETTINGS
# =========================
IMG_SIZE = 640
CONF_THRES = 0.25
MAX_IMAGES = 100
TARGET_LAYER_INDEX = 22      # Final C3k2 block before Detect head
OVERLAY_ALPHA = 0.35
DPI = 600

print("=" * 80)
print("YOLO26n EigenCAM Script")
print("TARGET_LAYER_INDEX:", TARGET_LAYER_INDEX)
print("INPUT_DIR:", INPUT_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("INPUT_EXISTS:", os.path.exists(INPUT_DIR))
print("=" * 80)

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(INPUT_DIR)

model = YOLO(MODEL_PATH)
target_layer = model.model.model[TARGET_LAYER_INDEX]

print("[INFO] Target layer:", target_layer.__class__.__name__)


# =========================
# ACTIVATION HOOK
# =========================
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


# =========================
# UTILS
# =========================
def get_detections(img_bgr):
    result = model.predict(
        source=img_bgr,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.array([]), np.array([])

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    return boxes, confs


def padded_box(box, pad=3):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(IMG_SIZE - 1, x2 + pad)
    y2 = min(IMG_SIZE - 1, y2 + pad)
    return x1, y1, x2, y2


def draw_boxes(img_rgb, boxes, confs):
    out = img_rgb.copy()

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = padded_box(box, pad=3)

        color = (0, 255, 0)
        thickness = 2
        label = f"plastic {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        font_scale = 0.60
        font_thickness = 2
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
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
            cv2.LINE_AA
        )

    return out


def normalize_cam(cam):
    cam = np.nan_to_num(cam)
    cam = cam - np.min(cam)

    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    return cam


def compute_eigencam(activation):
    """
    EigenCAM from selected YOLO26n feature map.
    Expected activation shape: [1, C, H, W]
    """
    if activation is None:
        raise RuntimeError("No activation captured. Forward hook failed.")

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

    # Ensure positive dominant activation
    if abs(cam.min()) > abs(cam.max()):
        cam = -cam

    cam = np.maximum(cam, 0)
    cam = normalize_cam(cam)

    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2.5, sigmaY=2.5)
    cam = normalize_cam(cam)

    return cam


def make_heatmap(cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def make_overlay(img_rgb, cam):
    heatmap = make_heatmap(cam)
    overlay = cv2.addWeighted(img_rgb, 1 - OVERLAY_ALPHA, heatmap, OVERLAY_ALPHA, 0)
    return overlay


def save_cam_only(cam, save_path):
    plt.figure(figsize=(5.2, 5.2))
    im = plt.imshow(cam, cmap="jet", vmin=0, vmax=1)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("EigenCAM activation", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_combined(detection_img, cam, overlay_img, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].imshow(detection_img)
    axes[0].set_title("(a) YOLO26n Detection", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("(b) EigenCAM Activation Map", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("(c) EigenCAM Overlay", fontsize=11)
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation intensity", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close()


# =========================
# MAIN LOOP
# =========================
image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:MAX_IMAGES]

print(f"Processing {len(image_files)} images...")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "detections", "max_confidence", "target_layer", "status"])

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        bgr = cv2.imread(img_path)

        if bgr is None:
            print("[ERROR] Cannot read:", img_name)
            writer.writerow([img_name, 0, 0, TARGET_LAYER_INDEX, "read_error"])
            continue

        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        hook.clear()

        try:
            boxes, confs = get_detections(bgr)

            if hook.activation is None:
                print("[SKIP] No activation:", img_name)
                writer.writerow([img_name, 0, 0, TARGET_LAYER_INDEX, "no_activation"])
                continue

            if len(boxes) == 0:
                print("[SKIP] No detection:", img_name)
                writer.writerow([img_name, 0, 0, TARGET_LAYER_INDEX, "no_detection"])
                continue

            max_conf = float(np.max(confs))

            detection_img = draw_boxes(rgb, boxes, confs)

            cam = compute_eigencam(hook.activation)
            overlay_img = make_overlay(rgb, cam)
            overlay_img = draw_boxes(overlay_img, boxes, confs)

            cv2.imwrite(
                os.path.join(DETECTION_DIR, f"{base}_detection.png"),
                cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
            )

            save_cam_only(
                cam,
                os.path.join(CAM_DIR, f"{base}_eigencam.png")
            )

            cv2.imwrite(
                os.path.join(OVERLAY_DIR, f"{base}_overlay.png"),
                cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            )

            save_combined(
                detection_img,
                cam,
                overlay_img,
                os.path.join(COMBINED_DIR, f"{base}_combined.png")
            )

            writer.writerow([img_name, len(boxes), max_conf, TARGET_LAYER_INDEX, "success"])
            print(f"[DONE] {img_name} | detections={len(boxes)} | max_conf={max_conf:.3f}")

        except Exception as e:
            writer.writerow([img_name, 0, 0, TARGET_LAYER_INDEX, f"failed: {e}"])
            print("[FAILED]", img_name, "|", e)

hook.remove()

print("\nDone.")
print("Saved to:", OUTPUT_DIR)
print("Summary:", CSV_PATH)