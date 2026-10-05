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
OUTPUT_DIR = r"D:\PLADICS\explainability\drise_final"

DETECTION_DIR = os.path.join(OUTPUT_DIR, "detection")
DRISE_DIR = os.path.join(OUTPUT_DIR, "drise_map")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "drise_overlay")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")

for d in [OUTPUT_DIR, DETECTION_DIR, DRISE_DIR, OVERLAY_DIR, COMBINED_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "drise_summary.csv")

# =========================
# SETTINGS
# =========================
IMG_SIZE = 416
CONF_THRES = 0.25
MIN_TARGET_CONF = 0.60
MAX_IMAGES = 21

# D-RISE settings
NUM_MASKS = 1500        # trial: 300, final paper: 1500-3000
MASK_SIZE = 16          # 14-20 good for object detection
MASK_PROB = 0.50
BATCH_SIZE = 1          # CPU safe
OVERLAY_ALPHA = 0.35
DPI = 600

np.random.seed(42)

print("=" * 80)
print("YOLO26n D-RISE Explainability Script")
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


def center_score(ref_box, box):
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

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    return boxes, confs


def choose_target(boxes, confs):
    best_idx = None
    best_score = -1.0

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < MIN_TARGET_CONF:
            continue

        x1, y1, x2, y2 = box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        rel_area = area / (IMG_SIZE * IMG_SIZE)

        score = float(conf) * np.sqrt(rel_area)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        best_idx = int(np.argmax(confs))

    return best_idx


def padded_box(box, pad=3):
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

        x1, y1, x2, y2 = padded_box(box, pad=3)

        selected = target_idx is not None and i == target_idx
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


def detection_specific_score(img_rgb, ref_box):
    boxes, confs = get_detections(img_rgb)

    if len(boxes) == 0:
        return 0.0

    best_score = 0.0

    for box, conf in zip(boxes, confs):
        overlap = iou(ref_box, box)
        cscore = center_score(ref_box, box)
        match_score = 0.75 * overlap + 0.25 * cscore

        if match_score > 0.10:
            best_score = max(best_score, float(conf) * match_score)

    return best_score


def generate_random_mask():
    small_mask = (np.random.rand(MASK_SIZE, MASK_SIZE) < MASK_PROB).astype(np.float32)

    mask = cv2.resize(
        small_mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8, sigmaY=8)
    mask = np.clip(mask, 0, 1)

    return mask


def normalize_map(x):
    x = np.nan_to_num(x)
    x = x - np.min(x)

    if np.max(x) > 0:
        x = x / np.max(x)

    return x


def run_drise(img_rgb, ref_box):
    saliency = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    score_sum = 0.0

    for n in range(NUM_MASKS):
        mask = generate_random_mask()

        masked_img = img_rgb.astype(np.float32) * mask[:, :, None]
        masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)

        score = detection_specific_score(masked_img, ref_box)

        saliency += score * mask
        score_sum += score

        if (n + 1) % 100 == 0:
            print(f"  D-RISE masks: {n + 1}/{NUM_MASKS}")

    if score_sum > 0:
        saliency = saliency / score_sum

    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=4, sigmaY=4)
    saliency = normalize_map(saliency)

    return saliency


def make_heatmap(saliency):
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def make_overlay(img_rgb, saliency):
    heatmap = make_heatmap(saliency)
    overlay = cv2.addWeighted(
        img_rgb,
        1 - OVERLAY_ALPHA,
        heatmap,
        OVERLAY_ALPHA,
        0
    )
    return overlay


def save_drise_map(saliency, save_path):
    plt.figure(figsize=(5.2, 5.2))
    im = plt.imshow(saliency, cmap="turbo", vmin=0, vmax=1)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("D-RISE importance", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_combined(detection_img, saliency, overlay_img, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].imshow(detection_img)
    axes[0].set_title("(a) YOLO26n Detection", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(saliency, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("(b) D-RISE Importance Map", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("(c) D-RISE Attribution Overlay", fontsize=11)
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Importance", fontsize=8)

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
    writer.writerow([
        "image",
        "detections",
        "target_confidence",
        "target_box",
        "status"
    ])

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

        if len(boxes) == 0:
            print("[SKIP] No detection:", img_name)
            writer.writerow([img_name, 0, 0, "", "no_detection"])
            continue

        target_idx = choose_target(boxes, confs)
        target_box = boxes[target_idx]
        target_conf = float(confs[target_idx])

        print(f"\n[D-RISE RUNNING] {img_name} | target_conf={target_conf:.3f}")

        detection_img = draw_boxes(
            rgb,
            boxes,
            confs,
            target_idx=target_idx,
            only_target=False
        )

        saliency = run_drise(rgb, target_box)

        overlay_img = make_overlay(rgb, saliency)
        overlay_img = draw_boxes(
            overlay_img,
            boxes,
            confs,
            target_idx=target_idx,
            only_target=True
        )

        cv2.imwrite(
            os.path.join(DETECTION_DIR, f"{base}_detection.png"),
            cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
        )

        save_drise_map(
            saliency,
            os.path.join(DRISE_DIR, f"{base}_drise.png")
        )

        cv2.imwrite(
            os.path.join(OVERLAY_DIR, f"{base}_overlay.png"),
            cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        )

        save_combined(
            detection_img,
            saliency,
            overlay_img,
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

print("\nDone.")
print("Saved to:", OUTPUT_DIR)
print("Summary:", CSV_PATH)