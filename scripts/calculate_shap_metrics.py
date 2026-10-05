import os
import ast
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
SHAP_OUTPUT_DIR = r"D:\PLADICS\explainability\shap_final"

CSV_PATH = os.path.join(SHAP_OUTPUT_DIR, "shap_summary.csv")
ORIGINAL_DIR = os.path.join(SHAP_OUTPUT_DIR, "original")
SHAP_DIR = os.path.join(SHAP_OUTPUT_DIR, "shap_map")

OUT_CSV = os.path.join(SHAP_OUTPUT_DIR, "shap_quantitative_metrics.csv")
OUT_SUMMARY = os.path.join(SHAP_OUTPUT_DIR, "shap_quantitative_summary.csv")

# =========================
# SETTINGS
# =========================
IMG_SIZE = 416
CONF_THRES = 0.25
MATCH_THRES = 0.10

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


def get_matched_confidence(img_rgb, ref_box):
    boxes, confs = get_detections(img_rgb)

    if len(boxes) == 0:
        return 0.0

    best_match = 0.0
    best_conf = 0.0

    for box, conf in zip(boxes, confs):
        overlap = iou(ref_box, box)
        center_score = center_distance_score(ref_box, box)
        match_score = 0.75 * overlap + 0.25 * center_score

        if match_score > best_match:
            best_match = match_score
            best_conf = float(conf)

    if best_match < MATCH_THRES:
        return 0.0

    return best_conf


def clip_box(box):
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(IMG_SIZE - 1, x1))
    y1 = max(0, min(IMG_SIZE - 1, y1))
    x2 = max(0, min(IMG_SIZE - 1, x2))
    y2 = max(0, min(IMG_SIZE - 1, y2))

    return x1, y1, x2, y2


def mask_top_shap_regions(img_rgb, signed_map, percent):
    """
    Positive SHAP attribution er top percent region blur kore confidence drop measure kore.
    percent = 20, 40, 60
    """
    img_masked = img_rgb.copy()

    positive_map = np.maximum(signed_map, 0)

    if positive_map.max() <= 0:
        return img_masked

    values = positive_map[positive_map > 0]

    if len(values) == 0:
        return img_masked

    threshold = np.percentile(values, 100 - percent)
    mask = positive_map >= threshold

    blurred = cv2.GaussianBlur(img_rgb, (31, 31), 0)
    img_masked[mask] = blurred[mask]

    return img_masked


def pointing_game(signed_map, target_box):
    """
    Max positive SHAP point bbox er vitore kina.
    """
    positive_map = np.maximum(signed_map, 0)

    if positive_map.max() <= 0:
        return 0, -1, -1

    y, x = np.unravel_index(np.argmax(positive_map), positive_map.shape)

    x1, y1, x2, y2 = clip_box(target_box)

    inside = 1 if (x1 <= x <= x2 and y1 <= y <= y2) else 0

    return inside, int(x), int(y)


def inside_attribution_ratio(signed_map, target_box):
    """
    Total positive SHAP attribution er koto percent bbox er vitore.
    """
    positive_map = np.maximum(signed_map, 0)
    total_positive = positive_map.sum()

    if total_positive <= 0:
        return 0.0

    x1, y1, x2, y2 = clip_box(target_box)

    bbox_mask = np.zeros_like(positive_map, dtype=np.uint8)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = 1

    inside_sum = positive_map[bbox_mask == 1].sum()

    return float(inside_sum / total_positive)


def safe_parse_box(box_text):
    if isinstance(box_text, str):
        return ast.literal_eval(box_text)
    return box_text


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "success"].copy()

    rows = []

    print("=" * 80)
    print("SHAP QUANTITATIVE METRICS")
    print("Successful SHAP images:", len(df))
    print("=" * 80)

    for _, row in df.iterrows():
        img_name = row["image"]
        base = os.path.splitext(img_name)[0]

        original_path = os.path.join(ORIGINAL_DIR, f"{base}_original.png")
        signed_map_path = os.path.join(SHAP_DIR, f"{base}_signed_map.npy")

        if not os.path.exists(original_path):
            print("[SKIP] Missing original:", original_path)
            continue

        if not os.path.exists(signed_map_path):
            print("[SKIP] Missing signed map:", signed_map_path)
            continue

        bgr = cv2.imread(original_path)

        if bgr is None:
            print("[SKIP] Cannot read original:", original_path)
            continue

        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        signed_map = np.load(signed_map_path)
        signed_map = cv2.resize(signed_map, (IMG_SIZE, IMG_SIZE))

        target_box = safe_parse_box(row["target_box"])
        original_conf = float(row["target_confidence"])

        # Confidence drop
        masked20 = mask_top_shap_regions(rgb, signed_map, 20)
        masked40 = mask_top_shap_regions(rgb, signed_map, 40)
        masked60 = mask_top_shap_regions(rgb, signed_map, 60)

        conf20 = get_matched_confidence(masked20, target_box)
        conf40 = get_matched_confidence(masked40, target_box)
        conf60 = get_matched_confidence(masked60, target_box)

        drop20 = original_conf - conf20
        drop40 = original_conf - conf40
        drop60 = original_conf - conf60

        # Prevent negative drop from numerical/detection fluctuation
        drop20 = max(0.0, drop20)
        drop40 = max(0.0, drop40)
        drop60 = max(0.0, drop60)

        # Pointing game
        pointing_inside, pointing_x, pointing_y = pointing_game(
            signed_map,
            target_box
        )

        # Inside attribution ratio
        inside_ratio = inside_attribution_ratio(
            signed_map,
            target_box
        )

        rows.append({
            "image": img_name,
            "original_conf": original_conf,
            "conf_top20_removed": conf20,
            "conf_top40_removed": conf40,
            "conf_top60_removed": conf60,
            "drop_top20": drop20,
            "drop_top40": drop40,
            "drop_top60": drop60,
            "pointing_inside": pointing_inside,
            "pointing_x": pointing_x,
            "pointing_y": pointing_y,
            "inside_attr_ratio": inside_ratio,
            "target_box": target_box
        })

        print(
            f"[DONE] {img_name} | "
            f"drop20={drop20:.3f}, "
            f"drop40={drop40:.3f}, "
            f"drop60={drop60:.3f}, "
            f"point={pointing_inside}, "
            f"inside={inside_ratio:.3f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    summary_rows = [
        ["num_images", len(out), ""],

        ["mean_original_conf", out["original_conf"].mean(), out["original_conf"].std()],

        ["mean_conf_top20_removed", out["conf_top20_removed"].mean(), out["conf_top20_removed"].std()],
        ["mean_conf_top40_removed", out["conf_top40_removed"].mean(), out["conf_top40_removed"].std()],
        ["mean_conf_top60_removed", out["conf_top60_removed"].mean(), out["conf_top60_removed"].std()],

        ["mean_drop_top20", out["drop_top20"].mean(), out["drop_top20"].std()],
        ["mean_drop_top40", out["drop_top40"].mean(), out["drop_top40"].std()],
        ["mean_drop_top60", out["drop_top60"].mean(), out["drop_top60"].std()],

        ["pointing_accuracy", out["pointing_inside"].mean(), ""],

        ["mean_inside_attr_ratio", out["inside_attr_ratio"].mean(), out["inside_attr_ratio"].std()],
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "mean", "std"])
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved detail CSV:", OUT_CSV)
    print("Saved summary CSV:", OUT_SUMMARY)

    print("\n===== SUMMARY =====")
    print(summary)


if __name__ == "__main__":
    main()