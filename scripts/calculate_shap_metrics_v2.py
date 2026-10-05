import os, ast, cv2
import numpy as np
import pandas as pd

SHAP_OUTPUT_DIR = r"D:\PLADICS\explainability\shap_final"
CSV_PATH = os.path.join(SHAP_OUTPUT_DIR, "shap_summary.csv")
SHAP_DIR = os.path.join(SHAP_OUTPUT_DIR, "shap_map")

OUT_CSV = os.path.join(SHAP_OUTPUT_DIR, "shap_localization_metrics_v2.csv")
OUT_SUMMARY = os.path.join(SHAP_OUTPUT_DIR, "shap_localization_summary_v2.csv")

IMG_SIZE = 416
EXPAND_RATIO = 0.25
TOP_PERCENT = 5


def clip_box(box):
    x1, y1, x2, y2 = map(int, box)
    return (
        max(0, min(IMG_SIZE - 1, x1)),
        max(0, min(IMG_SIZE - 1, y1)),
        max(0, min(IMG_SIZE - 1, x2)),
        max(0, min(IMG_SIZE - 1, y2)),
    )


def expand_box(box, ratio=0.25):
    x1, y1, x2, y2 = clip_box(box)
    w, h = x2 - x1, y2 - y1
    px, py = int(w * ratio), int(h * ratio)

    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(IMG_SIZE - 1, x2 + px),
        min(IMG_SIZE - 1, y2 + py),
    )


def box_mask(box):
    x1, y1, x2, y2 = clip_box(box)
    m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    m[y1:y2 + 1, x1:x2 + 1] = 1
    return m


def topk_hit_rate(signed_map, box, top_percent=5):
    positive = np.maximum(signed_map, 0)

    if positive.max() <= 0:
        return 0.0

    values = positive[positive > 0]
    threshold = np.percentile(values, 100 - top_percent)
    top_mask = positive >= threshold

    bbox = box_mask(box)

    hit_pixels = np.logical_and(top_mask, bbox).sum()
    total_top_pixels = top_mask.sum()

    if total_top_pixels == 0:
        return 0.0

    return float(hit_pixels / total_top_pixels)


def inside_ratio(signed_map, box):
    positive = np.maximum(signed_map, 0)
    total = positive.sum()

    if total <= 0:
        return 0.0

    bbox = box_mask(box)
    inside = positive[bbox == 1].sum()

    return float(inside / total)


def centroid_inside(signed_map, box):
    positive = np.maximum(signed_map, 0)

    if positive.sum() <= 0:
        return 0, -1, -1

    ys, xs = np.indices(positive.shape)

    cx = int((xs * positive).sum() / positive.sum())
    cy = int((ys * positive).sum() / positive.sum())

    x1, y1, x2, y2 = clip_box(box)

    inside = 1 if (x1 <= cx <= x2 and y1 <= cy <= y2) else 0

    return inside, cx, cy


def parse_box(text):
    return ast.literal_eval(text) if isinstance(text, str) else text


df = pd.read_csv(CSV_PATH)
df = df[df["status"] == "success"].copy()

rows = []

for _, row in df.iterrows():
    img_name = row["image"]
    base = os.path.splitext(img_name)[0]

    npy_path = os.path.join(SHAP_DIR, f"{base}_signed_map.npy")

    if not os.path.exists(npy_path):
        print("[SKIP] Missing:", npy_path)
        continue

    signed_map = np.load(npy_path)
    signed_map = cv2.resize(signed_map, (IMG_SIZE, IMG_SIZE))

    target_box = parse_box(row["target_box"])
    expanded = expand_box(target_box, EXPAND_RATIO)

    top5_hit_original = topk_hit_rate(signed_map, target_box, TOP_PERCENT)
    top5_hit_expanded = topk_hit_rate(signed_map, expanded, TOP_PERCENT)

    inside_original = inside_ratio(signed_map, target_box)
    inside_expanded = inside_ratio(signed_map, expanded)

    centroid_original, cx, cy = centroid_inside(signed_map, target_box)
    centroid_expanded, _, _ = centroid_inside(signed_map, expanded)

    rows.append({
        "image": img_name,
        "top5_hit_original_bbox": top5_hit_original,
        "top5_hit_expanded_bbox": top5_hit_expanded,
        "inside_ratio_original_bbox": inside_original,
        "inside_ratio_expanded_bbox": inside_expanded,
        "centroid_inside_original_bbox": centroid_original,
        "centroid_inside_expanded_bbox": centroid_expanded,
        "centroid_x": cx,
        "centroid_y": cy,
        "target_box": target_box,
        "expanded_box": expanded
    })

    print(
        f"[DONE] {img_name} | "
        f"top5_exp={top5_hit_expanded:.3f}, "
        f"inside_exp={inside_expanded:.3f}, "
        f"centroid_exp={centroid_expanded}"
    )

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

summary = pd.DataFrame([
    ["num_images", len(out), ""],
    ["top5_hit_original_bbox", out["top5_hit_original_bbox"].mean(), out["top5_hit_original_bbox"].std()],
    ["top5_hit_expanded_bbox", out["top5_hit_expanded_bbox"].mean(), out["top5_hit_expanded_bbox"].std()],
    ["inside_ratio_original_bbox", out["inside_ratio_original_bbox"].mean(), out["inside_ratio_original_bbox"].std()],
    ["inside_ratio_expanded_bbox", out["inside_ratio_expanded_bbox"].mean(), out["inside_ratio_expanded_bbox"].std()],
    ["centroid_inside_original_bbox", out["centroid_inside_original_bbox"].mean(), ""],
    ["centroid_inside_expanded_bbox", out["centroid_inside_expanded_bbox"].mean(), ""],
], columns=["metric", "mean", "std"])

summary.to_csv(OUT_SUMMARY, index=False)

print("\nSaved:", OUT_CSV)
print("Saved:", OUT_SUMMARY)
print("\n===== SUMMARY V2 =====")
print(summary)