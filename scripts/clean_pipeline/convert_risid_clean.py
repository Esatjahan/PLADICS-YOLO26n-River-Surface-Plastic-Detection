import os
import json
import random
import shutil
from pathlib import Path

# ===== PATH =====
RISID_ROOT = r"D:/PLADICS/datasets/RiSID"
IMAGES_DIR = r"D:/PLADICS/datasets/RiSID/images/images"
ANNOTATION_FILE = r"D:/PLADICS/datasets/RiSID/annotations_2cat.json"

OUTPUT_ROOT = r"D:/PLADICS/datasets_clean/RiSID_single_clean"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

PLASTIC_ID = 1  # confirmed earlier

# ===== CREATE FOLDERS =====
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

# ===== LOAD JSON =====
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

image_id_to_filename = {img["id"]: img["file_name"] for img in images}

ann_by_image = {}
for ann in annotations:
    if ann["category_id"] != PLASTIC_ID:
        continue  # ❌ remove non-plastic

    img_id = ann["image_id"]
    ann_by_image.setdefault(img_id, []).append(ann)

# ===== SPLIT =====
all_images = list(image_id_to_filename.items())
random.shuffle(all_images)

n = len(all_images)
train_split = int(n * TRAIN_RATIO)
val_split = int(n * VAL_RATIO)

splits = {
    "train": all_images[:train_split],
    "val": all_images[train_split:train_split + val_split],
    "test": all_images[train_split + val_split:]
}

# ===== SAFE BBOX =====
def convert_bbox(bbox, w, h):
    x, y, bw, bh = bbox

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + bw / 2) / w
    yc = (y1 + bh / 2) / h
    bw /= w
    bh /= h

    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        return None

    return xc, yc, bw, bh

# ===== PROCESS =====
for split_name, data in splits.items():

    for image_id, file_name in data:

        src = os.path.join(IMAGES_DIR, file_name)
        dst = os.path.join(OUTPUT_ROOT, "images", split_name, file_name)

        if not os.path.exists(src):
            continue

        img_info = next(i for i in images if i["id"] == image_id)

        w, h = img_info["width"], img_info["height"]

        label_path = os.path.join(
            OUTPUT_ROOT, "labels", split_name, Path(file_name).stem + ".txt"
        )

        valid_boxes = []

        for ann in ann_by_image.get(image_id, []):
            bbox = convert_bbox(ann["bbox"], w, h)
            if bbox:
                valid_boxes.append(bbox)

        if not valid_boxes:
            continue  # skip image with no plastic

        shutil.copy2(src, dst)

        with open(label_path, "w") as f:
            for x, y, bw, bh in valid_boxes:
                f.write(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ RiSID clean conversion DONE")