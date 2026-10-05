import os
import json
import random
import shutil
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
ANNOTATION_FILE = r"D:\PLADICS\datasets\RiSID\annotations_7cat.json"
IMAGES_DIR = r"D:\PLADICS\datasets\RiSID\images\images"
OUTPUT_ROOT = r"D:\PLADICS\datasets_multiclass\risid_multiclass"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
random.seed(42)

# ----------------------------
# CREATE OUTPUT FOLDERS
# ----------------------------
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

# ----------------------------
# LOAD COCO JSON
# ----------------------------
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Build category mapping from original category_id -> YOLO class index
categories_sorted = sorted(categories, key=lambda x: x["id"])
cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(categories_sorted)}
class_names = [cat["name"] for cat in categories_sorted]

# Save class mapping
mapping_path = os.path.join(OUTPUT_ROOT, "class_mapping.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    for idx, name in enumerate(class_names):
        f.write(f"{idx} = {name}\n")

# Save YAML
yaml_path = os.path.join(OUTPUT_ROOT, "risid.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write("path: D:/PLADICS/datasets_multiclass/risid_multiclass\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for idx, name in enumerate(class_names):
        f.write(f"  {idx}: {name}\n")

# ----------------------------
# INDEX IMAGES AND ANNOTATIONS
# ----------------------------
image_id_to_info = {img["id"]: img for img in images}
ann_by_image = {}

for ann in annotations:
    img_id = ann["image_id"]
    ann_by_image.setdefault(img_id, []).append(ann)

all_images = list(image_id_to_info.items())
random.shuffle(all_images)

n = len(all_images)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_data = all_images[:train_end]
val_data = all_images[train_end:val_end]
test_data = all_images[val_end:]

splits = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w = w / img_w
    h = h / img_h
    return x_center, y_center, w, h

# ----------------------------
# COPY IMAGES AND WRITE LABELS
# ----------------------------
missing = 0

for split_name, data in splits.items():
    for image_id, img_info in data:
        file_name = img_info["file_name"]

        src = os.path.join(IMAGES_DIR, file_name)
        dst = os.path.join(OUTPUT_ROOT, "images", split_name, file_name)

        if not os.path.exists(src):
            print("Missing image:", src)
            missing += 1
            continue

        shutil.copy2(src, dst)

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_path = os.path.join(
            OUTPUT_ROOT,
            "labels",
            split_name,
            Path(file_name).stem + ".txt"
        )

        with open(label_path, "w", encoding="utf-8") as f:
            if image_id in ann_by_image:
                for ann in ann_by_image[image_id]:
                    orig_cat_id = ann["category_id"]
                    yolo_class_id = cat_id_to_yolo[orig_cat_id]
                    x, y, w, h = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                    f.write(f"{yolo_class_id} {x} {y} {w} {h}\n")

print("DONE")
print("Missing images:", missing)
print("YAML saved to:", yaml_path)
print("Class mapping saved to:", mapping_path)