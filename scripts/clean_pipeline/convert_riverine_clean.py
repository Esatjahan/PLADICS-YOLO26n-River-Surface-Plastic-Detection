import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# PATH CONFIG
# =========================
ROOT = r"D:\PLADICS\datasets\Riverine Plastic Litter Dataset\Riverine Plastic Litter Dataset\5 categories"

ANNOTATIONS_DIR = os.path.join(ROOT, "Annotations")
PHOTO_DIR = os.path.join(ROOT, "photo")
TEST_DIR = os.path.join(ROOT, "test")

OUTPUT_ROOT = r"D:\PLADICS\datasets_clean\Riverine_single_clean"

# =========================
# SPLIT CONFIG
# =========================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

PLASTIC_CLASSES = {
    "plastic bag",
    "plastic bottle",
    "plastic box",
    "plastic cup"
}

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

# =========================
# HELPER FUNCTIONS
# =========================
def find_image(img_id):
    candidates = [
        os.path.join(PHOTO_DIR, f"{img_id}.jpg"),
        os.path.join(PHOTO_DIR, f"{img_id}.png"),
        os.path.join(TEST_DIR, f"{img_id}.jpg"),
        os.path.join(TEST_DIR, f"{img_id}.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def convert_box_safe(img_w, img_h, xmin, xmax, ymin, ymax):
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(img_w), xmax)
    ymax = min(float(img_h), ymax)

    bw = xmax - xmin
    bh = ymax - ymin

    if bw <= 0 or bh <= 0:
        return None

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    w = bw / img_w
    h = bh / img_h

    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None

    return x_center, y_center, w, h


# =========================
# LOAD ALL XML FILES
# =========================
xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.lower().endswith(".xml")]
random.shuffle(xml_files)

n = len(xml_files)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

splits = {
    "train": xml_files[:n_train],
    "val": xml_files[n_train:n_train + n_val],
    "test": xml_files[n_train + n_val:]
}

# =========================
# MAIN PROCESS
# =========================
removed_can = 0
invalid_boxes = 0
missing_images = 0
skipped_no_plastic = 0
written_images = 0
written_labels = 0
total_xml = 0

for split_name, files in splits.items():
    for xml_file in files:

        total_xml += 1
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        img_id = Path(xml_file).stem

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

        src_img = find_image(img_id)

        if src_img is None:
            print("Missing image:", img_id)
            missing_images += 1
            continue

        valid_boxes = []

        for obj in root.findall("object"):
            cls = obj.find("name").text.strip()

            # ❌ remove non-plastic
            if cls == "can":
                removed_can += 1
                continue

            # ✅ keep only plastic
            if cls not in PLASTIC_CLASSES:
                continue

            xmlbox = obj.find("bndbox")

            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            bbox = convert_box_safe(img_w, img_h, xmin, xmax, ymin, ymax)

            if bbox is None:
                invalid_boxes += 1
                continue

            valid_boxes.append(bbox)

        if not valid_boxes:
            skipped_no_plastic += 1
            continue

        ext = Path(src_img).suffix

        dst_img = os.path.join(OUTPUT_ROOT, "images", split_name, f"{img_id}{ext}")
        label_path = os.path.join(OUTPUT_ROOT, "labels", split_name, f"{img_id}.txt")

        shutil.copy2(src_img, dst_img)

        with open(label_path, "w", encoding="utf-8") as f:
            for x, y, w, h in valid_boxes:
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        written_images += 1
        written_labels += 1

# =========================
# SUMMARY
# =========================
print("\n✅ Riverine CLEAN conversion DONE\n")
print("Total XML scanned:", total_xml)
print("Removed 'can' objects:", removed_can)
print("Invalid boxes skipped:", invalid_boxes)
print("Missing images:", missing_images)
print("Skipped images with no plastic:", skipped_no_plastic)
print("Written images:", written_images)
print("Written labels:", written_labels)
print("Output folder:", OUTPUT_ROOT)