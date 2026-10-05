import os
import random
import shutil
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT = r"D:\PLADICS\datasets\Riverine Plastic Litter Dataset\Riverine Plastic Litter Dataset\5 categories"
ANNOTATIONS_DIR = os.path.join(ROOT, "Annotations")
PHOTO_DIR = os.path.join(ROOT, "photo")
TEST_DIR = os.path.join(ROOT, "test")

OUTPUT_ROOT = r"D:\PLADICS\datasets_multiclass\riverine_multiclass"

# Exact class names found in XML
CLASSES = [
    "can",
    "plastic bag",
    "plastic bottle",
    "plastic box",
    "plastic cup",
]

random.seed(42)

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

# =========================
# SAVE YAML
# =========================
yaml_path = os.path.join(OUTPUT_ROOT, "riverine.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write("path: D:/PLADICS/datasets_multiclass/riverine_multiclass\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")
    for i, name in enumerate(CLASSES):
        safe_name = name.replace(" ", "_")
        f.write(f"  {i}: {safe_name}\n")

# =========================
# SAVE CLASS MAPPING
# =========================
mapping_path = os.path.join(OUTPUT_ROOT, "class_mapping.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    for i, name in enumerate(CLASSES):
        f.write(f"{i} = {name}\n")

class_to_id = {name: i for i, name in enumerate(CLASSES)}

# =========================
# FIND IMAGE FILE ROBUSTLY
# =========================
def normalize_name(name: str) -> str:
    # remove extension if present
    stem = Path(name).stem
    # normalize spaces in names like "train (1)" -> "train(1)"
    stem = stem.replace(" (", "(")
    return stem

def find_image_candidates(image_stem: str):
    """
    Try to find matching image in PHOTO_DIR or TEST_DIR
    using flexible matching.
    """
    candidates = []

    normalized = normalize_name(image_stem)

    # direct patterns
    patterns = [
        os.path.join(PHOTO_DIR, normalized + ".*"),
        os.path.join(TEST_DIR, normalized + ".*"),
        os.path.join(PHOTO_DIR, image_stem + ".*"),
        os.path.join(TEST_DIR, image_stem + ".*"),
    ]

    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    # remove duplicates while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique

# =========================
# PARSE XML AND BUILD VALID PAIRS
# =========================
xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
print("Total XML files:", len(xml_files))

valid_data = []
unknown_classes = set()
missing_images = []

for xml_file in xml_files:
    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Skipping broken XML: {xml_file} -> {e}")
        continue

    filename_tag = root.find("filename")
    if filename_tag is None or not filename_tag.text:
        print(f"Skipping XML without filename: {xml_file}")
        continue

    filename = filename_tag.text.strip()
    image_stem = Path(filename).stem

    # verify classes in XML
    xml_classes = []
    for obj in root.iter("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue
        cls = name_tag.text.strip()
        xml_classes.append(cls)
        if cls not in class_to_id:
            unknown_classes.add(cls)

    if unknown_classes:
        # continue checking others; don't stop here
        pass

    image_candidates = find_image_candidates(image_stem)

    if not image_candidates:
        missing_images.append((xml_file, filename))
        continue

    image_path = image_candidates[0]
    valid_data.append((xml_path, image_path, os.path.basename(image_path)))

print("Valid pairs:", len(valid_data))
print("Missing images:", len(missing_images))

if missing_images:
    print("\nFirst 20 missing image examples:")
    for xml_name, fname in missing_images[:20]:
        print(f"  XML: {xml_name} -> filename tag: {fname}")

if unknown_classes:
    print("\nUnknown classes found in XML:")
    for c in sorted(unknown_classes):
        print("-", c)
    print("\nPlease fix CLASSES before continuing.")
    raise SystemExit(1)

if len(valid_data) == 0:
    print("\nNo valid image-annotation pairs found. Stopping.")
    raise SystemExit(1)

# =========================
# SHUFFLE AND SPLIT
# =========================
random.shuffle(valid_data)

n = len(valid_data)
train_end = int(n * 0.7)
val_end = train_end + int(n * 0.2)

train_data = valid_data[:train_end]
val_data = valid_data[train_end:val_end]
test_data = valid_data[val_end:]

splits = {
    "train": train_data,
    "val": val_data,
    "test": test_data,
}

print("\nSplit sizes:")
print("Train:", len(train_data))
print("Val  :", len(val_data))
print("Test :", len(test_data))

# =========================
# XML -> YOLO BBOX
# =========================
def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    xmin, xmax, ymin, ymax = box

    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    return x * dw, y * dh, w * dw, h * dh

# =========================
# WRITE LABELS + COPY IMAGES
# =========================
processed = 0

for split_name, data in splits.items():
    for xml_path, image_path, image_filename in data:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            print(f"Skipping XML without size: {xml_path}")
            continue

        width_tag = size.find("width")
        height_tag = size.find("height")
        if width_tag is None or height_tag is None:
            print(f"Skipping XML with incomplete size: {xml_path}")
            continue

        w = int(float(width_tag.text))
        h = int(float(height_tag.text))

        label_filename = Path(image_filename).stem + ".txt"
        label_path = os.path.join(OUTPUT_ROOT, "labels", split_name, label_filename)

        with open(label_path, "w", encoding="utf-8") as f:
            for obj in root.iter("object"):
                name_tag = obj.find("name")
                if name_tag is None or not name_tag.text:
                    continue

                cls = name_tag.text.strip()
                if cls not in class_to_id:
                    continue

                cls_id = class_to_id[cls]

                xmlbox = obj.find("bndbox")
                if xmlbox is None:
                    continue

                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)

                bb = convert_box((w, h), (xmin, xmax, ymin, ymax))
                f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

        dst_image = os.path.join(OUTPUT_ROOT, "images", split_name, image_filename)
        shutil.copy2(image_path, dst_image)
        processed += 1

print("\nDONE — CLEAN DATASET CREATED")
print("Processed files:", processed)
print("YAML saved to:", yaml_path)
print("Class mapping saved to:", mapping_path)