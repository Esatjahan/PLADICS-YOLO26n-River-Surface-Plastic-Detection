import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT = r"D:\PLADICS\datasets\Riverine Plastic Litter Dataset\Riverine Plastic Litter Dataset\5 categories"
ANNOTATIONS_DIR = os.path.join(ROOT, "Annotations")
IMAGESETS_MAIN = os.path.join(ROOT, "ImageSets", "Main")
PHOTO_DIR = os.path.join(ROOT, "photo")
TEST_DIR = os.path.join(ROOT, "test")

OUTPUT_ROOT = r"D:\PLADICS\datasets_multiclass\riverine_multiclass"

# -------------------------
# EDIT THIS LIST ONLY AFTER CHECKING XML CLASS NAMES
# -------------------------
CLASSES = [
    "can",
    "plastic bag",
    "plastic bottle",
    "plastic box",
    "plastic cup"
]

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
        f.write(f"  {i}: {name}\n")

# =========================
# SAVE CLASS MAPPING
# =========================
mapping_path = os.path.join(OUTPUT_ROOT, "class_mapping.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    for i, name in enumerate(CLASSES):
        f.write(f"{i} = {name}\n")

class_to_id = {name: i for i, name in enumerate(CLASSES)}

# =========================
# READ SPLITS
# =========================
def read_split_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    cleaned = []
    for line in lines:
        # convert: "train (1)" -> "train(1)"
        line = line.replace(" (", "(")
        cleaned.append(line)

    return cleaned

train_ids = read_split_file(os.path.join(IMAGESETS_MAIN, "train.txt"))
val_ids = read_split_file(os.path.join(IMAGESETS_MAIN, "val.txt"))
test_ids = read_split_file(os.path.join(IMAGESETS_MAIN, "test.txt"))

splits = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

# =========================
# XML -> YOLO
# =========================
def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmin, xmax, ymin, ymax = box
    x = ((xmin + xmax) / 2.0) * dw
    y = ((ymin + ymax) / 2.0) * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh
    return x, y, w, h

missing_images = 0
unknown_classes = set()

for split_name, ids in splits.items():
    for img_id in ids:
        xml_path = os.path.join(ANNOTATIONS_DIR, f"{img_id}.xml")

        if not os.path.exists(xml_path):
            print("Missing XML:", xml_path)
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        # find image file
        img_candidates = [
            os.path.join(PHOTO_DIR, f"{img_id}.jpg"),
            os.path.join(PHOTO_DIR, f"{img_id}.png"),
            os.path.join(TEST_DIR, f"{img_id}.jpg"),
            os.path.join(TEST_DIR, f"{img_id}.png"),
        ]

        src_img = None
        for cand in img_candidates:
            if os.path.exists(cand):
                src_img = cand
                break

        if src_img is None:
            print("Missing image for:", img_id)
            missing_images += 1
            continue

        ext = Path(src_img).suffix
        dst_img = os.path.join(OUTPUT_ROOT, "images", split_name, f"{img_id}{ext}")
        shutil.copy2(src_img, dst_img)

        label_path = os.path.join(OUTPUT_ROOT, "labels", split_name, f"{img_id}.txt")

        with open(label_path, "w", encoding="utf-8") as out_file:
            for obj in root.findall("object"):
                cls = obj.find("name").text.strip()

                if cls not in class_to_id:
                    unknown_classes.add(cls)
                    continue

                cls_id = class_to_id[cls]

                xmlbox = obj.find("bndbox")
                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)

                bb = convert_box((w, h), (xmin, xmax, ymin, ymax))
                out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

print("DONE")
print("Missing images:", missing_images)

if unknown_classes:
    print("Unknown classes found in XML:")
    for c in sorted(unknown_classes):
        print("-", c)

print("YAML saved to:", yaml_path)
print("Class mapping saved to:", mapping_path)