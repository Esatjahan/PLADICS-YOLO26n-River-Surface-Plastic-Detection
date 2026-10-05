import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# =========================
# PATHS
# =========================
RIVERINE_ROOT = r"E:\PLADICS\datasets\Riverine Plastic Litter Dataset\Riverine Plastic Litter Dataset\5 categories"
ANNOTATIONS_DIR = os.path.join(RIVERINE_ROOT, "Annotations")
IMAGESETS_DIR = os.path.join(RIVERINE_ROOT, "ImageSets", "Main")
PHOTO_DIR = os.path.join(RIVERINE_ROOT, "photo")
TEST_DIR = os.path.join(RIVERINE_ROOT, "test")

OUTPUT_ROOT = r"E:\PLADICS\merged_dataset"

# =========================
# CLASSES TO KEEP
# We have to make everything (final class 0 = plastic ) like that.....
# =========================
KEEP_CLASSES = {
    "plastic bag",
    "plastic bottle",
    "plastic cup",
    "plastic box",
    "plastic bags",
    "plastic bottles",
    "plastic cups",
    "plastic boxes",
    "bag",
    "bottle",
    "cup",
    "box"
}

def read_split_file(txt_path):
    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                # remove unwanted spaces like: train (1014) -> train(1014)
                name = name.replace(" (", "(").replace(") ", ")")
                names.append(name)
    return names

def find_image_file(stem):
    candidates = [
        os.path.join(PHOTO_DIR, stem + ".jpg"),
        os.path.join(PHOTO_DIR, stem + ".png"),
        os.path.join(TEST_DIR, stem + ".jpg"),
        os.path.join(TEST_DIR, stem + ".png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def voc_to_yolo(size, box):
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return x_center, y_center, w, h

def process_split(split_name, split_file):
    split_path = os.path.join(IMAGESETS_DIR, split_file)
    if not os.path.exists(split_path):
        print(f"Split file missing: {split_path}")
        return

    names = read_split_file(split_path)
    print(f"{split_name}: {len(names)} items")

    out_img_dir = os.path.join(OUTPUT_ROOT, "images", split_name)
    out_lbl_dir = os.path.join(OUTPUT_ROOT, "labels", split_name)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    copied = 0
    kept_objects = 0

    for stem in names:
        clean_stem = stem.replace(" (", "(").replace(") ", ")").strip()

        xml_path = os.path.join(ANNOTATIONS_DIR, clean_stem + ".xml")
        if not os.path.exists(xml_path):
            print(f"Missing XML: {xml_path}")
            continue

        img_path = find_image_file(clean_stem)
        if img_path is None:
            print(f"Missing image for: {clean_stem}")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_tag = root.find("size")
        if size_tag is None:
            print(f"No size tag: {xml_path}")
            continue

        img_w = int(size_tag.find("width").text)
        img_h = int(size_tag.find("height").text)

        yolo_lines = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip().lower()

            if cls_name not in KEEP_CLASSES:
                continue

            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            x, y, w, h = voc_to_yolo((img_w, img_h), (xmin, ymin, xmax, ymax))
            yolo_lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            kept_objects += 1

        if not yolo_lines:
            continue

        img_ext = Path(img_path).suffix.lower()
        out_img_path = os.path.join(out_img_dir, clean_stem + img_ext)
        out_lbl_path = os.path.join(out_lbl_dir, clean_stem + ".txt")

        if os.path.exists(out_img_path):
            out_img_path = os.path.join(out_img_dir, "riverine_" + clean_stem + img_ext)
            out_lbl_path = os.path.join(out_lbl_dir, "riverine_" + clean_stem + ".txt")

        shutil.copy2(img_path, out_img_path)

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        copied += 1

    print(f"{split_name}: copied {copied} images, kept {kept_objects} plastic objects")
    
if __name__ == "__main__":
    process_split("train", "train.txt")
    process_split("val", "val.txt")
    process_split("test", "test.txt")
    print("Riverine merge complete.")