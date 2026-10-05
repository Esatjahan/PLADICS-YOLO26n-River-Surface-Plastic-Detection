import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC_ROOT = Path(r"D:/PLADICS/datasets/Riverine Plastic Litter Dataset/Riverine Plastic Litter Dataset/5 categories")

ANN_DIR = SRC_ROOT / "Annotations"
IMG_DIR = SRC_ROOT / "photo"

OUT_DIR = Path(r"D:/PLADICS/datasets_clean/Riverine_5class_clean")

CLASS_MAP = {
    "can": 0,
    "plastic bag": 1,
    "plastic bottle": 2,
    "plastic box": 3,
    "plastic cup": 4,
}

random.seed(42)

def make_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    bw = xmax - xmin
    bh = ymax - ymin

    if bw <= 0 or bh <= 0:
        return None

    xc = (xmin + bw / 2) / w
    yc = (ymin + bh / 2) / h
    bw = bw / w
    bh = bh / h

    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        return None

    return xc, yc, bw, bh

ALL_IMAGE_DIRS = [
    SRC_ROOT / "photo",
    SRC_ROOT / "test"
]

def find_image(stem):
    for folder in ALL_IMAGE_DIRS:
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None

def convert():
    make_dirs()

    xml_files = sorted(ANN_DIR.glob("*.xml"))
    items = []

    missing_images = 0
    invalid_boxes = 0
    total_objects = 0
    class_counts = {k: 0 for k in CLASS_MAP}

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename")
        stem = xml_path.stem

        size = root.find("size")
        img_w = int(size.findtext("width"))
        img_h = int(size.findtext("height"))

        img_path = find_image(stem)
        if img_path is None and filename:
            img_path = IMG_DIR / filename

        if img_path is None or not img_path.exists():
            missing_images += 1
            continue

        label_lines = []

        for obj in root.findall("object"):
            cls_name = obj.findtext("name").strip()

            if cls_name not in CLASS_MAP:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))

            yolo_box = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

            if yolo_box is None:
                invalid_boxes += 1
                continue

            cls_id = CLASS_MAP[cls_name]
            xc, yc, bw, bh = yolo_box

            label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            total_objects += 1
            class_counts[cls_name] += 1

        if len(label_lines) == 0:
            continue

        items.append((img_path, label_lines))

    random.shuffle(items)

    n = len(items)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)

    splits = {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }

    for split, split_items in splits.items():
        for img_path, label_lines in split_items:
            shutil.copy2(img_path, OUT_DIR / "images" / split / img_path.name)

            label_name = Path(img_path.name).stem + ".txt"
            with open(OUT_DIR / "labels" / split / label_name, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    yaml_path = OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR.as_posix()}\n\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        f.write("  0: can\n")
        f.write("  1: plastic_bag\n")
        f.write("  2: plastic_bottle\n")
        f.write("  3: plastic_box\n")
        f.write("  4: plastic_cup\n")

    print("Riverine 5-class conversion DONE")
    print("Total XML scanned:", len(xml_files))
    print("Written images:", n)
    print("Train:", len(splits["train"]))
    print("Val:", len(splits["val"]))
    print("Test:", len(splits["test"]))
    print("Total objects:", total_objects)
    print("Class counts:", class_counts)
    print("Missing images:", missing_images)
    print("Invalid boxes skipped:", invalid_boxes)
    print("YAML:", yaml_path)

if __name__ == "__main__":
    convert()