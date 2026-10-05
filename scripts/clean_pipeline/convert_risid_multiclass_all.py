import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path("D:/PLADICS/datasets/RiSID")
IMAGE_ROOT = ROOT / "images"

OUTPUT_ROOT = Path("D:/PLADICS/datasets_clean")

TASKS = {
    "RiSID_2class_clean": ROOT / "annotations_2cat.json",
    "RiSID_5class_clean": ROOT / "annotations_5cat.json",
    "RiSID_7class_clean": ROOT / "annotations_7cat.json",
}

random.seed(42)


def reset_dataset(out_dir):
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_image(file_name):
    candidates = list(IMAGE_ROOT.rglob(Path(file_name).name))
    if len(candidates) == 0:
        return None
    return candidates[0]


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + bw / 2) / img_w
    yc = (y1 + bh / 2) / img_h
    bw = bw / img_w
    bh = bh / img_h

    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        return None

    return xc, yc, bw, bh


def convert(task_name, ann_path):
    out_dir = OUTPUT_ROOT / task_name
    reset_dataset(out_dir)

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_new_id = {cat["id"]: i for i, cat in enumerate(categories)}
    names = {i: cat["name"] for i, cat in enumerate(categories)}

    images = {img["id"]: img for img in data["images"]}

    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    valid_items = []
    skipped_missing_image = 0
    skipped_invalid_box = 0

    for img_id, img in images.items():
        img_path = find_image(img["file_name"])
        if img_path is None:
            skipped_missing_image += 1
            continue

        img_w = img["width"]
        img_h = img["height"]

        label_lines = []

        for ann in anns_by_img.get(img_id, []):
            if ann["category_id"] not in cat_id_to_new_id:
                continue

            yolo_box = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            if yolo_box is None:
                skipped_invalid_box += 1
                continue

            cls_id = cat_id_to_new_id[ann["category_id"]]
            xc, yc, bw, bh = yolo_box
            label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if len(label_lines) == 0:
            continue

        valid_items.append((img_path, Path(img["file_name"]).stem, label_lines))

    random.shuffle(valid_items)

    n = len(valid_items)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)

    splits = {
        "train": valid_items[:n_train],
        "val": valid_items[n_train:n_train + n_val],
        "test": valid_items[n_train + n_val:],
    }

    for split, items in splits.items():
        for img_path, stem, label_lines in items:
            img_dst = out_dir / "images" / split / img_path.name
            label_dst = out_dir / "labels" / split / f"{Path(img_path.name).stem}.txt"

            shutil.copy2(img_path, img_dst)

            with open(label_dst, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.as_posix()}\n\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for i, name in names.items():
            safe_name = str(name).replace(" ", "_")
            f.write(f"  {i}: {safe_name}\n")

    print("\nDONE:", task_name)
    print("Annotation:", ann_path)
    print("Classes:", names)
    print("Train:", len(splits["train"]))
    print("Val:", len(splits["val"]))
    print("Test:", len(splits["test"]))
    print("Missing images:", skipped_missing_image)
    print("Invalid boxes skipped:", skipped_invalid_box)
    print("YAML:", yaml_path)


if __name__ == "__main__":
    for task_name, ann_path in TASKS.items():
        convert(task_name, ann_path)