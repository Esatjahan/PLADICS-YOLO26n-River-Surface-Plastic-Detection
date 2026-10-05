import os
import json
import cv2
from pathlib import Path

DATASET_ROOT = Path("D:/PLADICS/datasets_clean/merged_clean_dataset")
OUTPUT_DIR = DATASET_ROOT / "annotations"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["plastic"]


def convert_split(split):
    img_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    output_json = OUTPUT_DIR / f"instances_{split}.json"

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        label_path = label_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id, x_c, y_c, bw, bh = map(float, parts)

                x_c *= w
                y_c *= h
                bw *= w
                bh *= h

                x_min = x_c - bw / 2
                y_min = y_c - bh / 2

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                bw = min(bw, w - x_min)
                bh = min(bh, h - y_min)

                if bw <= 1 or bh <= 1:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [x_min, y_min, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })

                ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "plastic", "supercategory": "plastic"}
        ]
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"{split} done")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Saved: {output_json}")


for split in ["train", "val", "test"]:
    convert_split(split)