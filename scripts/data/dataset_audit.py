import os
from PIL import Image

DATASET_PATH = r"D:/PLADICS/datasets_clean/RiSID_7class_clean"

IMAGE_DIRS = [
    os.path.join(DATASET_PATH, "images", "train"),
    os.path.join(DATASET_PATH, "images", "val"),
    os.path.join(DATASET_PATH, "images", "test"),
]

LABEL_DIRS = [
    os.path.join(DATASET_PATH, "labels", "train"),
    os.path.join(DATASET_PATH, "labels", "val"),
    os.path.join(DATASET_PATH, "labels", "test"),
]

total_images = 0
total_labels = 0
missing_labels = 0
empty_labels = 0
corrupt_images = 0
invalid_bbox = 0
invalid_class_id = 0
out_of_range_bbox = 0

print("🔍 Starting dataset audit...\n")

for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
    image_files = os.listdir(img_dir)

    for img_file in image_files:
        total_images += 1

        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(lbl_dir, label_file)

        # --- 1. Corrupt image check ---
        try:
            with Image.open(img_path) as img:
                img.verify()
        except:
            corrupt_images += 1
            continue

        # --- 2. Missing label ---
        if not os.path.exists(label_path):
            missing_labels += 1
            continue

        total_labels += 1

        # --- 3. Empty label ---
        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            empty_labels += 1
            continue

        # --- 4. Label content check ---
        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                invalid_bbox += 1
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except:
                invalid_bbox += 1
                continue

            # --- 5. Class ID check ---
            if class_id != 0:
                invalid_class_id += 1

            # --- 6. Range check ---
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                out_of_range_bbox += 1

            # --- 7. Zero area check ---
            if w <= 0 or h <= 0:
                invalid_bbox += 1

print("\n DATASET AUDIT RESULT:\n")
print(f"Total Images: {total_images}")
print(f"Total Labels: {total_labels}")
print(f"Missing Labels: {missing_labels}")
print(f"Empty Labels: {empty_labels}")
print(f"Corrupt Images: {corrupt_images}")
print(f"Invalid BBox Format: {invalid_bbox}")
print(f"Invalid Class IDs (!=0): {invalid_class_id}")
print(f"Out-of-Range BBoxes: {out_of_range_bbox}")