import os

DATASET_PATH = r"D:/PLADICS/merged_dataset"

LABEL_DIRS = [
    os.path.join(DATASET_PATH, "labels", "train"),
    os.path.join(DATASET_PATH, "labels", "val"),
    os.path.join(DATASET_PATH, "labels", "test"),
]

bad_files = []

for lbl_dir in LABEL_DIRS:
    for file in os.listdir(lbl_dir):
        path = os.path.join(lbl_dir, file)

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                bad_files.append(path)
                break

            _, x, y, w, h = map(float, parts)

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                bad_files.append(path)
                break

print(f"\n❌ Bad label files: {len(set(bad_files))}\n")

for f in list(set(bad_files))[:20]:
    print(f)