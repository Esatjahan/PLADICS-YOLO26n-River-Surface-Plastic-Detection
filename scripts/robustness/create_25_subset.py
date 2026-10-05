from pathlib import Path
import random
import shutil

random.seed(42)

PERCENT = 0.25
NAME = "25"

SRC = Path(r"D:/PLADICS/datasets_clean/merged_clean_dataset")

DST = Path(
    fr"D:/PLADICS/datasets_clean/merged_clean_dataset_{NAME}"
)

# delete old folder if exists
if DST.exists():
    shutil.rmtree(DST)

# create folders
for split in ["train", "val", "test"]:

    (DST / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (DST / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# reduce train
train_images = list(
    (SRC / "images/train").glob("*.*")
)

random.shuffle(train_images)

keep_count = int(len(train_images) * PERCENT)

selected = train_images[:keep_count]

print(f"Original train images: {len(train_images)}")
print(f"Keeping: {keep_count}")

for img_path in selected:

    label_path = (
        SRC / "labels/train" / f"{img_path.stem}.txt"
    )

    shutil.copy2(
        img_path,
        DST / "images/train" / img_path.name
    )

    shutil.copy2(
        label_path,
        DST / "labels/train" / label_path.name
    )

# copy full val/test
for split in ["val", "test"]:

    for img_path in (
        SRC / f"images/{split}"
    ).glob("*.*"):

        label_path = (
            SRC /
            f"labels/{split}/{img_path.stem}.txt"
        )

        shutil.copy2(
            img_path,
            DST / f"images/{split}/{img_path.name}"
        )

        shutil.copy2(
            label_path,
            DST / f"labels/{split}/{label_path.name}"
        )

print("DONE")