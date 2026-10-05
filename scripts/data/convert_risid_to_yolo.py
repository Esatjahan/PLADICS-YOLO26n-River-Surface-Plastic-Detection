import os
import json
import random
import shutil
from pathlib import Path

RISID_ROOT = r"E:\PLADICS\datasets\RiSID"
IMAGES_DIR = r"E:\PLADICS\datasets\RiSID\images\images"
ANNOTATION_FILE = r"E:\PLADICS\datasets\RiSID\annotations_2cat.json"

OUTPUT_ROOT = r"E:\PLADICS\merged_dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

# create folders
for split in ["train","val","test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT,"images",split),exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT,"labels",split),exist_ok=True)

# load json
with open(ANNOTATION_FILE,"r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

image_id_to_filename = {img["id"]:img["file_name"] for img in images}

ann_by_image = {}

for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in ann_by_image:
        ann_by_image[img_id] = []
    ann_by_image[img_id].append(ann)

all_images = list(image_id_to_filename.items())
random.shuffle(all_images)

n=len(all_images)

train_split=int(n*TRAIN_RATIO)
val_split=int(n*VAL_RATIO)

train_data=all_images[:train_split]
val_data=all_images[train_split:train_split+val_split]
test_data=all_images[train_split+val_split:]

splits={
"train":train_data,
"val":val_data,
"test":test_data
}

def coco_to_yolo_bbox(bbox,img_w,img_h):
    x,y,w,h=bbox

    x_center=(x+w/2)/img_w
    y_center=(y+h/2)/img_h
    w=w/img_w
    h=h/img_h

    return x_center,y_center,w,h

for split_name,data in splits.items():

    for image_id,file_name in data:

        src=os.path.join(IMAGES_DIR,file_name)
        dst=os.path.join(OUTPUT_ROOT,"images",split_name,file_name)

        if not os.path.exists(src):
            print("missing:",src)
            continue

        shutil.copy2(src,dst)

        img_info=[i for i in images if i["id"]==image_id][0]

        img_w=img_info["width"]
        img_h=img_info["height"]

        label_path=os.path.join(
            OUTPUT_ROOT,
            "labels",
            split_name,
            Path(file_name).stem+".txt"
        )

        with open(label_path,"w") as f:

            if image_id in ann_by_image:

                for ann in ann_by_image[image_id]:

                    bbox=ann["bbox"]

                    x,y,w,h=coco_to_yolo_bbox(bbox,img_w,img_h)

                    f.write(f"0 {x} {y} {w} {h}\n")

print("DONE")