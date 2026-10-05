import json

ANNOTATION_FILE = r"D:\PLADICS\datasets\RiSID\annotations_2cat.json"

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

print(coco["categories"])

counts = {}
for ann in coco["annotations"]:
    cid = ann["category_id"]
    counts[cid] = counts.get(cid, 0) + 1

print(counts)