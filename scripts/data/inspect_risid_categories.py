import json

json_path = r"D:\PLADICS\datasets\RiSID\annotations_7cat.json"

with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

cats = coco["categories"]

print("RiSID categories:")
for c in sorted(cats, key=lambda x: x["id"]):
    print(f"id={c['id']}  name={c['name']}")