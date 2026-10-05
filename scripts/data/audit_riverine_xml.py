import os
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = r"D:\PLADICS\datasets\Riverine Plastic Litter Dataset\Riverine Plastic Litter Dataset\5 categories"
ANNOTATIONS_DIR = os.path.join(ROOT, "Annotations")

PLASTIC_CLASSES = {
    "plastic bag",
    "plastic bottle",
    "plastic box",
    "plastic cup",
}

NON_PLASTIC_CLASSES = {"can"}

class_counts = Counter()
bad_boxes = []
unknown_classes = set()
total_objects = 0
plastic_objects = 0
nonplastic_objects = 0

for file in os.listdir(ANNOTATIONS_DIR):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, file)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print("XML parse error:", xml_path, e)
        continue

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    for obj in root.findall("object"):
        total_objects += 1

        cls = obj.find("name").text.strip()
        class_counts[cls] += 1

        if cls in PLASTIC_CLASSES:
            plastic_objects += 1
        elif cls in NON_PLASTIC_CLASSES:
            nonplastic_objects += 1
        else:
            unknown_classes.add(cls)

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        if (
            xmin < 0 or ymin < 0 or
            xmax > img_w or ymax > img_h or
            xmax <= xmin or ymax <= ymin
        ):
            bad_boxes.append((xml_path, cls, xmin, ymin, xmax, ymax, img_w, img_h))

print("\nRIVERINE XML AUDIT RESULT\n")
print("Total objects:", total_objects)
print("Plastic objects:", plastic_objects)
print("Non-plastic objects:", nonplastic_objects)
print("Class counts:")
for k, v in class_counts.items():
    print(f"  {k}: {v}")

print("\nUnknown classes:", sorted(unknown_classes))
print("Bad XML boxes:", len(bad_boxes))

print("\nFirst 20 bad boxes:")
for item in bad_boxes[:20]:
    print(item)