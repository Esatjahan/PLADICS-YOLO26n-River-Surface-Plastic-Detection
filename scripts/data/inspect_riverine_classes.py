import os
import xml.etree.ElementTree as ET

ANNOTATIONS_DIR = r"D:\PLADICS\datasets\Riverine Plastic Litter Dataset\Riverine Plastic Litter Dataset\5 categories\Annotations"

classes = set()

for file in os.listdir(ANNOTATIONS_DIR):
    if file.endswith(".xml"):
        xml_path = os.path.join(ANNOTATIONS_DIR, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            classes.add(name)

print("Riverine XML classes:")
for c in sorted(classes):
    print(c)