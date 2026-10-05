import os
import cv2

image_dir = 'D:/PLADICS/merged_dataset_v2/images/train/'
label_dir = 'D:/PLADICS/merged_dataset_v2/labels/train/'

image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Check if images and labels match
for image_file in image_files:
    label_file = image_file.replace(".jpg", ".txt")
    if label_file not in label_files:
        print(f"Missing label for image: {image_file}")

    # Check if image can be opened
    img = cv2.imread(os.path.join(image_dir, image_file))
    if img is None:
        print(f"Corrupt image found: {image_file}")

print("Dataset check complete")