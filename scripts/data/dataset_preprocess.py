import os
import cv2
import shutil
import torch
import torchvision.transforms as T

# Define dataset paths
input_train_dir = "D:/PLADICS/merged_dataset_cv/images/train"
output_train_dir = "D:/PLADICS/merged_dataset_cv/images/train_processed"

input_val_dir = "D:/PLADICS/merged_dataset_cv/images/val"
output_val_dir = "D:/PLADICS/merged_dataset_cv/images/val_processed"

# Create output directories if they don't exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)

# Load any model or set up transformations for preprocessing (e.g., EfficientNet or MobileNet)
transform = T.Compose([
    T.ToTensor(),
    T.Resize((640, 640)),  # Resize to 640x640 for YOLO training
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
])

# Preprocess and save the images
def preprocess_images(input_dir, output_dir):
    images = os.listdir(input_dir)
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        if img_name.endswith('.png'):  # Ensure you are processing .png images
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_resized = cv2.resize(img, (640, 640))  # Resize to match YOLO's expected input size
            tensor = transform(img_resized).unsqueeze(0)

            output_img = (tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype('uint8')
            cv2.imwrite(os.path.join(output_dir, img_name), output_img)
        else:
            print(f"Skipping invalid image: {img_name}")

# Run preprocessing for train and val datasets
preprocess_images(input_train_dir, output_train_dir)
preprocess_images(input_val_dir, output_val_dir)

print("Preprocessing complete!")