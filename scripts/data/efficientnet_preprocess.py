import os
import cv2
import torch
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T

INPUT_DIR = "D:/PLADICS/merged_dataset_v2/images/train"
OUTPUT_DIR = "D:/PLADICS/merged_dataset_efficient/images/train"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load EfficientNet as feature extractor
model = EfficientNet.from_pretrained('efficientnet-b0')  # You can use b1, b2 for more capacity
model.eval()

transform = T.Compose([T.ToTensor()])

# Process images
for img_name in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(path)
    if img is None:
        continue

    img_resized = cv2.resize(img, (640, 640))
    tensor = transform(img_resized).unsqueeze(0)

    with torch.no_grad():
        feat = model.extract_features(tensor)

    feat = torch.nn.functional.interpolate(feat, size=(640, 640))

    enhanced = tensor + 0.05 * feat[:, :3, :, :]
    enhanced_img = (enhanced.squeeze().permute(1,2,0).numpy() * 255).astype('uint8')
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), enhanced_img)

print("Hybrid dataset with EfficientNet features created successfully!")