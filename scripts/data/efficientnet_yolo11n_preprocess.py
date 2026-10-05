import os
import cv2
import torch
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet

BASE_INPUT = r"D:/PLADICS/merged_dataset_v2/images"
BASE_OUTPUT = r"D:/PLADICS/merged_dataset_yolo11n_efficient/images"
SPLITS = ["train", "val", "test"]

# Load EfficientNet feature extractor
model = EfficientNet.from_pretrained("efficientnet-b0")
model.eval()

transform = T.Compose([
    T.ToTensor()
])

for split in SPLITS:
    input_dir = os.path.join(BASE_INPUT, split)
    output_dir = os.path.join(BASE_OUTPUT, split)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_resized = cv2.resize(img, (640, 640))
        tensor = transform(img_resized).unsqueeze(0)

        with torch.no_grad():
            feat = model.extract_features(tensor)

        feat = torch.nn.functional.interpolate(
            feat, size=(640, 640), mode="bilinear", align_corners=False
        )

        feat_rgb = feat[:, :3, :, :]

        feat_min = feat_rgb.amin(dim=(2, 3), keepdim=True)
        feat_max = feat_rgb.amax(dim=(2, 3), keepdim=True)
        feat_rgb = (feat_rgb - feat_min) / (feat_max - feat_min + 1e-6)

        enhanced = tensor + 0.05 * feat_rgb
        enhanced = torch.clamp(enhanced, 0, 1)

        enhanced_img = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        cv2.imwrite(os.path.join(output_dir, img_name), enhanced_img)

    print(f"{split} split processed.")

print("EfficientNet + YOLO11n hybrid dataset created successfully.")