import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2  # fallback

# NOTE: GhostNetV2 official torch version নেই by default
# তাই আমরা lightweight approximation use করবো

BASE_INPUT = "D:/PLADICS/merged_dataset_v2/images"
BASE_OUTPUT = "D:/PLADICS/merged_dataset_ghost/images"
SPLITS = ["train", "val", "test"]

model = mobilenet_v2(weights="DEFAULT").features
model.eval()

transform = T.Compose([
    T.ToTensor()
])

for split in SPLITS:
    input_dir = os.path.join(BASE_INPUT, split)
    output_dir = os.path.join(BASE_OUTPUT, split)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        path = os.path.join(input_dir, img_name)
        img = cv2.imread(path)

        if img is None:
            continue

        img_resized = cv2.resize(img, (640, 640))
        tensor = transform(img_resized).unsqueeze(0)

        with torch.no_grad():
            feat = model(tensor)

        feat = torch.nn.functional.interpolate(feat, size=(640, 640))

        # slightly stronger fusion than MobileNet test
        enhanced = tensor + 0.08 * feat[:, :3, :, :]

        enhanced = torch.clamp(enhanced, 0, 1)
        enhanced_img = (enhanced.squeeze().permute(1, 2, 0).numpy() * 255).astype("uint8")

        cv2.imwrite(os.path.join(output_dir, img_name), enhanced_img)

    print(f"{split} processed")

print("GhostNet-style hybrid dataset ready")