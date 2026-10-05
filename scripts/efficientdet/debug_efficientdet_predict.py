import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
from effdet import create_model

MODEL_PATH = r"D:\PLADICS\experiments_efficientdet_d0\efficientdet_d0_final.pth"
IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 512

model = create_model(
    "tf_efficientdet_d0",
    bench_task="predict",
    num_classes=1,
    pretrained=False
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.to(DEVICE)
model.eval()

image_files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:5]

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)

        output = model(tensor)

        print("\nImage:", img_name)
        print("Output type:", type(output))
        print("Output shape:", output.shape if torch.is_tensor(output) else "not tensor")
        print("Output sample:", output[0][:10] if torch.is_tensor(output) else output)