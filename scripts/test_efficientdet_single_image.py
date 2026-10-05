import torch
from PIL import Image
from torchvision.transforms import functional as F
from effdet import create_model

MODEL_PATH = r"D:\PLADICS\experiments_efficientdet_d0\efficientdet_d0_final.pth"

# test image
IMAGE_PATH = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test\riverine_test(106).jpg"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 512

model = create_model(
    "tf_efficientdet_d0",
    bench_task="predict",
    num_classes=1,
    pretrained=False
)

state = torch.load(MODEL_PATH, map_location=DEVICE)

missing, unexpected = model.load_state_dict(state, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.to(DEVICE)
model.eval()

img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(tensor)

print("\nOutput type:")
print(type(outputs))

print("\nOutput:")
print(outputs)

if len(outputs) > 0:
    print("\nFirst detections:")
    print(outputs[0][:20])