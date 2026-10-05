import os
import time
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from sklearn.metrics import precision_score, recall_score

MODEL_PATH = r"D:\PLADICS\experiments_mobilenetssd\mobilenetssd_final.pth"
TEST_IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"
TEST_LABEL_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\labels\test"

DEVICE = torch.device("cpu")
CONF_THRES = 0.5

model = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
)

model.head.classification_head.num_classes = 2
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)

model.to(DEVICE)
model.eval()

image_files = [
    f for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

all_preds = []
all_targets = []

total_time = 0.0
num_images = 0

print("Running MobileNet-SSD evaluation...")
print("Test images:", len(image_files))

with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(TEST_IMG_DIR, img_name)

        img = Image.open(img_path).convert("RGB")
        tensor = F.to_tensor(img).unsqueeze(0).to(DEVICE)

        start = time.time()
        outputs = model(tensor)
        end = time.time()

        total_time += (end - start)
        num_images += 1

        output = outputs[0]
        scores = output["scores"].cpu().numpy()

        pred_detected = 1 if len(scores) > 0 and scores.max() >= CONF_THRES else 0

        label_path = os.path.join(
            TEST_LABEL_DIR,
            os.path.splitext(img_name)[0] + ".txt"
        )

        gt_detected = 1 if os.path.exists(label_path) and os.path.getsize(label_path) > 0 else 0

        all_preds.append(pred_detected)
        all_targets.append(gt_detected)

precision = precision_score(all_targets, all_preds, zero_division=0)
recall = recall_score(all_targets, all_preds, zero_division=0)

avg_inference_ms = (total_time / num_images) * 1000
fps = 1000 / avg_inference_ms

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("\n========== FINAL MobileNet-SSD RESULTS ==========")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("mAP@0.5: Not computed in this lightweight script")
print("mAP@0.5:0.95: Not computed in this lightweight script")
print(f"Average inference time: {avg_inference_ms:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Model size: {model_size_mb:.2f} MB")
print("================================================")