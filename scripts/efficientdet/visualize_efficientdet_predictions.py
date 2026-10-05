import os
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
from effdet import create_model

MODEL_PATH = r"D:\PLADICS\experiments_efficientdet_d0\efficientdet_d0_final.pth"
IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\test"
LABEL_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\labels\test"
OUT_DIR = r"D:\PLADICS\efficientdet_debug_predictions"

IMAGE_SIZE = 512
TOP_K = 10

os.makedirs(OUT_DIR, exist_ok=True)

model = create_model(
    "tf_efficientdet_d0",
    bench_task="predict",
    num_classes=1,
    pretrained=False
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
model.eval()


def draw_gt(img, label_path):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue

            _, xc, yc, bw, bh = map(float, p)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GT", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:20]

with torch.no_grad():
    for name in files:
        img_path = os.path.join(IMG_DIR, name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(name)[0] + ".txt")

        pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil.size

        resized = pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = F.to_tensor(resized).unsqueeze(0)

        output = model(tensor)[0].cpu()

        img = cv2.imread(img_path)
        img = draw_gt(img, label_path)

        if output.numel() > 0:
            output = output[output[:, 4].argsort(descending=True)]
            output = output[:TOP_K]

            for det in output:
                x1, y1, x2, y2, score, cls = det.tolist()

                x1 = int(x1 * orig_w / IMAGE_SIZE)
                x2 = int(x2 * orig_w / IMAGE_SIZE)
                y1 = int(y1 * orig_h / IMAGE_SIZE)
                y2 = int(y2 * orig_h / IMAGE_SIZE)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"P {score:.6f}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        save_path = os.path.join(OUT_DIR, "debug_" + name)
        cv2.imwrite(save_path, img)
        print("Saved:", save_path)

print("Done.")