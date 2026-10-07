import os
import time
import cv2
import torch

from nanodet.util import cfg, load_config, Logger, load_model_weight
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline

CONFIG_PATH = r"D:\nanodet\config\pladics_nanodet_official.yml"
MODEL_PATH = r"D:\nanodet\workspace\pladics_nanodet_640_fair\model_best\model_best.ckpt"
SIZE_PATH = r"D:\nanodet\workspace\pladics_nanodet_640_fair\model_best\nanodet_model_best.pth"
IMG_DIR = r"D:\PLADICS\datasets_clean\merged_clean_dataset\images\val"

DEVICE = "cpu"
WARMUP = 20

load_config(cfg, CONFIG_PATH)
device = torch.device(DEVICE)
logger = Logger(-1, use_tensorboard=False)

model = build_model(cfg.model)
ckpt = torch.load(MODEL_PATH, map_location=device)
load_model_weight(model, ckpt, logger)

model.to(device)
model.eval()

pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

images = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Images:", len(images))

times = []

with torch.no_grad():
    for i, name in enumerate(images):
        img_path = os.path.join(IMG_DIR, name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        meta = {
            "img_info": {
                "id": i,
                "file_name": name,
                "height": img.shape[0],
                "width": img.shape[1],
            },
            "raw_img": img,
            "img": img,
        }

        meta = pipeline(None, meta, cfg.data.val.input_size)

        img_tensor = torch.from_numpy(
            meta["img"].transpose(2, 0, 1)
        ).unsqueeze(0).float().to(device)

        start = time.perf_counter()
        _ = model(img_tensor)
        end = time.perf_counter()

        if i >= WARMUP:
            times.append(end - start)

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time
size_mb = os.path.getsize(SIZE_PATH) / (1024 * 1024)

print("\n========== NanoDet FPS Result ==========")
print(f"Average inference time: {avg_time * 1000:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Model size: {size_mb:.2f} MB")
print("=======================================")