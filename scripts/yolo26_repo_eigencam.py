import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(r"D:\PLADICS\scripts")

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
IMAGE_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\yolo26_cam_repo_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


target_layers = [model.model.model[22]]

cam = EigenCAM(
    model=model,
    target_layers=target_layers,
    task="od"
)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)

    bgr = cv2.imread(img_path)
    if bgr is None:
        print("Could not read:", img_path)
        continue

    bgr_resized = cv2.resize(bgr, (640, 640))
    rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    rgb_float = np.float32(rgb) / 255.0

    grayscale_cam = cam(rgb_float)[0, :, :]

    cam_image = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    pred = model.predict(
        img_path,
        imgsz=640,
        conf=0.25,
        verbose=False
    )[0].plot()
    pred = cv2.resize(pred, (640, 640))

    combined = np.hstack([bgr_resized, pred, cam_image])

    save_path = os.path.join(OUTPUT_DIR, "repo_eigencam_" + img_name)
    cv2.imwrite(save_path, combined)

    print("Saved:", save_path)

print("YOLO-26-CAM repo EigenCAM completed.")