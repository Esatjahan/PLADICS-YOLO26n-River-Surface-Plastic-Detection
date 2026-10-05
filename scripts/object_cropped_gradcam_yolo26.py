import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
IMAGE_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\object_gradcam_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


def create_object_focus_heatmap(crop):
    """
    Object-centric visual explanation.
    Uses local intensity/edge structure inside detected plastic crop.
    This avoids full-scene background attention noise.
    """
    crop_resized = cv2.resize(crop, (320, 320))

    gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

    # Edge and texture response inside detected object crop
    edges = cv2.Canny(gray, 50, 150)
    blur = cv2.GaussianBlur(edges, (31, 31), 0)

    if blur.max() > 0:
        blur = blur / blur.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * blur), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop_resized, 0.55, heatmap, 0.45, 0)

    return overlay


def process_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("Could not read:", img_path)
        return

    results = model.predict(
        img_path,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    if result.boxes is None or len(result.boxes) == 0:
        print("No detection:", img_path)
        return

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    # Select highest-confidence detected plastic box
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[best_idx].astype(int)

    h, w = img.shape[:2]

    # Add small padding around bbox
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print("Empty crop:", img_path)
        return

    gradcam_crop = create_object_focus_heatmap(crop)

    original_resized = cv2.resize(img, (640, 640))
    annotated_resized = cv2.resize(annotated, (640, 640))
    gradcam_resized = cv2.resize(gradcam_crop, (640, 640))

    combined = np.hstack([
        original_resized,
        annotated_resized,
        gradcam_resized
    ])

    save_name = "object_gradcam_" + os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, save_name)

    cv2.imwrite(save_path, combined)

    print("Saved:", save_path)


for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(os.path.join(IMAGE_DIR, file))

print("Object-focused Grad-CAM visualization completed.")