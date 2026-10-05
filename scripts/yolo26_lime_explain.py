import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from lime import lime_image
from skimage.segmentation import mark_boundaries


MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
IMAGE_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\lime_outputs"

IMG_SIZE = 640
CONF_THRES = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


def yolo_lime_predict(images):
    """
    LIME expects classifier-like output: [not_plastic_score, plastic_score].
    Here plastic_score = highest YOLO plastic confidence in image.
    """
    outputs = []

    for img in images:
        img_uint8 = np.uint8(img)

        results = model.predict(
            img_uint8,
            imgsz=IMG_SIZE,
            conf=0.001,
            verbose=False,
            device="cpu"
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            plastic_score = 0.0
        else:
            plastic_score = float(boxes.conf.max().cpu().item())

        plastic_score = max(0.0, min(1.0, plastic_score))
        outputs.append([1.0 - plastic_score, plastic_score])

    return np.array(outputs)


def process_image(img_path):
    bgr = cv2.imread(img_path)

    if bgr is None:
        print("Could not read:", img_path)
        return

    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # YOLO detection image
    det = model.predict(
        rgb,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        verbose=False,
        device="cpu"
    )[0].plot()

    det = cv2.resize(det, (IMG_SIZE, IMG_SIZE))

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        rgb,
        yolo_lime_predict,
        top_labels=2,
        hide_color=0,
        num_samples=1000,
        batch_size=8
    )

    temp, mask = explanation.get_image_and_mask(
        label=1,
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    lime_vis = mark_boundaries(temp / 255.0, mask)
    lime_vis = np.uint8(255 * lime_vis)
    lime_vis = cv2.cvtColor(lime_vis, cv2.COLOR_RGB2BGR)

    combined = np.hstack([
        bgr,
        det,
        lime_vis
    ])

    save_path = os.path.join(
        OUTPUT_DIR,
        "lime_" + os.path.basename(img_path)
    )

    cv2.imwrite(save_path, combined)
    print("Saved:", save_path)


for img_name in os.listdir(IMAGE_DIR):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(os.path.join(IMAGE_DIR, img_name))

print("YOLO26 LIME explanation completed.")