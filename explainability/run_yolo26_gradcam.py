import os
import cv2
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
INPUT_DIR = r"D:\PLADICS\explainability\sample_images"
OUTPUT_DIR = r"D:\PLADICS\explainability\proper_gradcam"

IMG_SIZE = 640
CONF_THRES = 0.25
TARGET_LAYER_INDEX = -3

os.makedirs(os.path.join(OUTPUT_DIR, "original"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "detection"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gradcam"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "combined"), exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "gradcam_summary.csv")

print("Loading detection model...")
det_yolo = YOLO(MODEL_PATH)

print("Loading Grad-CAM model...")
cam_yolo = YOLO(MODEL_PATH)
cam_model = cam_yolo.model

for p in cam_model.parameters():
    p.requires_grad_(True)

cam_model.train()
for m in cam_model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

target_layer = cam_model.model[TARGET_LAYER_INDEX]
print("Using target layer:", TARGET_LAYER_INDEX)

saved_activation = None

def forward_hook(module, inp, out):
    global saved_activation
    if isinstance(out, (tuple, list)):
        out = out[0]
    saved_activation = out
    if hasattr(saved_activation, "requires_grad") and saved_activation.requires_grad:
        saved_activation.retain_grad()

target_layer.register_forward_hook(forward_hook)

def draw_boxes(img_rgb, boxes, confs):
    out = img_rgb.copy()
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            out, f"plastic {conf:.2f}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (0, 255, 0), 2, cv2.LINE_AA
        )
    return out

def get_detection(rgb_img):
    result = det_yolo.predict(
        source=rgb_img,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device="cpu",
        verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

def compute_gradcam(input_tensor, target_box):
    global saved_activation
    saved_activation = None

    cam_model.zero_grad(set_to_none=True)
    input_tensor.requires_grad_(True)

    with torch.enable_grad():
        _ = cam_model(input_tensor)

        if saved_activation is None:
            raise RuntimeError("No activation captured.")

        _, c, fh, fw = saved_activation.shape

        x1, y1, x2, y2 = target_box
        fx1 = int(max(0, min(fw - 1, x1 / IMG_SIZE * fw)))
        fy1 = int(max(0, min(fh - 1, y1 / IMG_SIZE * fh)))
        fx2 = int(max(fx1 + 1, min(fw, x2 / IMG_SIZE * fw)))
        fy2 = int(max(fy1 + 1, min(fh, y2 / IMG_SIZE * fh)))

        roi = saved_activation[:, :, fy1:fy2, fx1:fx2]
        score = roi.mean()

        if not score.requires_grad:
            raise RuntimeError("ROI score has no gradient.")

        score.backward(retain_graph=True)

    grads = saved_activation.grad
    if grads is None:
        raise RuntimeError("Activation gradient is None.")

    acts = saved_activation.detach()
    grads = grads.detach()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam

def overlay_cam(img_rgb, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heatmap, 0.45, 0)

def make_combined(original, detection, gradcam_img, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Test Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(detection)
    axes[1].set_title("YOLO26n Detection", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(gradcam_img)
    axes[2].set_title("ROI-targeted Grad-CAM", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} images.")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "detections", "selected_confidence", "status"])

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        bgr = cv2.imread(img_path)
        if bgr is None:
            writer.writerow([img_name, 0, 0, "read_error"])
            print("[ERROR] Cannot read:", img_name)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        rgb_float = np.float32(rgb) / 255.0

        boxes, confs = get_detection(rgb)

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "original", f"{base}_original.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

        if boxes is None or len(boxes) == 0:
            writer.writerow([img_name, 0, 0, "no_detection"])
            print("[SKIP] No detection:", img_name)
            continue

        best_idx = int(np.argmax(confs))
        target_box = boxes[best_idx]
        target_conf = float(confs[best_idx])

        detection_img = draw_boxes(rgb, boxes, confs)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "detection", f"{base}_detection.jpg"),
            cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
        )

        input_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).float()

        try:
            cam = compute_gradcam(input_tensor, target_box)
            gradcam_img = overlay_cam(rgb, cam)
            gradcam_img = draw_boxes(gradcam_img, boxes, confs)

            cv2.imwrite(
                os.path.join(OUTPUT_DIR, "gradcam", f"{base}_gradcam.jpg"),
                cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)
            )

            make_combined(
                rgb,
                detection_img,
                gradcam_img,
                os.path.join(OUTPUT_DIR, "combined", f"{base}_combined.jpg")
            )

            writer.writerow([img_name, len(boxes), target_conf, "success"])
            print(f"[DONE] {img_name} | det={len(boxes)} | selected_conf={target_conf:.3f}")

        except Exception as e:
            writer.writerow([img_name, len(boxes), target_conf, f"failed: {e}"])
            print(f"[FAILED] {img_name} | {e}")

print("\nDone.")
print("Saved to:", OUTPUT_DIR)