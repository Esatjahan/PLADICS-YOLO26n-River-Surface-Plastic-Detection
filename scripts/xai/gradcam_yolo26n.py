import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"D:\PLADICS\experiments_clean_final\yolo26n_clean_100\weights\best.pt"
IMAGE_DIR = r"D:\PLADICS\gradcam_selected_inputs"
OUTPUT_DIR = r"D:\PLADICS\gradcam_outputs"

IMG_SIZE = 640
NUM_IMAGES = 14

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")

yolo = YOLO(MODEL_PATH)
model = yolo.model.to(device)
model.eval()

for p in model.parameters():
    p.requires_grad_(True)

target_layer = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        target_layer = module

if target_layer is None:
    raise RuntimeError("No Conv2d layer found.")

activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Image not found: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    x.requires_grad_(True)

    return img, x

def extract_score(output):
    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise RuntimeError("Model output is not tensor.")

    score = output.float().max()

    if not score.requires_grad:
        raise RuntimeError("Score still has no gradient.")

    return score

def make_gradcam(img_path):
    global activations, gradients
    activations = None
    gradients = None

    img, x = preprocess(img_path)

    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        output = model(x)
        score = extract_score(output)
        score.backward(retain_graph=True)

    if activations is None or gradients is None:
        raise RuntimeError("Hook failed: activation/gradient missing.")

    act = activations.detach()
    grad = gradients.detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)

    cam = cam.cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)

    return overlay, heatmap

def main():
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = files[:NUM_IMAGES]

    print(f"Generating Grad-CAM for {len(files)} images...")

    for i, f in enumerate(files, 1):
        path = os.path.join(IMAGE_DIR, f)
        base = os.path.splitext(f)[0]

        try:
            overlay, heatmap = make_gradcam(path)

            overlay_path = os.path.join(OUTPUT_DIR, base + "_gradcam_overlay.jpg")
            heatmap_path = os.path.join(OUTPUT_DIR, base + "_gradcam_heatmap.jpg")

            cv2.imwrite(overlay_path, overlay)
            cv2.imwrite(heatmap_path, heatmap)

            print(f"[OK] {i}/{len(files)} saved: {overlay_path}")

        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    print("Done.")
    print("Output:", OUTPUT_DIR)

if __name__ == "__main__":
    main()