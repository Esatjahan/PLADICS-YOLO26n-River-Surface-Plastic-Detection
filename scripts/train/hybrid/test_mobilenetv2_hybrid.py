import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"D:\PLADICS")
sys.path.append(str(PROJECT_ROOT))

from models_custom.mobilenetv2_hybrid.mobilenetv2_yolo26_hybrid import MobileNetV2YOLOHybrid


def main():
    model = MobileNetV2YOLOHybrid(num_classes=1, pretrained_backbone=True)
    model.eval()

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)

    print("Model loaded successfully.")
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()