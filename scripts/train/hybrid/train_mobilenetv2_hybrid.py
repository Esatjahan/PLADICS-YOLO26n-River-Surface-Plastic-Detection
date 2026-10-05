import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\PLADICS")
sys.path.append(str(PROJECT_ROOT))

from models_custom.mobilenetv2_hybrid.mobilenetv2_yolo26_hybrid import MobileNetV2YOLOHybrid


# Simple dataset (classification-style proxy)
class SimpleDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.image_paths.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        # dummy target
        target = torch.zeros((5, 20, 20))

        return img, target


def main():
    device = torch.device("cpu")

    model = MobileNetV2YOLOHybrid(num_classes=1).to(device)

    dataset = SimpleDataset("D:/PLADICS/merged_dataset_v2/images/train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print("Starting training...")

    for epoch in range(2):
        total_loss = 0

        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)

            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    main()