import os
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

DATA_ROOT = r"D:\PLADICS\datasets_clean\merged_clean_dataset"
OUTPUT_DIR = r"D:\PLADICS\experiments_mobilenetssd_correct"

EPOCHS = 50
BATCH_SIZE = 8
DEVICE = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class YOLODataset(Dataset):
    def __init__(self, root, split):
        self.img_dir = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "labels", split)
        self.images = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, xc, yc, bw, bh = map(float, parts)

                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(1)  # class 1 = plastic, class 0 = background

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return F.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    train_dataset = YOLODataset(DATA_ROOT, "train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Correct MobileNet-SSD setup:
    # ImageNet-pretrained MobileNetV3 backbone + new SSD detection head for 2 classes.
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=2
    )

    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    print("Starting corrected MobileNet-SSD training...")
    print("Train images:", len(train_dataset))

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Iter [{i}/{len(train_loader)}] Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = (time.time() - start_time) / 60

        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f} min")

        save_path = os.path.join(OUTPUT_DIR, f"mobilenetssd_correct_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print("Saved:", save_path)

    final_path = os.path.join(OUTPUT_DIR, "mobilenetssd_correct_final.pth")
    torch.save(model.state_dict(), final_path)
    print("Final model saved:", final_path)


if __name__ == "__main__":
    main()