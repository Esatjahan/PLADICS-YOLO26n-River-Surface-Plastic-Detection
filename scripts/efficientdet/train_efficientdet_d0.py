import os
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from effdet import create_model

DATA_ROOT = r"D:\PLADICS\datasets_clean\merged_clean_dataset"
OUTPUT_DIR = r"D:\PLADICS\experiments_efficientdet_d0"

EPOCHS = 50
BATCH_SIZE = 2
IMAGE_SIZE = 512
DEVICE = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class PlasticDataset(Dataset):
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

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, xc, yc, bw, bh = map(float, parts)

                    x1 = (xc - bw / 2) * orig_w
                    y1 = (yc - bh / 2) * orig_h
                    x2 = (xc + bw / 2) * orig_w
                    y2 = (yc + bh / 2) * orig_h

                    x1 = x1 * IMAGE_SIZE / orig_w
                    x2 = x2 * IMAGE_SIZE / orig_w
                    y1 = y1 * IMAGE_SIZE / orig_h
                    y2 = y2 * IMAGE_SIZE / orig_h

                    x1 = max(0, min(x1, IMAGE_SIZE - 1))
                    y1 = max(0, min(y1, IMAGE_SIZE - 1))
                    x2 = max(0, min(x2, IMAGE_SIZE - 1))
                    y2 = max(0, min(y2, IMAGE_SIZE - 1))

                    if x2 > x1 and y2 > y1:
                        # effdet target bbox format: y1, x1, y2, x2
                        boxes.append([y1, x1, y2, x2])
                        labels.append(0)  # num_classes=1, so plastic class index = 0

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = F.to_tensor(image)

        target = {
            "bbox": boxes,
            "cls": labels,
            "img_size": torch.tensor([IMAGE_SIZE, IMAGE_SIZE], dtype=torch.float32),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
        }

        return image, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def main():
    train_dataset = PlasticDataset(DATA_ROOT, "train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print("Train images:", len(train_dataset))

    model = create_model(
        "tf_efficientdet_d0",
        bench_task="train",
        num_classes=1,
        pretrained=True,
        bench_labeler=True
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    print("\nStarting EfficientDet-D0 training...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)

            formatted_targets = {
                "bbox": [t["bbox"].to(DEVICE) for t in targets],
                "cls": [t["cls"].to(DEVICE) for t in targets],
                "img_size": torch.stack([t["img_size"] for t in targets]).to(DEVICE),
                "img_scale": torch.stack([t["img_scale"] for t in targets]).to(DEVICE),
            }

            output = model(images, formatted_targets)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Iter [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        elapsed = (time.time() - start_time) / 60

        print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f} min\n")

        save_path = os.path.join(OUTPUT_DIR, f"efficientdet_d0_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print("Saved:", save_path)

    final_path = os.path.join(OUTPUT_DIR, "efficientdet_d0_final.pth")
    torch.save(model.state_dict(), final_path)
    print("\nFinal model saved:", final_path)


if __name__ == "__main__":
    main()