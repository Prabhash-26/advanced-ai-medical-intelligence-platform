from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from app.ml.model import build_model
from app.ml.preprocessing import IMAGE_SIZE, image_to_tensor


CLASSES = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]


class SyntheticChestDataset(Dataset):
    """Synthetic radiograph-like dataset for reproducible model bootstrapping."""

    def __init__(self, samples_per_class: int = 80, seed: int = 42) -> None:
        self.items: list[tuple[Image.Image, int]] = []
        rng = random.Random(seed)
        for label in range(len(CLASSES)):
            for _ in range(samples_per_class):
                self.items.append((draw_synthetic_chest(label, rng), label))
        rng.shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.items[index]
        return image_to_tensor(image).squeeze(0), label


def draw_synthetic_chest(label: int, rng: random.Random) -> Image.Image:
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=18)
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 38, 108, 184), fill=64, outline=92)
    draw.ellipse((116, 38, 182, 184), fill=64, outline=92)
    draw.rectangle((103, 28, 121, 190), fill=82)
    draw.arc((74, 22, 150, 92), 185, 355, fill=120, width=2)

    noise = np.random.default_rng(rng.randint(0, 999999)).normal(0, 8, (IMAGE_SIZE, IMAGE_SIZE))
    array = np.clip(np.array(image).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    draw = ImageDraw.Draw(image)

    if label == 1:
        for _ in range(8):
            x = rng.randint(48, 166)
            y = rng.randint(70, 158)
            draw.ellipse((x, y, x + rng.randint(12, 28), y + rng.randint(8, 20)), fill=rng.randint(118, 172))
    elif label == 2:
        for _ in range(12):
            x = rng.randint(44, 170)
            y = rng.randint(58, 170)
            draw.line((x, y, x + rng.randint(-20, 20), y + rng.randint(10, 24)), fill=155, width=2)
    elif label == 3:
        for _ in range(5):
            x = rng.randint(50, 156)
            y = rng.randint(45, 120)
            draw.ellipse((x, y, x + 18, y + 18), outline=178, width=3)

    return image.filter(ImageFilter.GaussianBlur(radius=0.7))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    dataset = SyntheticChestDataset(args.samples_per_class, args.seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        accuracy = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} loss={total_loss / train_size:.4f} val_accuracy={accuracy:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASSES,
            "image_size": IMAGE_SIZE,
            "validation_accuracy": evaluate(model, val_loader, device),
            "training_data": "Synthetic radiograph-like educational dataset. Replace with real approved dataset.",
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps({"classes": CLASSES, "artifact": str(output), "clinical_use": False}, indent=2),
        encoding="utf-8",
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = torch.argmax(model(images), dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / max(total, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/models/medical_cnn.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())

