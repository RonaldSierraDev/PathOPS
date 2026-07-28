#!/usr/bin/env python3
"""Minimal fine-tuning loop for the PCam binary classifier."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pathml.data.dataset import PCamDataset
from pathml.models.classifier import build_classifier


def train(data_dir: Path, epochs: int, batch_size: int, lr: float, model_name: str, out_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PCamDataset(data_dir, "train")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = build_classifier(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        print(f"epoch {epoch + 1}/{epochs}  loss={running_loss / len(train_ds):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved checkpoint to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--out", default="models/pcam_resnet18.pt")
    args = parser.parse_args()

    train(Path(args.data_dir), args.epochs, args.batch_size, args.lr, args.model_name, Path(args.out))


if __name__ == "__main__":
    main()
