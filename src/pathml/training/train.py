#!/usr/bin/env python3
"""Minimal fine-tuning loop for the PCam binary classifier."""
import argparse
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from pathml.data.dataset import PCamDataset
from pathml.models.classifier import build_classifier
from pathml.training.feedback import FeedbackDataset
from pathml.training.metrics import evaluate

EXPERIMENT_NAME = "pcam-classification"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "pcam-resnet18"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    model_name: str,
    out_path: Path,
    pretrained: bool = True,
    seed: int = 42,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    feedback_export_url: str | None = None,
    feedback_images_s3_bucket: str | None = None,
) -> None:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PCamDataset(data_dir, "train")
    feedback_size = 0
    # The design doc's "close the loop" requirement: retraining should learn
    # from pathologist corrections logged via the API's /feedback endpoint,
    # not just the static PCam split.
    if feedback_export_url:
        feedback_ds = FeedbackDataset(feedback_export_url, feedback_images_s3_bucket)
        feedback_size = len(feedback_ds)
        if feedback_size:
            train_ds = ConcatDataset([train_ds, feedback_ds])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_ds = PCamDataset(data_dir, "valid")
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=4)

    model = build_classifier(model_name, pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "model_name": model_name,
            "pretrained": pretrained,
            "seed": seed,
            "train_size": len(train_ds),
            "valid_size": len(valid_ds),
            "feedback_size": feedback_size,
        })

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_ds)
            val_metrics = evaluate(model, valid_loader, device)
            print(f"epoch {epoch + 1}/{epochs}  loss={train_loss:.4f}  valid: {val_metrics}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "valid_accuracy": val_metrics.accuracy,
                "valid_auc": val_metrics.auc,
                "valid_sensitivity": val_metrics.sensitivity,
                "valid_specificity": val_metrics.specificity,
            }, step=epoch)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"saved checkpoint to {out_path}")
        mlflow.log_artifact(str(out_path))

        model_info = mlflow.pytorch.log_model(
            model, name="model", registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )
        version = model_info.registered_model_version
        MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "staging", version)
        print(f"registered {REGISTERED_MODEL_NAME} v{version}, aliased 'staging'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--out", default="models/pcam_resnet18.pt")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                         help="train the backbone from random init instead of pretrained weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                         help="MLflow tracking store, e.g. sqlite:///mlflow.db")
    parser.add_argument("--feedback-export-url", default=None,
                         help="e.g. http://<api-host>:8000/feedback/export -- pulls pathologist corrections "
                              "from the running API; omit to train on the static PCam split only")
    parser.add_argument("--feedback-images-s3-bucket", default=None,
                         help="S3 bucket the API stored corrected predictions' images in; required if "
                              "--feedback-export-url is set")
    args = parser.parse_args()
    if args.feedback_export_url and not args.feedback_images_s3_bucket:
        parser.error("--feedback-images-s3-bucket is required when --feedback-export-url is set")

    train(Path(args.data_dir), args.epochs, args.batch_size, args.lr, args.model_name, Path(args.out),
          pretrained=args.pretrained, seed=args.seed, tracking_uri=args.tracking_uri,
          feedback_export_url=args.feedback_export_url, feedback_images_s3_bucket=args.feedback_images_s3_bucket)


if __name__ == "__main__":
    main()
