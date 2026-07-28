#!/usr/bin/env python3
"""Report held-out performance for a trained PCam checkpoint.

Picks the operating threshold on the *valid* split (never test), then reports
final metrics on *test* at that threshold. The threshold is chosen to
guarantee a minimum sensitivity, since a missed tumor (false negative) is far
costlier than a false alarm for a screening task like this one -- see
docs/pathml-pipeline-project.md.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pathml.data.dataset import PCamDataset
from pathml.models.classifier import build_classifier
from pathml.training.metrics import metrics_at_threshold, scores_and_labels, select_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--checkpoint", default="models/pcam_resnet18.pt")
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-sensitivity", type=float, default=0.95,
                         help="minimum tumor recall the chosen threshold must guarantee")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_classifier(args.model_name, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    valid_loader = DataLoader(PCamDataset(data_dir, "valid"), batch_size=args.batch_size)
    valid_scores, valid_labels = scores_and_labels(model, valid_loader, device)
    threshold = select_threshold(valid_scores, valid_labels, min_sensitivity=args.min_sensitivity)
    print(f"selected threshold={threshold:.4f} on valid (min sensitivity={args.min_sensitivity})")
    print(f"  valid: {metrics_at_threshold(valid_scores, valid_labels, threshold)}")

    test_loader = DataLoader(PCamDataset(data_dir, "test"), batch_size=args.batch_size)
    test_scores, test_labels = scores_and_labels(model, test_loader, device)
    test_metrics = metrics_at_threshold(test_scores, test_labels, threshold)
    print(f"  test:  {test_metrics}")


if __name__ == "__main__":
    main()
