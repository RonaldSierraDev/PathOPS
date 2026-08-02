#!/usr/bin/env python3
"""Report held-out performance for a trained PCam checkpoint.

Picks the operating threshold on the *valid* split (never test), then reports
final metrics -- plus calibration (temperature scaling, ECE) and diagnostic
plots (confusion matrix, PR curve, reliability diagrams) -- on *test* at that
threshold. The threshold is chosen to guarantee a minimum sensitivity, since a
missed tumor (false negative) is far costlier than a false alarm for a
screening task like this one -- see docs/pathml-pipeline-project.md.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch

from pathml.models.classifier import build_classifier
from pathml.training.evaluation import evaluate_checkpoint
from pathml.training.train import DEFAULT_TRACKING_URI, EXPERIMENT_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--checkpoint", default="models/pcam_resnet18.pt")
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-sensitivity", type=float, default=0.95,
                         help="minimum tumor recall the chosen threshold must guarantee")
    parser.add_argument("--reports-dir", default="reports",
                         help="directory to write confusion matrix / PR curve / reliability plots to")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                         help="MLflow tracking store, e.g. sqlite:///mlflow.db")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_classifier(args.model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    result = evaluate_checkpoint(
        model, data_dir, batch_size=args.batch_size, min_sensitivity=args.min_sensitivity, device=device,
    )

    print(f"selected threshold={result.threshold:.4f} on valid (min sensitivity={args.min_sensitivity})")
    print(f"  valid: {result.valid_metrics}")
    print(f"  test:  {result.test_metrics}")
    print(f"  calibration: temperature={result.temperature:.4f}  "
          f"ece_before={result.ece_before:.4f}  ece_after={result.ece_after:.4f}")

    report_dir = Path(args.reports_dir) / checkpoint.stem
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}
    for name, fig in result.figures.items():
        path = report_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[name] = path
    print(f"saved plots to {report_dir}/")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(tags={"run_type": "evaluation"}):
        mlflow.log_params({
            "checkpoint": str(checkpoint),
            "model_name": args.model_name,
            "min_sensitivity": args.min_sensitivity,
            "threshold": result.threshold,
            "temperature": result.temperature,
        })
        mlflow.log_metrics({
            "test_accuracy": result.test_metrics.accuracy,
            "test_auc": result.test_metrics.auc,
            "test_sensitivity": result.test_metrics.sensitivity,
            "test_specificity": result.test_metrics.specificity,
            "ece_before": result.ece_before,
            "ece_after": result.ece_after,
        })
        for path in plot_paths.values():
            mlflow.log_artifact(str(path))
    print("logged evaluation run to MLflow")


if __name__ == "__main__":
    main()
