"""Evaluation plots for the PCam classifier: confusion matrix, PR curve, reliability diagram.

Each function is a pure transform from data to a matplotlib Figure -- no file
I/O -- so callers decide whether to save, show, or log the figure (e.g. as an
MLflow artifact). Uses the non-interactive Agg backend since this runs in
training/evaluation scripts, not a GUI.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from pathml.training.calibration import ReliabilityBins
from pathml.training.metrics import Metrics

LABELS = ("no_tumor", "tumor")


def plot_confusion_matrix(metrics: Metrics) -> Figure:
    """2x2 confusion matrix heatmap at the metrics object's decision threshold."""
    matrix = [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=LABELS)
    ax.set_yticks([0, 1], labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={metrics.threshold:.3f})")

    max_count = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i][j] > max_count / 2 else "black"
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color=color)

    fig.tight_layout()
    return fig


def plot_pr_curve(precision: torch.Tensor, recall: torch.Tensor) -> Figure:
    """Precision-recall curve from pathml.training.metrics.precision_recall_curve output."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall.numpy(), precision.numpy())
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    return fig


def plot_reliability_diagram(bins: ReliabilityBins, title: str = "Reliability Diagram") -> Figure:
    """Per-bin accuracy vs. confidence, with the diagonal marking perfect calibration."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", label="perfectly calibrated")

    has_data = bins.count > 0
    ax.bar(
        bins.confidence[has_data].numpy(),
        bins.accuracy[has_data].numpy(),
        width=0.05,
        edgecolor="black",
        alpha=0.7,
        label="model",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
