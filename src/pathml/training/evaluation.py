"""Reusable evaluation core: threshold selection, calibration, and plots for a trained model.

Shared by scripts/evaluate.py (a local checkpoint file) and
scripts/promote_model.py (a model loaded from the MLflow registry) so both
run the exact same evaluation logic regardless of where the model came from.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from pathml.data.dataset import PCamDataset
from pathml.training.calibration import expected_calibration_error, fit_temperature, reliability_bins
from pathml.training.metrics import (
    Metrics,
    logits_and_labels,
    metrics_at_threshold,
    precision_recall_curve,
    select_threshold,
)
from pathml.training.plots import plot_confusion_matrix, plot_pr_curve, plot_reliability_diagram


@dataclass
class EvaluationResult:
    valid_metrics: Metrics
    test_metrics: Metrics
    threshold: float
    temperature: float
    ece_before: float
    ece_after: float
    figures: dict[str, Figure]


def evaluate_checkpoint(
    model: torch.nn.Module,
    data_dir: Union[str, Path],
    batch_size: int = 64,
    min_sensitivity: float = 0.95,
    device: Optional[torch.device] = None,
) -> EvaluationResult:
    """Run the full evaluation suite against an already-loaded model.

    Threshold selection happens on `valid` (never `test`) to guarantee a
    minimum sensitivity; temperature is also fit on `valid`. All final
    metrics, calibration, and plots are reported on the held-out `test`
    split. Temperature scaling is a monotonic transform of the logits, so it
    changes reported confidence (and ECE) but never the AUC, PR curve, or the
    threshold-based decision -- see pathml.training.calibration.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    valid_loader = DataLoader(PCamDataset(data_dir, "valid"), batch_size=batch_size)
    valid_logits, valid_labels = logits_and_labels(model, valid_loader, device)
    valid_scores = torch.softmax(valid_logits, dim=1)[:, 1]

    threshold = select_threshold(valid_scores, valid_labels, min_sensitivity=min_sensitivity)
    valid_metrics = metrics_at_threshold(valid_scores, valid_labels, threshold)
    temperature = fit_temperature(valid_logits, valid_labels)

    test_loader = DataLoader(PCamDataset(data_dir, "test"), batch_size=batch_size)
    test_logits, test_labels = logits_and_labels(model, test_loader, device)
    test_scores = torch.softmax(test_logits, dim=1)[:, 1]
    test_calibrated_scores = torch.softmax(test_logits / temperature, dim=1)[:, 1]

    test_metrics = metrics_at_threshold(test_scores, test_labels, threshold)
    ece_before = expected_calibration_error(test_scores, test_labels)
    ece_after = expected_calibration_error(test_calibrated_scores, test_labels)

    _, precision, recall = precision_recall_curve(test_scores, test_labels)

    figures = {
        "confusion_matrix": plot_confusion_matrix(test_metrics),
        "pr_curve": plot_pr_curve(precision, recall),
        "reliability_before": plot_reliability_diagram(
            reliability_bins(test_scores, test_labels), title="Reliability (uncalibrated)"
        ),
        "reliability_after": plot_reliability_diagram(
            reliability_bins(test_calibrated_scores, test_labels),
            title=f"Reliability (T={temperature:.2f})",
        ),
    }

    return EvaluationResult(
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        threshold=threshold,
        temperature=temperature,
        ece_before=ece_before,
        ece_after=ece_after,
        figures=figures,
    )
