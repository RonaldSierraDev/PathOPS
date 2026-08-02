"""Post-hoc probability calibration for the PCam classifier.

Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural
Networks"): a single scalar T > 0 rescales logits before softmax so predicted
confidence matches empirical accuracy. Dividing logits by T is a monotonic
transform of the score, so it never changes which class wins or how examples
rank against each other -- AUC, the PR curve, and any threshold-based
decision are identical before and after. It only changes the reported
confidence number, which is exactly what calibration is for.
"""
from dataclasses import dataclass

import torch


@dataclass
class ReliabilityBins:
    """Per-bin data for a reliability diagram, one entry per confidence bucket."""

    confidence: torch.Tensor  # mean predicted confidence in each bin
    accuracy: torch.Tensor    # empirical accuracy in each bin
    count: torch.Tensor       # number of examples in each bin


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """Fit a single scalar temperature T minimizing NLL of (logits / T) vs labels.

    T is parametrized as exp(log_T) so LBFGS can't drive it negative or
    through zero.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_t).item()


def reliability_bins(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> ReliabilityBins:
    """Bucket predictions into confidence bins and compute per-bin accuracy.

    `scores` is the model's predicted probability of the positive (tumor)
    class. Confidence is max(score, 1-score) -- the model's certainty in
    whichever class it actually predicts -- and an example is "correct" if
    that predicted class matches the true label. Bins are equal-width over
    [0.5, 1.0], since confidence is always >= 0.5 by construction.
    """
    confidence = torch.maximum(scores, 1 - scores)
    predicted = (scores >= 0.5).long()
    correct = (predicted == labels).float()

    bin_edges = torch.linspace(0.5, 1.0, n_bins + 1)
    bin_idx = torch.bucketize(confidence, bin_edges[1:-1])

    bin_confidence = torch.zeros(n_bins)
    bin_accuracy = torch.zeros(n_bins)
    bin_count = torch.zeros(n_bins, dtype=torch.long)

    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_count[b] = count
        if count:
            bin_confidence[b] = confidence[mask].mean()
            bin_accuracy[b] = correct[mask].mean()

    return ReliabilityBins(confidence=bin_confidence, accuracy=bin_accuracy, count=bin_count)


def expected_calibration_error(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Weighted mean absolute gap between confidence and accuracy across bins (Guo et al.)."""
    bins = reliability_bins(scores, labels, n_bins=n_bins)
    total = int(bins.count.sum())
    if total == 0:
        return float("nan")
    weights = bins.count.float() / total
    gap = torch.abs(bins.confidence - bins.accuracy)
    return float((weights * gap).sum())
