"""Evaluation metrics for the PCam binary classifier.

Accuracy alone is a poor signal for a screening task like this one: a false
negative (a missed tumor) is far costlier than a false positive, so AUC plus
threshold-dependent sensitivity/specificity matter more than raw accuracy.
"""
import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class Metrics:
    accuracy: float
    auc: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def sensitivity(self) -> float:
        """Recall on the tumor class: TP / (TP + FN). The false-negative rate matters most here."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def specificity(self) -> float:
        """True negative rate: TN / (TN + FP)."""
        denom = self.tn + self.fp
        return self.tn / denom if denom else float("nan")

    def __str__(self) -> str:
        return (
            f"acc={self.accuracy:.4f}  auc={self.auc:.4f}  "
            f"sensitivity={self.sensitivity:.4f}  specificity={self.specificity:.4f}  "
            f"(threshold={self.threshold:.2f}, tp={self.tp} fp={self.fp} tn={self.tn} fn={self.fn})"
        )


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-indexed ranks with tied values assigned their average rank (like scipy's rankdata)."""
    sorted_vals, order = torch.sort(x)
    n = len(x)
    ranks = torch.empty(n, dtype=torch.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average of ranks i+1..j+1
        i = j + 1

    return ranks


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUC via the rank-sum (Mann-Whitney U) identity -- exact, no sklearn dependency."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return ((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def scores_and_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference over a loader once, returning tumor-class probabilities and true labels."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            all_scores.append(probs.cpu())
            all_labels.append(labels)
    return torch.cat(all_scores), torch.cat(all_labels)


def metrics_at_threshold(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> Metrics:
    preds = (scores >= threshold).long()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    return Metrics(
        accuracy=(tp + tn) / len(labels),
        auc=roc_auc(scores, labels),
        threshold=threshold,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Metrics:
    """Evaluate a model over a loader at a fixed decision threshold (default 0.5)."""
    scores, labels = scores_and_labels(model, loader, device)
    return metrics_at_threshold(scores, labels, threshold)


def select_threshold(scores: torch.Tensor, labels: torch.Tensor, min_sensitivity: float = 0.95) -> float:
    """Pick the highest decision threshold that still meets a minimum sensitivity.

    In a cancer-screening setting a false negative (missed tumor) is far more
    costly than a false positive, so the threshold is chosen to guarantee a
    sensitivity floor first, and only maximize specificity subject to that.
    Raising the threshold monotonically trades sensitivity for specificity, so
    the highest threshold still meeting the floor is also the most specific
    one that meets it.

    Sensitivity depends only on the positive-class scores: guaranteeing at
    least `k = ceil(min_sensitivity * n_pos)` positives score at or above the
    threshold means the threshold is exactly the k-th highest positive score
    -- no need to scan every candidate against the full label set (O(n) per
    candidate), which is what made the naive sweep over ~30K thresholds hang.
    """
    pos_scores = scores[labels == 1]
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0

    k = min(max(math.ceil(min_sensitivity * n_pos), 1), n_pos)
    sorted_desc, _ = torch.sort(pos_scores, descending=True)
    return sorted_desc[k - 1].item()
