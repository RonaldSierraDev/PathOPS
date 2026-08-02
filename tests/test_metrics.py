import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pathml.training.metrics import (
    evaluate,
    logits_and_labels,
    metrics_at_threshold,
    precision_recall_curve,
    roc_auc,
    scores_and_labels,
    select_threshold,
)


def test_roc_auc_perfect_separation():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert roc_auc(scores, labels) == 1.0


def test_roc_auc_inverted_is_zero():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([0, 0, 1, 1])
    assert roc_auc(scores, labels) == 0.0


def test_roc_auc_chance_level():
    scores = torch.tensor([0.1, 0.9, 0.1, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    assert roc_auc(scores, labels) == 0.5


def test_metrics_at_threshold_confusion_counts():
    scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    labels = torch.tensor([0, 1, 0, 1])

    m = metrics_at_threshold(scores, labels, threshold=0.5)

    assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
    assert m.accuracy == 0.5
    assert m.sensitivity == 0.5
    assert m.specificity == 0.5


def test_metrics_at_threshold_all_positive_gives_full_sensitivity():
    scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    labels = torch.tensor([0, 1, 0, 1])

    m = metrics_at_threshold(scores, labels, threshold=0.0)

    assert m.sensitivity == 1.0
    assert m.specificity == 0.0


class _ConstantLogitModel(torch.nn.Module):
    """Always predicts a fixed tumor probability, so evaluate() is checkable end-to-end."""

    def __init__(self, tumor_logit: float):
        super().__init__()
        self.tumor_logit = tumor_logit

    def forward(self, x):
        n = x.shape[0]
        return torch.tensor([[0.0, self.tumor_logit]] * n)


def test_evaluate_end_to_end():
    images = torch.zeros(4, 3, 96, 96)
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    model = _ConstantLogitModel(tumor_logit=5.0)  # softmax -> tumor prob ~1 for everyone
    result = evaluate(model, loader, device=torch.device("cpu"), threshold=0.5)

    assert result.sensitivity == 1.0
    assert result.specificity == 0.0


def test_logits_and_labels_returns_raw_logits():
    images = torch.zeros(4, 3, 96, 96)
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    model = _ConstantLogitModel(tumor_logit=5.0)
    logits, out_labels = logits_and_labels(model, loader, device=torch.device("cpu"))

    assert logits.shape == (4, 2)
    assert torch.equal(logits[:, 1], torch.full((4,), 5.0))
    assert torch.equal(out_labels, labels)


def test_scores_and_logits_agree_after_softmax():
    images = torch.zeros(4, 3, 96, 96)
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    model = _ConstantLogitModel(tumor_logit=1.5)
    scores, _ = scores_and_labels(model, loader, device=torch.device("cpu"))
    logits, _ = logits_and_labels(model, loader, device=torch.device("cpu"))

    assert torch.allclose(scores, torch.softmax(logits, dim=1)[:, 1])


def test_precision_recall_curve_perfect_separation():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])

    thresholds, precision, recall = precision_recall_curve(scores, labels)

    # Sorted descending: 0.9(pos) 0.8(pos) 0.2(neg) 0.1(neg) -- recall hits 1.0
    # as soon as both positives are included, at the second threshold.
    assert torch.equal(thresholds, torch.tensor([0.9, 0.8, 0.2, 0.1]))
    assert torch.allclose(precision, torch.tensor([1.0, 1.0, 2 / 3, 0.5], dtype=torch.float64))
    assert torch.allclose(recall, torch.tensor([0.5, 1.0, 1.0, 1.0], dtype=torch.float64))


def test_precision_recall_curve_collapses_ties():
    # Two positives and one negative all tied at 0.5: one curve point, not three.
    scores = torch.tensor([0.5, 0.5, 0.5, 0.1])
    labels = torch.tensor([1, 1, 0, 0])

    thresholds, precision, recall = precision_recall_curve(scores, labels)

    assert len(thresholds) == 2  # tie group at 0.5, then the point at 0.1
    assert precision[0] == pytest.approx(2 / 3)
    assert recall[0] == 1.0


def test_precision_recall_curve_no_positives_returns_empty():
    scores = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([0, 0, 0])

    thresholds, precision, recall = precision_recall_curve(scores, labels)

    assert len(thresholds) == len(precision) == len(recall) == 0


def test_select_threshold_meets_sensitivity_floor():
    scores = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = torch.tensor([0, 0, 1, 1, 1])  # positives at 0.5, 0.7, 0.9

    # Requiring perfect sensitivity forces the threshold down to the lowest positive score.
    threshold = select_threshold(scores, labels, min_sensitivity=1.0)
    assert threshold == 0.5
    assert metrics_at_threshold(scores, labels, threshold).sensitivity == 1.0


def test_select_threshold_prefers_higher_threshold_when_floor_allows():
    scores = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = torch.tensor([0, 0, 1, 1, 1])

    # 2/3 positives still clears a 0.6 floor, and 0.7 is the highest threshold that does.
    threshold = select_threshold(scores, labels, min_sensitivity=0.6)
    assert threshold == pytest.approx(0.7)
    assert metrics_at_threshold(scores, labels, threshold).sensitivity >= 0.6
