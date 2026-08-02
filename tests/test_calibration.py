import pytest
import torch

from pathml.training.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
)


def test_reliability_bins_counts_sum_to_total_and_are_bounded():
    torch.manual_seed(0)
    scores = torch.rand(50)
    labels = torch.randint(0, 2, (50,))

    bins = reliability_bins(scores, labels, n_bins=5)

    assert int(bins.count.sum()) == 50
    for c in bins.confidence[bins.count > 0]:
        assert 0.5 <= c.item() <= 1.0
    for a in bins.accuracy[bins.count > 0]:
        assert 0.0 <= a.item() <= 1.0


def test_expected_calibration_error_near_zero_for_well_calibrated_predictions():
    # Confident and always correct -> confidence and accuracy both ~1.0 in every bin.
    scores = torch.tensor([0.99, 0.98, 0.97, 0.01, 0.02, 0.03])
    labels = torch.tensor([1, 1, 1, 0, 0, 0])

    ece = expected_calibration_error(scores, labels, n_bins=5)

    assert ece < 0.05


def test_expected_calibration_error_high_for_overconfident_wrong_predictions():
    # Maximally confident but only 50% correct -> full confidence/accuracy gap.
    scores = torch.tensor([0.99, 0.99, 0.99, 0.99])
    labels = torch.tensor([1, 1, 0, 0])  # half agree with the "predicted tumor" call, half don't

    ece = expected_calibration_error(scores, labels, n_bins=5)

    assert ece == pytest.approx(0.49, abs=0.02)


def test_fit_temperature_reduces_ece_on_overconfident_logits():
    n = 100
    labels = torch.cat([torch.ones(70, dtype=torch.long), torch.zeros(30, dtype=torch.long)])
    # Always maximally confident in class 1, but only 70% correct.
    logits = torch.stack([torch.zeros(n), torch.full((n,), 10.0)], dim=1)

    scores_before = torch.softmax(logits, dim=1)[:, 1]
    ece_before = expected_calibration_error(scores_before, labels)

    temperature = fit_temperature(logits, labels)
    scores_after = torch.softmax(logits / temperature, dim=1)[:, 1]
    ece_after = expected_calibration_error(scores_after, labels)

    assert temperature > 1.0
    assert ece_after < ece_before


def test_fit_temperature_leaves_well_calibrated_logits_near_one():
    # logit=0 -> p=0.5 for every example; labels are a coin flip -> already calibrated.
    n = 200
    labels = torch.cat([torch.ones(100, dtype=torch.long), torch.zeros(100, dtype=torch.long)])
    logits = torch.zeros(n, 2)

    temperature = fit_temperature(logits, labels)

    assert temperature == pytest.approx(1.0, abs=0.1)
