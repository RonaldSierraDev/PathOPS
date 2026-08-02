import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pathml.monitoring.drift import FEATURE_COLUMNS, compute_drift, extract_features


def _solid_image(color) -> Image.Image:
    return Image.new("RGB", (96, 96), color=color)


def _noisy_image(rng: np.random.Generator, base_color, noise_std=15) -> Image.Image:
    """A more realistic stand-in for a tissue patch than a flat color -- solid
    images give every std_* feature exactly zero variance across a whole
    dataset, which is a degenerate case Evidently's stat tests handle poorly
    (divide-by-zero warnings), not something PCam patches ever look like."""
    noise = rng.normal(0, noise_std, size=(96, 96, 3))
    array = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_extract_features_returns_expected_columns_and_ranges():
    features = extract_features(_solid_image((128, 64, 200)))

    assert set(features) == set(FEATURE_COLUMNS)
    assert 0.0 <= features["mean_r"] <= 1.0
    assert features["std_r"] == pytest.approx(0.0, abs=1e-9)  # solid color -> ~zero variance
    assert features["mean_r"] == pytest.approx(128 / 255)


def _feature_rows(images) -> pd.DataFrame:
    return pd.DataFrame([extract_features(image) for image in images])


def test_compute_drift_reports_low_share_for_matching_distributions():
    rng = np.random.default_rng(0)
    reference = _feature_rows([_noisy_image(rng, (130, 130, 130)) for _ in range(30)])
    current = _feature_rows([_noisy_image(rng, (130, 130, 130)) for _ in range(15)])

    result = compute_drift(reference, current)

    assert result.share < 0.5


def test_compute_drift_reports_high_share_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = _feature_rows([_noisy_image(rng, (130, 130, 130)) for _ in range(30)])
    # Wildly different color/brightness profile -- stands in for "non-pathology image".
    current = _feature_rows([_noisy_image(rng, (10, 220, 10)) for _ in range(15)])

    result = compute_drift(reference, current)

    assert result.share > 0.5
