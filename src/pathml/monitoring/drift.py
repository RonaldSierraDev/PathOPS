"""Data-drift detection: compares live traffic against the training distribution.

Evidently's drift tooling operates on tabular feature columns, not raw
images, so each image is first reduced to a small numeric feature vector
(per-channel mean/std + overall brightness) -- simple statistics, but real
histopathology stain/exposure shifts and obviously-out-of-domain inputs
(e.g. non-pathology photos) both show up clearly in them, which is what the
Week 5 exit criteria demos.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from evidently import Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset
from PIL import Image

FEATURE_COLUMNS = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "brightness"]


def extract_features(image: Image.Image) -> dict:
    """Reduce one image to the numeric feature row drift is computed over."""
    array = np.array(image.convert("RGB").resize((96, 96))).astype(np.float64) / 255.0
    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    return {
        "mean_r": means[0], "mean_g": means[1], "mean_b": means[2],
        "std_r": stds[0], "std_g": stds[1], "std_b": stds[2],
        "brightness": float(array.mean()),
    }


@dataclass
class DriftResult:
    share: float  # fraction of feature columns flagged as drifted, in [0, 1]
    snapshot: Snapshot


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    """Run Evidently's DataDriftPreset over reference (training) vs current (live) features."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current_df[FEATURE_COLUMNS], reference_data=reference_df[FEATURE_COLUMNS])
    share = snapshot.dict()["metrics"][0]["value"]["share"]
    return DriftResult(share=share, snapshot=snapshot)
