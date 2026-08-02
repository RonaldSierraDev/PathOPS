from pathlib import Path

import h5py
import numpy as np
from matplotlib.figure import Figure

from pathml.data.dataset import SPLIT_FILES
from pathml.models.classifier import build_classifier
from pathml.training.evaluation import EvaluationResult, evaluate_checkpoint


def _write_split(tmp_path: Path, split: str, n: int) -> None:
    x_name, y_name = SPLIT_FILES[split]
    images = np.random.randint(0, 256, size=(n, 96, 96, 3), dtype=np.uint8)
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8).reshape(n, 1, 1, 1)

    with h5py.File(tmp_path / x_name, "w") as f:
        f.create_dataset("x", data=images)
    with h5py.File(tmp_path / y_name, "w") as f:
        f.create_dataset("y", data=labels)


def test_evaluate_checkpoint_end_to_end(tmp_path):
    _write_split(tmp_path, "valid", n=16)
    _write_split(tmp_path, "test", n=16)

    model = build_classifier("resnet18", pretrained=False)
    result = evaluate_checkpoint(model, tmp_path, batch_size=8, min_sensitivity=0.8)

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.valid_metrics.sensitivity <= 1.0
    assert 0.0 <= result.test_metrics.sensitivity <= 1.0
    assert result.temperature > 0
    assert 0.0 <= result.ece_before <= 1.0
    assert 0.0 <= result.ece_after <= 1.0

    assert set(result.figures) == {"confusion_matrix", "pr_curve", "reliability_before", "reliability_after"}
    for fig in result.figures.values():
        assert isinstance(fig, Figure)
