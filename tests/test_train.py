from pathlib import Path

import h5py
import numpy as np

from pathml.data.dataset import SPLIT_FILES
from pathml.training.train import train


def _write_split(tmp_path: Path, split: str, n: int) -> None:
    x_name, y_name = SPLIT_FILES[split]
    images = np.random.randint(0, 256, size=(n, 96, 96, 3), dtype=np.uint8)
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8).reshape(n, 1, 1, 1)

    with h5py.File(tmp_path / x_name, "w") as f:
        f.create_dataset("x", data=images)
    with h5py.File(tmp_path / y_name, "w") as f:
        f.create_dataset("y", data=labels)


def test_train_runs_and_saves_checkpoint(tmp_path):
    _write_split(tmp_path, "train", n=8)
    _write_split(tmp_path, "valid", n=4)
    out_path = tmp_path / "checkpoint.pt"

    train(
        data_dir=tmp_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        model_name="resnet18",
        out_path=out_path,
        pretrained=False,
    )

    assert out_path.exists()
