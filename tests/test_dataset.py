from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pathml.data.dataset import SPLIT_FILES, PCamDataset


def _write_split(tmp_path: Path, split: str, n: int = 8) -> Path:
    x_name, y_name = SPLIT_FILES[split]
    images = np.random.randint(0, 256, size=(n, 96, 96, 3), dtype=np.uint8)
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8).reshape(n, 1, 1, 1)

    with h5py.File(tmp_path / x_name, "w") as f:
        f.create_dataset("x", data=images)
    with h5py.File(tmp_path / y_name, "w") as f:
        f.create_dataset("y", data=labels)
    return tmp_path


def test_length_and_item_shape(tmp_path):
    _write_split(tmp_path, "train", n=8)
    ds = PCamDataset(tmp_path, "train")

    assert len(ds) == 8
    image, label = ds[0]
    assert image.shape == (3, 96, 96)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert label.item() in (0, 1)


def test_labels_match_source(tmp_path):
    _write_split(tmp_path, "valid", n=6)
    ds = PCamDataset(tmp_path, "valid")

    labels = [ds[i][1].item() for i in range(len(ds))]
    assert labels == [i % 2 for i in range(6)]


def test_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PCamDataset(tmp_path, "train")


def test_invalid_split_raises(tmp_path):
    with pytest.raises(ValueError):
        PCamDataset(tmp_path, "bogus")
