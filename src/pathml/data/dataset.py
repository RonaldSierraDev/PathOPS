"""PyTorch Dataset for the PatchCamelyon (PCam) histopathology dataset."""
from collections.abc import Callable
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

SPLIT_FILES = {
    "train": ("camelyonpatch_level_2_split_train_x.h5", "camelyonpatch_level_2_split_train_y.h5"),
    "valid": ("camelyonpatch_level_2_split_valid_x.h5", "camelyonpatch_level_2_split_valid_y.h5"),
    "test": ("camelyonpatch_level_2_split_test_x.h5", "camelyonpatch_level_2_split_test_y.h5"),
}


class PCamDataset(Dataset):
    """One split (train/valid/test) of PCam, backed by the downloaded HDF5 files.

    File handles are opened lazily on first access rather than in __init__,
    since h5py file handles cannot be shared safely across the worker
    processes a DataLoader forks (num_workers > 0).
    """

    def __init__(self, data_dir: str | Path, split: str, transform: Callable | None = None):
        self._xf = None
        self._yf = None

        if split not in SPLIT_FILES:
            raise ValueError(f"split must be one of {list(SPLIT_FILES)}, got {split!r}")

        data_dir = Path(data_dir)
        x_name, y_name = SPLIT_FILES[split]
        self.x_path = data_dir / x_name
        self.y_path = data_dir / y_name
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                f"PCam {split!r} files not found in {data_dir}. Run scripts/download_pcam.py first."
            )

        self.transform = transform
        with h5py.File(self.y_path, "r") as f:
            self._length = f["y"].shape[0]

    def _ensure_open(self) -> None:
        if self._xf is None:
            self._xf = h5py.File(self.x_path, "r")
            self._yf = h5py.File(self.y_path, "r")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        self._ensure_open()
        image = self._xf["x"][idx]  # HWC uint8, 0-255
        label = int(self._yf["y"][idx].reshape(-1)[0])

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        for f in (self._xf, self._yf):
            if f is not None:
                f.close()
