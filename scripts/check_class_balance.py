#!/usr/bin/env python3
"""Report tumor / no-tumor class balance for each downloaded PCam split."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from pathml.data.dataset import SPLIT_FILES


def report(data_dir: Path, split: str) -> None:
    _, y_name = SPLIT_FILES[split]
    y_path = data_dir / y_name
    if not y_path.exists():
        print(f"[skip] {split}: {y_name} not found in {data_dir}")
        return

    with h5py.File(y_path, "r") as f:
        labels = np.array(f["y"]).reshape(-1)

    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    print(
        f"{split:>5}: {n:>7,} patches  |  "
        f"tumor: {n_pos:>7,} ({n_pos / n:.1%})  |  "
        f"no tumor: {n_neg:>7,} ({n_neg / n:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for split in ("train", "valid", "test"):
        report(data_dir, split)


if __name__ == "__main__":
    main()
