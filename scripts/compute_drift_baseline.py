#!/usr/bin/env python3
"""Compute the reference feature distribution the drift monitor compares live traffic against.

Samples patches from the PCam *train* split (what the model was actually
fit on), reduces each to the same feature vector pathml.monitoring.drift
uses for live predictions, and writes the result as a CSV -- optionally
uploaded to S3 for the drift-monitor Lambda to read.
"""
import argparse
import random
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from PIL import Image

from pathml.data.dataset import PCamDataset
from pathml.monitoring.drift import extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--out", default="reports/drift_baseline.csv")
    parser.add_argument("--s3-uri", default=None, help="e.g. s3://bucket/monitoring/baseline.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = PCamDataset(Path(args.data_dir), "train")
    indices = random.sample(range(len(dataset)), min(args.sample_size, len(dataset)))

    rows = []
    for idx in indices:
        tensor, _label = dataset[idx]
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rows.append(extract_features(Image.fromarray(array)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {len(rows)}-row baseline to {out_path}")

    if args.s3_uri:
        bucket, _, key = args.s3_uri.removeprefix("s3://").partition("/")
        boto3.client("s3").upload_file(str(out_path), bucket, key)
        print(f"uploaded to {args.s3_uri}")


if __name__ == "__main__":
    main()
