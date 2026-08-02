"""Pulls pathologist corrections out of the API so retraining can learn from them.

This is the design doc's Week 4 "close the loop" requirement: retraining
should draw on the `feedback` table, not just the static PCam split. Reads
happen over the API's /feedback/export endpoint rather than a direct Postgres
connection -- retraining commonly runs somewhere (e.g. a self-hosted CI
runner on a contributor's own machine) with no network path into RDS, which
is deliberately private, but does have a path to the already-public
inference API. Image bytes come from wherever the API stashed them in S3
(see PREDICTION_IMAGES_S3_BUCKET / prediction_image_key in pathml.db.schema)
-- without that, a correction has a label but nothing to train on.
"""
import io

import boto3
import httpx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pathml.db.schema import LABELS, prediction_image_key


class FeedbackDataset(Dataset):
    """Corrected (image, label) pairs pulled from /feedback/export + S3.

    Loads everything eagerly at construction time -- the feedback set is
    expected to be small relative to the static PCam split it gets
    concatenated with, so there's no need for the lazy-open trick
    PCamDataset uses for its much larger HDF5 files.
    """

    def __init__(self, feedback_export_url: str, images_s3_bucket: str):
        response = httpx.get(feedback_export_url, timeout=30)
        response.raise_for_status()
        corrections = response.json()

        s3 = boto3.client("s3")
        self._examples = []
        for correction in corrections:
            obj = s3.get_object(
                Bucket=images_s3_bucket, Key=prediction_image_key(correction["input_hash"]),
            )
            image = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB").resize((96, 96))
            tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1).copy()).float() / 255.0
            self._examples.append((tensor, LABELS.index(correction["corrected_label"])))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int):
        image, label = self._examples[idx]
        return image, torch.tensor(label, dtype=torch.long)
