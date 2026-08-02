"""FastAPI inference service for the PCam classifier."""
import hashlib
import io
import os
from pathlib import Path

import boto3
import numpy as np
import psycopg2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from pathml.db.schema import init_schema, record_model_version
from pathml.models.classifier import build_classifier

app = FastAPI(title="PathML Inference API")

CHECKPOINT_PATH = Path("models/pcam_resnet18.pt")
MODEL_NAME = "resnet18"
LABELS = ("no_tumor", "tumor")

# Both optional so local dev/tests need no S3 or Postgres: unset MODEL_S3_URI keeps
# the checkpoint local, unset DATABASE_URL skips prediction logging entirely.
MODEL_S3_URI = os.environ.get("MODEL_S3_URI")
MODEL_VERSION = int(os.environ.get("MODEL_VERSION", "1"))
DATABASE_URL = os.environ.get("DATABASE_URL")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_model_version_id = None


def _download_model_from_s3() -> None:
    bucket, _, key = MODEL_S3_URI.removeprefix("s3://").partition("/")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(CHECKPOINT_PATH))


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        if MODEL_S3_URI and not CHECKPOINT_PATH.exists():
            _download_model_from_s3()
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(status_code=503, detail=f"no checkpoint found at {CHECKPOINT_PATH}")
        model = build_classifier(MODEL_NAME, pretrained=False)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=_device))
        _model = model.to(_device).eval()
    return _model


def _get_model_version_id() -> int:
    global _model_version_id
    if _model_version_id is None:
        init_schema(DATABASE_URL)
        record_model_version(DATABASE_URL, MODEL_NAME, MODEL_VERSION, "production", os.environ.get("MLFLOW_RUN_ID", ""))
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM model_versions WHERE model_name = %s AND version = %s",
                (MODEL_NAME, MODEL_VERSION),
            )
            _model_version_id = cur.fetchone()[0]
    return _model_version_id


def _log_prediction(input_bytes: bytes, label: str, confidence: float) -> None:
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (model_version_id, input_hash, predicted_label, confidence) "
            "VALUES (%s, %s, %s, %s)",
            (_get_model_version_id(), input_hash, label, confidence),
        )
        conn.commit()


def _preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((96, 96))
    array = np.array(image).transpose(2, 0, 1).copy()
    tensor = torch.from_numpy(array).float() / 255.0
    return tensor.unsqueeze(0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    model = _load_model()
    raw_bytes = await file.read()
    image = Image.open(io.BytesIO(raw_bytes))
    tensor = _preprocess(image).to(_device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    label_idx = int(probs.argmax())
    label, confidence = LABELS[label_idx], float(probs[label_idx])

    if DATABASE_URL:
        _log_prediction(raw_bytes, label, confidence)

    return {"label": label, "confidence": confidence}
