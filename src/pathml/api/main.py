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
from pydantic import BaseModel

from pathml.db.schema import (
    LABELS,
    init_schema,
    prediction_image_key,
    record_model_version,
)
from pathml.models.classifier import build_classifier

app = FastAPI(title="PathML Inference API")

CHECKPOINT_PATH = Path("models/pcam_resnet18.pt")
MODEL_NAME = "resnet18"

# All optional so local dev/tests need no S3 or Postgres: unset MODEL_S3_URI keeps
# the checkpoint local, unset DATABASE_URL skips prediction logging entirely, and
# unset PREDICTION_IMAGES_S3_BUCKET skips storing the uploaded image (in which case
# feedback corrections are still recorded but retraining can't recover the image).
MODEL_S3_URI = os.environ.get("MODEL_S3_URI")
MODEL_VERSION = int(os.environ.get("MODEL_VERSION", "1"))
DATABASE_URL = os.environ.get("DATABASE_URL")
PREDICTION_IMAGES_S3_BUCKET = os.environ.get("PREDICTION_IMAGES_S3_BUCKET")

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


def _log_prediction(input_bytes: bytes, label: str, confidence: float) -> int:
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (model_version_id, input_hash, predicted_label, confidence) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (_get_model_version_id(), input_hash, label, confidence),
        )
        prediction_id = cur.fetchone()[0]
        conn.commit()

    # Predictions are only content-addressed by hash in Postgres -- without also
    # keeping the actual bytes somewhere, a later correction has no image for
    # retraining to learn from. Best-effort: a duplicate upload for a repeated
    # hash just overwrites the same key, which is fine.
    if PREDICTION_IMAGES_S3_BUCKET:
        boto3.client("s3").put_object(
            Bucket=PREDICTION_IMAGES_S3_BUCKET, Key=prediction_image_key(input_hash), Body=input_bytes,
        )

    return prediction_id


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

    response = {"label": label, "confidence": confidence}
    if DATABASE_URL:
        response["prediction_id"] = _log_prediction(raw_bytes, label, confidence)

    return response


class FeedbackRequest(BaseModel):
    prediction_id: int
    corrected_label: str
    note: str | None = None


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest) -> dict:
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="feedback requires DATABASE_URL to be configured")
    if feedback.corrected_label not in LABELS:
        raise HTTPException(status_code=400, detail=f"corrected_label must be one of {LABELS}")

    try:
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (prediction_id, corrected_label, note) VALUES (%s, %s, %s) RETURNING id",
                (feedback.prediction_id, feedback.corrected_label, feedback.note),
            )
            feedback_id = cur.fetchone()[0]
            conn.commit()
    except psycopg2.errors.ForeignKeyViolation:
        raise HTTPException(status_code=404, detail=f"no prediction with id {feedback.prediction_id}")

    return {"feedback_id": feedback_id}


@app.get("/feedback/export")
def export_feedback() -> list[dict]:
    """Corrections available for retraining, as (input_hash, corrected_label) pairs.

    Exists so retraining can pull feedback over plain HTTPS from wherever it
    actually runs (e.g. a self-hosted runner on a contributor's own machine)
    without needing a network path into RDS, which is deliberately not
    reachable from outside the VPC. Not authenticated, same as /predict and
    /feedback -- it reveals labels only, not the images themselves.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="feedback export requires DATABASE_URL to be configured")

    with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT p.input_hash, f.corrected_label FROM feedback f JOIN predictions p ON p.id = f.prediction_id"
        )
        rows = cur.fetchall()

    return [{"input_hash": input_hash, "corrected_label": corrected_label} for input_hash, corrected_label in rows]


@app.get("/predictions/recent")
def recent_predictions(limit: int = 200) -> list[dict]:
    """The most recent predictions, for the drift-monitor Lambda to sample live traffic.

    Same reasoning as /feedback/export: the Lambda runs outside the VPC (to
    avoid a NAT gateway/VPC endpoints just to reach CloudWatch/SNS), so it
    reads over plain HTTPS instead of connecting to RDS directly. Reveals
    labels/confidence/hash only, not the images themselves -- same exposure
    level as the other unauthenticated endpoints.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="requires DATABASE_URL to be configured")

    limit = min(limit, 1000)
    with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT input_hash, predicted_label, confidence, created_at FROM predictions "
            "ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()

    return [
        {
            "input_hash": input_hash,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "created_at": created_at.isoformat(),
        }
        for input_hash, predicted_label, confidence, created_at in rows
    ]
