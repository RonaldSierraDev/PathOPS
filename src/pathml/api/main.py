"""FastAPI inference service for the PCam classifier."""
import hashlib
import io
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
import psycopg2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from pathml.db.schema import (
    LABELS,
    init_schema,
    prediction_image_key,
    record_model_version,
)
from pathml.models.classifier import build_classifier

app = FastAPI(title="PathML Inference API")

# Same env-driven pattern as ALLOWED_ORIGINS elsewhere: defaults cover local Vite
# dev (localhost and 127.0.0.1 are distinct origins to a browser); set the env var
# to the deployed frontend's origin in production.
allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = Path("models/pcam_resnet18.pt")
MODEL_NAME = "resnet18"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # generous headroom over a 96x96 patch; guards against unbounded reads

# All optional so local dev/tests need no S3 or Postgres: unset MODEL_S3_URI keeps
# the checkpoint local, unset DATABASE_URL skips prediction logging entirely, and
# unset PREDICTION_IMAGES_S3_BUCKET skips storing the uploaded image (in which case
# feedback corrections are still recorded but retraining can't recover the image).
MODEL_S3_URI = os.environ.get("MODEL_S3_URI")
MODEL_VERSION = int(os.environ.get("MODEL_VERSION", "1"))
DATABASE_URL = os.environ.get("DATABASE_URL")
PREDICTION_IMAGES_S3_BUCKET = os.environ.get("PREDICTION_IMAGES_S3_BUCKET")

# Read-only view of what the drift-monitor Lambda already produces (see
# pathml.monitoring.lambda_handler): the DriftShare metric in CloudWatch and
# the Evidently HTML reports in S3. Unset bucket disables the endpoint rather
# than failing per-request, matching DATABASE_URL's optionality above.
S3_ARTIFACTS_BUCKET = os.environ.get("S3_ARTIFACTS_BUCKET")
CLOUDWATCH_NAMESPACE = os.environ.get("CLOUDWATCH_NAMESPACE", "PathML/Monitoring")
# Mirrors terraform's drift_share_threshold so the console can draw the same
# line the CloudWatch alarm actually fires on.
DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.5"))
DRIFT_REPORTS_PREFIX = "monitoring/reports/"
DRIFT_REPORT_URL_TTL = 3600

# Built into the image by docker/Dockerfile's first stage, relative to its
# WORKDIR. Absent in a plain source checkout, where the console runs from the
# Vite dev server instead -- so serving it is conditional (see the mount at
# the bottom of this module).
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", "frontend/dist"))

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
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=True))
        _model = model.to(_device).eval()
    return _model


def _get_model_version_id() -> int:
    global _model_version_id
    if _model_version_id is None:
        init_schema(DATABASE_URL)
        record_model_version(DATABASE_URL, MODEL_NAME, MODEL_VERSION, "production", os.environ.get("MLFLOW_RUN_ID", ""))
        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM model_versions WHERE model_name = %s AND version = %s",
                    (MODEL_NAME, MODEL_VERSION),
                )
                _model_version_id = cur.fetchone()[0]
        finally:
            conn.close()
    return _model_version_id


def _log_prediction(input_bytes: bytes, label: str, confidence: float, latency_ms: float) -> int:
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (model_version_id, input_hash, predicted_label, confidence, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (_get_model_version_id(), input_hash, label, confidence, latency_ms),
            )
            prediction_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()

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
    raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file must be at most {MAX_UPLOAD_BYTES} bytes")

    start = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="file is not a valid image")
    tensor = _preprocess(image).to(_device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    label_idx = int(probs.argmax())
    label, confidence = LABELS[label_idx], float(probs[label_idx])

    response = {"label": label, "confidence": confidence}
    if DATABASE_URL:
        response["prediction_id"] = _log_prediction(raw_bytes, label, confidence, latency_ms)

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

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (prediction_id, corrected_label, note) VALUES (%s, %s, %s) RETURNING id",
                (feedback.prediction_id, feedback.corrected_label, feedback.note),
            )
            feedback_id = cur.fetchone()[0]
        conn.commit()
    except psycopg2.errors.ForeignKeyViolation:
        raise HTTPException(status_code=404, detail=f"no prediction with id {feedback.prediction_id}")
    finally:
        conn.close()

    return {"feedback_id": feedback_id}


@app.get("/feedback/export")
def export_feedback() -> list[dict]:
    """Corrections available for retraining, as (input_hash, corrected_label) pairs.

    Exists so retraining can pull feedback over plain HTTP from wherever it
    actually runs (e.g. a self-hosted runner on a contributor's own machine)
    without needing a network path into RDS, which is deliberately not
    reachable from outside the VPC. Not authenticated, same as /predict and
    /feedback -- it reveals labels only, not the images themselves.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="feedback export requires DATABASE_URL to be configured")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT p.input_hash, f.corrected_label FROM feedback f JOIN predictions p ON p.id = f.prediction_id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [{"input_hash": input_hash, "corrected_label": corrected_label} for input_hash, corrected_label in rows]


def _drift_history(days: int) -> list[dict]:
    """DriftShare datapoints from CloudWatch, oldest first.

    Period is an hour and the statistic is Maximum, but the drift Lambda runs
    on a multi-hour schedule (default rate(6 hours)), so in practice each
    bucket holds at most one check -- the aggregation never actually collapses
    two distinct checks into one point.
    """
    end = datetime.now(timezone.utc)
    stats = boto3.client("cloudwatch").get_metric_statistics(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricName="DriftShare",
        StartTime=end - timedelta(days=days),
        EndTime=end,
        Period=3600,
        Statistics=["Maximum"],
    )
    points = sorted(stats.get("Datapoints", []), key=lambda d: d["Timestamp"])
    return [{"timestamp": p["Timestamp"].isoformat(), "share": p["Maximum"]} for p in points]


def _drift_reports(limit: int) -> list[dict]:
    """The most recent Evidently reports, newest first, each with a temporary download URL.

    S3 lists keys in ascending lexicographic order with no reverse option, and
    the Lambda's keys are timestamp-named (so lexicographic == chronological);
    getting the newest therefore means paging to the end. Only the trailing
    `limit` keys are kept in memory rather than the whole listing.
    """
    s3 = boto3.client("s3")
    tail: list[dict] = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=S3_ARTIFACTS_BUCKET, Prefix=DRIFT_REPORTS_PREFIX
    ):
        tail = (tail + page.get("Contents", []))[-limit:]

    return [
        {
            "key": obj["Key"],
            "created_at": obj["LastModified"].isoformat(),
            "url": s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_ARTIFACTS_BUCKET, "Key": obj["Key"]},
                ExpiresIn=DRIFT_REPORT_URL_TTL,
            ),
        }
        for obj in reversed(tail)
    ]


@app.get("/monitoring/drift")
def drift_status(days: int = 7, reports: int = 10) -> dict:
    """Drift history and recent Evidently reports, for the console's Drift view.

    Read-only over what the scheduled Lambda already wrote. The presigned
    report URLs expire in an hour; the reports themselves contain aggregate
    feature distributions (channel means/stds, brightness), not patch images,
    so this is the same exposure level as the other unauthenticated endpoints.
    """
    if not S3_ARTIFACTS_BUCKET:
        raise HTTPException(status_code=503, detail="requires S3_ARTIFACTS_BUCKET to be configured")

    days = max(1, min(days, 30))
    reports = max(1, min(reports, 50))
    history = _drift_history(days)

    return {
        "threshold": DRIFT_SHARE_THRESHOLD,
        "current_share": history[-1]["share"] if history else None,
        "checked_at": history[-1]["timestamp"] if history else None,
        "history": history,
        "reports": _drift_reports(reports),
    }


@app.get("/models")
def list_models() -> list[dict]:
    """Model registry versions with per-version prediction counts, for the console's Registry view.

    `serving` marks the version this API instance is configured to serve
    (MODEL_NAME/MODEL_VERSION), which is recorded in model_versions at startup.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="requires DATABASE_URL to be configured")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT mv.model_name, mv.version, mv.alias, mv.mlflow_run_id, mv.created_at, "
                "COUNT(p.id) AS prediction_count "
                "FROM model_versions mv LEFT JOIN predictions p ON p.model_version_id = mv.id "
                "GROUP BY mv.id ORDER BY mv.model_name, mv.version DESC"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "model_name": model_name,
            "version": version,
            "alias": alias,
            "mlflow_run_id": mlflow_run_id,
            "created_at": created_at.isoformat(),
            "prediction_count": prediction_count,
            "serving": model_name == MODEL_NAME and version == MODEL_VERSION,
        }
        for model_name, version, alias, mlflow_run_id, created_at, prediction_count in rows
    ]


@app.get("/predictions/recent")
def recent_predictions(limit: int = 200) -> list[dict]:
    """The most recent predictions, for the drift-monitor Lambda to sample live traffic.

    Same reasoning as /feedback/export: the Lambda runs outside the VPC (to
    avoid a NAT gateway/VPC endpoints just to reach CloudWatch/SNS), so it
    reads over plain HTTP instead of connecting to RDS directly. Reveals
    labels/confidence/hash only, not the images themselves -- same exposure
    level as the other unauthenticated endpoints.
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="requires DATABASE_URL to be configured")

    limit = max(1, min(limit, 1000))
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT input_hash, predicted_label, confidence, latency_ms, created_at FROM predictions "
                "ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "input_hash": input_hash,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "created_at": created_at.isoformat(),
        }
        for input_hash, predicted_label, confidence, latency_ms, created_at in rows
    ]


# Mounted last on purpose: Starlette matches routes in registration order, so
# every API route above wins over this catch-all. Serving the console from the
# same origin as the API it calls means no CORS in production and one public
# URL (and so one TLS certificate) for the whole service.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="console")
