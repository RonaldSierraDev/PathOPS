import hashlib
from datetime import datetime, timezone

import psycopg2
import pytest
import torch
from fastapi.testclient import TestClient

import pathml.api.main as api_main
from pathml.api.main import app
from pathml.db.schema import DEFAULT_DSN, init_schema

client = TestClient(app)


def _postgres_available() -> bool:
    try:
        with psycopg2.connect(DEFAULT_DSN, connect_timeout=2):
            return True
    except psycopg2.OperationalError:
        return False


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_checkpoint_returns_503(monkeypatch, tmp_path):
    # Point at a path that's guaranteed not to exist, rather than relying on
    # the real models/ dir being empty -- a trained checkpoint legitimately
    # lives there once training has run.
    monkeypatch.setattr(api_main, "CHECKPOINT_PATH", tmp_path / "no_such_checkpoint.pt")
    monkeypatch.setattr(api_main, "_model", None)

    files = {"file": ("patch.png", b"not a real image", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 503


def test_load_model_downloads_from_s3_when_uri_set_and_checkpoint_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "CHECKPOINT_PATH", tmp_path / "downloaded.pt")
    monkeypatch.setattr(api_main, "MODEL_S3_URI", "s3://my-bucket/models/pcam_resnet18.pt")
    monkeypatch.setattr(api_main, "_model", None)

    calls = []
    monkeypatch.setattr(
        api_main,
        "_download_model_from_s3",
        lambda: calls.append(api_main.CHECKPOINT_PATH) or _write_dummy_checkpoint(api_main.CHECKPOINT_PATH),
    )

    model = api_main._load_model()

    assert calls == [tmp_path / "downloaded.pt"]
    assert model is not None


def _write_dummy_checkpoint(path):
    from pathml.models.classifier import build_classifier

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_classifier(api_main.MODEL_NAME, pretrained=False).state_dict(), path)


def test_predict_logs_to_postgres_when_database_url_set(monkeypatch):
    monkeypatch.setattr(api_main, "DATABASE_URL", "postgresql://fake-dsn")
    monkeypatch.setattr(api_main, "_load_model", lambda: _ConstantLogitModel())

    logged = {}
    monkeypatch.setattr(
        api_main, "_log_prediction", lambda input_bytes, label, confidence, latency_ms: logged.update(
            input_bytes=input_bytes, label=label, confidence=confidence, latency_ms=latency_ms,
        ),
    )

    image_bytes = _png_bytes()
    files = {"file": ("patch.png", image_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert logged["input_bytes"] == image_bytes
    assert logged["label"] == response.json()["label"]
    assert logged["latency_ms"] >= 0


def _png_bytes() -> bytes:
    import io as _io

    from PIL import Image as _Image

    buf = _io.BytesIO()
    _Image.new("RGB", (96, 96)).save(buf, format="PNG")
    return buf.getvalue()


class _ConstantLogitModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.0, 5.0]]).expand(x.shape[0], 2)


def test_log_prediction_uploads_image_to_s3_when_bucket_set(monkeypatch):
    monkeypatch.setattr(api_main, "PREDICTION_IMAGES_S3_BUCKET", "my-bucket")

    class _FakeS3Client:
        def put_object(self, **kwargs):
            uploads.append(kwargs)

    uploads = []
    monkeypatch.setattr(api_main.boto3, "client", lambda service: _FakeS3Client())
    monkeypatch.setattr(api_main, "_get_model_version_id", lambda: 7)
    monkeypatch.setattr(api_main.psycopg2, "connect", lambda dsn: _FakeConnection())
    monkeypatch.setattr(api_main, "DATABASE_URL", "postgresql://fake-dsn")

    prediction_id = api_main._log_prediction(b"raw image bytes", "tumor", 0.9, 12.5)

    assert prediction_id == 42
    assert len(uploads) == 1
    assert uploads[0]["Bucket"] == "my-bucket"
    assert uploads[0]["Body"] == b"raw image bytes"
    assert uploads[0]["Key"] == api_main.prediction_image_key(
        hashlib.sha256(b"raw image bytes").hexdigest()
    )


class _FakeCursor:
    def execute(self, *args, **kwargs):
        pass

    def fetchone(self):
        return (42,)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConnection:
    def cursor(self):
        return _FakeCursor()

    def commit(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_feedback_without_database_url_returns_503(monkeypatch):
    monkeypatch.setattr(api_main, "DATABASE_URL", None)

    response = client.post("/feedback", json={"prediction_id": 1, "corrected_label": "tumor"})
    assert response.status_code == 503


def test_feedback_rejects_invalid_corrected_label(monkeypatch):
    monkeypatch.setattr(api_main, "DATABASE_URL", "postgresql://fake-dsn")

    response = client.post("/feedback", json={"prediction_id": 1, "corrected_label": "not_a_real_label"})
    assert response.status_code == 400


def test_export_feedback_without_database_url_returns_503(monkeypatch):
    monkeypatch.setattr(api_main, "DATABASE_URL", None)
    response = client.get("/feedback/export")
    assert response.status_code == 503


def test_recent_predictions_without_database_url_returns_503(monkeypatch):
    monkeypatch.setattr(api_main, "DATABASE_URL", None)
    response = client.get("/predictions/recent")
    assert response.status_code == 503


def test_list_models_without_database_url_returns_503(monkeypatch):
    monkeypatch.setattr(api_main, "DATABASE_URL", None)
    response = client.get("/models")
    assert response.status_code == 503


def test_drift_status_without_bucket_returns_503(monkeypatch):
    monkeypatch.setattr(api_main, "S3_ARTIFACTS_BUCKET", None)
    response = client.get("/monitoring/drift")
    assert response.status_code == 503


class _FakeCloudWatchClient:
    """Returns datapoints out of order, as CloudWatch itself does."""

    def get_metric_statistics(self, **kwargs):
        self.call = kwargs
        return {
            "Datapoints": [
                {"Timestamp": datetime(2026, 8, 2, 18, tzinfo=timezone.utc), "Maximum": 0.71},
                {"Timestamp": datetime(2026, 8, 2, 6, tzinfo=timezone.utc), "Maximum": 0.14},
            ]
        }


class _FakeDriftS3Client:
    def __init__(self):
        self.presigned = []

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        self.listed = kwargs
        # Two pages, ascending by key, mirroring S3's lexicographic ordering.
        return iter(
            [
                {"Contents": [{"Key": "monitoring/reports/20260801T000000.html",
                               "LastModified": datetime(2026, 8, 1, tzinfo=timezone.utc)}]},
                {"Contents": [{"Key": "monitoring/reports/20260802T060000.html",
                               "LastModified": datetime(2026, 8, 2, 6, tzinfo=timezone.utc)},
                              {"Key": "monitoring/reports/20260802T180000.html",
                               "LastModified": datetime(2026, 8, 2, 18, tzinfo=timezone.utc)}]},
            ]
        )

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presigned.append(Params["Key"])
        return f"https://s3.example/{Params['Key']}?expires={ExpiresIn}"


def test_drift_status_returns_latest_share_and_newest_reports_first(monkeypatch):
    monkeypatch.setattr(api_main, "S3_ARTIFACTS_BUCKET", "my-bucket")
    monkeypatch.setattr(api_main, "DRIFT_SHARE_THRESHOLD", 0.5)
    s3 = _FakeDriftS3Client()
    monkeypatch.setattr(
        api_main.boto3, "client",
        lambda service: s3 if service == "s3" else _FakeCloudWatchClient(),
    )

    body = client.get("/monitoring/drift").json()

    assert body["threshold"] == 0.5
    # Latest by timestamp, not by CloudWatch's arbitrary datapoint order.
    assert body["current_share"] == 0.71
    assert [point["share"] for point in body["history"]] == [0.14, 0.71]
    assert [report["key"] for report in body["reports"]] == [
        "monitoring/reports/20260802T180000.html",
        "monitoring/reports/20260802T060000.html",
        "monitoring/reports/20260801T000000.html",
    ]
    assert body["reports"][0]["url"].startswith("https://s3.example/")


def test_drift_status_keeps_only_the_newest_reports_when_limited(monkeypatch):
    monkeypatch.setattr(api_main, "S3_ARTIFACTS_BUCKET", "my-bucket")
    s3 = _FakeDriftS3Client()
    monkeypatch.setattr(
        api_main.boto3, "client",
        lambda service: s3 if service == "s3" else _FakeCloudWatchClient(),
    )

    body = client.get("/monitoring/drift?reports=1").json()

    assert [report["key"] for report in body["reports"]] == [
        "monitoring/reports/20260802T180000.html"
    ]
    # Only the kept key is signed -- paging must not sign every object it sees.
    assert s3.presigned == ["monitoring/reports/20260802T180000.html"]


def test_drift_status_with_no_checks_yet_reports_null_share(monkeypatch):
    monkeypatch.setattr(api_main, "S3_ARTIFACTS_BUCKET", "my-bucket")

    class _EmptyCloudWatch:
        def get_metric_statistics(self, **kwargs):
            return {"Datapoints": []}

    class _EmptyS3(_FakeDriftS3Client):
        def paginate(self, **kwargs):
            return iter([{}])

    monkeypatch.setattr(
        api_main.boto3, "client",
        lambda service: _EmptyS3() if service == "s3" else _EmptyCloudWatch(),
    )

    body = client.get("/monitoring/drift").json()

    assert body["current_share"] is None
    assert body["checked_at"] is None
    assert body["history"] == []
    assert body["reports"] == []


@pytest.mark.skipif(not _postgres_available(), reason="local Postgres not running (docker compose -f docker/docker-compose.yml up -d)")
class TestFeedbackAgainstRealPostgres:
    def setup_method(self):
        init_schema(DEFAULT_DSN)

    def test_predict_then_feedback_round_trip(self, monkeypatch):
        monkeypatch.setattr(api_main, "DATABASE_URL", DEFAULT_DSN)
        monkeypatch.setattr(api_main, "_model_version_id", None)
        monkeypatch.setattr(api_main, "MODEL_NAME", "test-feedback-round-trip")
        monkeypatch.setattr(api_main, "_load_model", lambda: _ConstantLogitModel())

        files = {"file": ("patch.png", _png_bytes(), "image/png")}
        predict_response = client.post("/predict", files=files)
        assert predict_response.status_code == 200
        prediction_id = predict_response.json()["prediction_id"]

        feedback_response = client.post(
            "/feedback", json={"prediction_id": prediction_id, "corrected_label": "no_tumor", "note": "mislabeled"},
        )
        assert feedback_response.status_code == 200

        with psycopg2.connect(DEFAULT_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT corrected_label, note FROM feedback WHERE id = %s",
                (feedback_response.json()["feedback_id"],),
            )
            row = cur.fetchone()
        assert row == ("no_tumor", "mislabeled")

        export_response = client.get("/feedback/export")
        assert export_response.status_code == 200
        with psycopg2.connect(DEFAULT_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT input_hash FROM predictions WHERE id = %s", (prediction_id,))
            input_hash = cur.fetchone()[0]
        assert {"input_hash": input_hash, "corrected_label": "no_tumor"} in export_response.json()

        # Uses the same fixed test image every run, so the shared local Postgres
        # instance may carry multiple rows for this input_hash across test runs --
        # assert at least one matches, not exactly one.
        recent_response = client.get("/predictions/recent")
        assert recent_response.status_code == 200
        matches = [row for row in recent_response.json() if row["input_hash"] == input_hash]
        assert len(matches) >= 1
        assert matches[0]["predicted_label"] == predict_response.json()["label"]
        assert matches[0]["confidence"] == pytest.approx(predict_response.json()["confidence"])

    def test_feedback_for_unknown_prediction_returns_404(self, monkeypatch):
        monkeypatch.setattr(api_main, "DATABASE_URL", DEFAULT_DSN)

        response = client.post("/feedback", json={"prediction_id": 999_999_999, "corrected_label": "tumor"})
        assert response.status_code == 404

    def test_list_models_returns_serving_version_with_prediction_count(self, monkeypatch):
        monkeypatch.setattr(api_main, "DATABASE_URL", DEFAULT_DSN)
        monkeypatch.setattr(api_main, "_model_version_id", None)
        monkeypatch.setattr(api_main, "MODEL_NAME", "test-list-models")
        monkeypatch.setattr(api_main, "_load_model", lambda: _ConstantLogitModel())

        files = {"file": ("patch.png", _png_bytes(), "image/png")}
        assert client.post("/predict", files=files).status_code == 200

        response = client.get("/models")
        assert response.status_code == 200
        ours = [row for row in response.json() if row["model_name"] == "test-list-models"]
        assert len(ours) == 1
        assert ours[0]["serving"] is True
        assert ours[0]["version"] == api_main.MODEL_VERSION
        # Shared local Postgres accumulates predictions across runs -- at least the one above.
        assert ours[0]["prediction_count"] >= 1
