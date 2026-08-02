from pathlib import Path

import torch
from fastapi.testclient import TestClient

import pathml.api.main as api_main
from pathml.api.main import app

client = TestClient(app)


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
        api_main, "_log_prediction", lambda input_bytes, label, confidence: logged.update(
            input_bytes=input_bytes, label=label, confidence=confidence,
        ),
    )

    image_bytes = _png_bytes()
    files = {"file": ("patch.png", image_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert logged["input_bytes"] == image_bytes
    assert logged["label"] == response.json()["label"]


def _png_bytes() -> bytes:
    import io as _io

    from PIL import Image as _Image

    buf = _io.BytesIO()
    _Image.new("RGB", (96, 96)).save(buf, format="PNG")
    return buf.getvalue()


class _ConstantLogitModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.0, 5.0]]).expand(x.shape[0], 2)
