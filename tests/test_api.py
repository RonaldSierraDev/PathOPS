from pathlib import Path

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
