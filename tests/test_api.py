from fastapi.testclient import TestClient

from pathml.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_checkpoint_returns_503():
    files = {"file": ("patch.png", b"not a real image", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 503
