import io
from pathlib import Path

import h5py
import httpx
import mlflow
import numpy as np
from mlflow import MlflowClient
from PIL import Image

import pathml.training.feedback as feedback_module
from pathml.data.dataset import SPLIT_FILES
from pathml.training.train import REGISTERED_MODEL_NAME, train


def _write_split(tmp_path: Path, split: str, n: int) -> None:
    x_name, y_name = SPLIT_FILES[split]
    images = np.random.randint(0, 256, size=(n, 96, 96, 3), dtype=np.uint8)
    labels = np.array([i % 2 for i in range(n)], dtype=np.uint8).reshape(n, 1, 1, 1)

    with h5py.File(tmp_path / x_name, "w") as f:
        f.create_dataset("x", data=images)
    with h5py.File(tmp_path / y_name, "w") as f:
        f.create_dataset("y", data=labels)


def test_train_runs_and_saves_checkpoint(tmp_path):
    _write_split(tmp_path, "train", n=8)
    _write_split(tmp_path, "valid", n=4)
    out_path = tmp_path / "checkpoint.pt"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    train(
        data_dir=tmp_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        model_name="resnet18",
        out_path=out_path,
        pretrained=False,
        tracking_uri=tracking_uri,
    )

    assert out_path.exists()

    mlflow.set_tracking_uri(tracking_uri)
    latest = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "staging")
    assert latest is not None


def test_train_blends_in_feedback_examples(tmp_path, monkeypatch):
    _write_split(tmp_path, "train", n=8)
    _write_split(tmp_path, "valid", n=4)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    input_hash = "test-train-feedback-hash"
    corrections = [{"input_hash": input_hash, "corrected_label": "tumor"}]

    buf = io.BytesIO()
    Image.new("RGB", (96, 96), color=(10, 20, 30)).save(buf, format="PNG")
    image_bytes = buf.getvalue()

    class _FakeS3Client:
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(image_bytes)}

    export_url = "http://fake-api/feedback/export"
    monkeypatch.setattr(
        feedback_module.httpx, "get",
        lambda url, timeout: httpx.Response(200, json=corrections, request=httpx.Request("GET", url)),
    )
    monkeypatch.setattr(feedback_module.boto3, "client", lambda service: _FakeS3Client())

    logged_params = {}
    original_log_params = mlflow.log_params
    monkeypatch.setattr(mlflow, "log_params", lambda params: (logged_params.update(params), original_log_params(params)))

    train(
        data_dir=tmp_path,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        model_name="resnet18",
        out_path=tmp_path / "checkpoint.pt",
        pretrained=False,
        tracking_uri=tracking_uri,
        feedback_export_url=export_url,
        feedback_images_s3_bucket="fake-bucket",
    )

    assert logged_params["feedback_size"] == 1
    assert logged_params["train_size"] == 8 + 1
