import io

import httpx
import pytest
from PIL import Image

import pathml.training.feedback as feedback_module
from pathml.db.schema import prediction_image_key
from pathml.training.feedback import FeedbackDataset


def _png_bytes(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (96, 96), color=color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeS3Client:
    def __init__(self, images_by_key):
        self._images_by_key = images_by_key

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self._images_by_key[Key])}


def test_feedback_dataset_loads_corrected_examples(monkeypatch):
    input_hash = "deadbeef-test-feedback-dataset"
    corrections = [{"input_hash": input_hash, "corrected_label": "tumor"}]
    image_bytes = _png_bytes(color=(200, 50, 50))
    key = prediction_image_key(input_hash)

    monkeypatch.setattr(feedback_module.httpx, "get", _fake_response(200, json=corrections))
    monkeypatch.setattr(feedback_module.boto3, "client", lambda service: _FakeS3Client({key: image_bytes}))

    dataset = FeedbackDataset("http://fake-api/feedback/export", images_s3_bucket="fake-bucket")

    assert len(dataset) == 1
    image_tensor, label = dataset[0]
    assert image_tensor.shape == (3, 96, 96)
    assert label.item() == 1  # "tumor" -> index 1 in LABELS


def _fake_response(status_code, **kwargs):
    return lambda url, timeout: httpx.Response(status_code, request=httpx.Request("GET", url), **kwargs)


def test_feedback_dataset_empty_when_no_corrections(monkeypatch):
    monkeypatch.setattr(feedback_module.httpx, "get", _fake_response(200, json=[]))
    monkeypatch.setattr(feedback_module.boto3, "client", lambda service: _FakeS3Client({}))

    dataset = FeedbackDataset("http://fake-api/feedback/export", images_s3_bucket="fake-bucket")

    assert len(dataset) == 0


def test_feedback_dataset_raises_on_export_error(monkeypatch):
    monkeypatch.setattr(feedback_module.httpx, "get", _fake_response(503))

    with pytest.raises(httpx.HTTPStatusError):
        FeedbackDataset("http://fake-api/feedback/export", images_s3_bucket="fake-bucket")
