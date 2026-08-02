import io
import os

# lambda_handler reads its required config from the environment at import
# time (matching how it actually runs -- Terraform sets these on the real
# Lambda), so they need to exist before the module is first imported here.
os.environ.setdefault("ECS_CLUSTER", "pathml-cluster-test")
os.environ.setdefault("ECS_SERVICE", "pathml-api-test")
os.environ.setdefault("S3_ARTIFACTS_BUCKET", "pathml-artifacts-test")

import httpx
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pathml.db.schema import prediction_image_key
from pathml.monitoring import lambda_handler


def _png_bytes(color, seed=0) -> bytes:
    # Per-pixel noise, not a flat fill -- a solid color gives every std_*
    # feature exactly zero variance across the batch, which is a degenerate
    # case Evidently's stat tests handle poorly (divide-by-zero warnings).
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 15, size=(96, 96, 3))
    array = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _baseline_csv_bytes(n=30) -> bytes:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "mean_r": rng.normal(0.5, 0.05, n), "mean_g": rng.normal(0.5, 0.05, n), "mean_b": rng.normal(0.5, 0.05, n),
        "std_r": rng.normal(0.1, 0.01, n), "std_g": rng.normal(0.1, 0.01, n), "std_b": rng.normal(0.1, 0.01, n),
        "brightness": rng.normal(0.5, 0.05, n),
    })
    return df.to_csv(index=False).encode()


class _NoSuchKey(Exception):
    pass


class _FakeECS:
    def __init__(self, task_arns):
        self._task_arns = task_arns

    def list_tasks(self, cluster, serviceName):
        return {"taskArns": self._task_arns}

    def describe_tasks(self, cluster, tasks):
        return {
            "tasks": [{"attachments": [{"details": [{"name": "networkInterfaceId", "value": "eni-fake"}]}]}],
        }


class _FakeEC2:
    def describe_network_interfaces(self, NetworkInterfaceIds):
        return {"NetworkInterfaces": [{"Association": {"PublicIp": "203.0.113.5"}}]}


class _FakeS3:
    def __init__(self, objects_by_key):
        self._objects_by_key = objects_by_key
        self.uploaded = []
        self.exceptions = type("Exceptions", (), {"NoSuchKey": _NoSuchKey})()

    def get_object(self, Bucket, Key):
        if Key not in self._objects_by_key:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": io.BytesIO(self._objects_by_key[Key])}

    def upload_file(self, local_path, bucket, key):
        self.uploaded.append((local_path, bucket, key))


class _FakeCloudWatch:
    def __init__(self):
        self.put_calls = []

    def put_metric_data(self, Namespace, MetricData):
        self.put_calls.append((Namespace, MetricData))


def _install_fakes(monkeypatch, *, task_arns, s3_objects, recent_predictions):
    ecs, ec2, s3, cloudwatch = _FakeECS(task_arns), _FakeEC2(), _FakeS3(s3_objects), _FakeCloudWatch()

    def fake_client(service_name):
        return {"ecs": ecs, "ec2": ec2, "s3": s3, "cloudwatch": cloudwatch}[service_name]

    monkeypatch.setattr(lambda_handler.boto3, "client", fake_client)
    monkeypatch.setattr(
        lambda_handler.httpx, "get",
        lambda url, params, timeout: httpx.Response(200, json=recent_predictions, request=httpx.Request("GET", url)),
    )
    return s3, cloudwatch


def test_handler_skips_when_no_running_task(monkeypatch):
    _install_fakes(monkeypatch, task_arns=[], s3_objects={}, recent_predictions=[])

    result = lambda_handler.handler({}, None)

    assert result == {"skipped": "no_running_task"}


def test_handler_skips_when_too_few_samples(monkeypatch):
    _install_fakes(
        monkeypatch, task_arns=["arn:task/1"], s3_objects={},
        recent_predictions=[{"input_hash": "missing", "predicted_label": "tumor", "confidence": 0.9}],
    )

    result = lambda_handler.handler({}, None)

    assert result["skipped"] == "insufficient_samples"


def test_handler_publishes_drift_metric_and_saves_report(monkeypatch):
    recent = [
        {"input_hash": f"hash-{i}", "predicted_label": "tumor", "confidence": 0.9}
        for i in range(lambda_handler.MIN_SAMPLES)
    ]
    s3_objects = {"monitoring/baseline.csv": _baseline_csv_bytes()}
    for i, prediction in enumerate(recent):
        s3_objects[prediction_image_key(prediction["input_hash"])] = _png_bytes((200, 30, 30), seed=i)

    s3, cloudwatch = _install_fakes(
        monkeypatch, task_arns=["arn:task/1"], s3_objects=s3_objects, recent_predictions=recent,
    )

    result = lambda_handler.handler({}, None)

    assert result["sample_count"] == len(recent)
    assert 0.0 <= result["drift_share"] <= 1.0
    assert len(cloudwatch.put_calls) == 1
    namespace, metric_data = cloudwatch.put_calls[0]
    assert namespace == lambda_handler.CLOUDWATCH_NAMESPACE
    assert metric_data[0]["MetricName"] == "DriftShare"
    assert metric_data[0]["Value"] == pytest.approx(result["drift_share"])
    assert len(s3.uploaded) == 1
    assert s3.uploaded[0][1:] == ("pathml-artifacts-test", result["report_key"])
