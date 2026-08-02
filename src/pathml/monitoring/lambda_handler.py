"""Scheduled drift-monitor Lambda entry point.

Runs outside the VPC entirely (see the module docstring in
pathml.training.feedback for why): it resolves the inference API's current
public IP itself (there's no ALB/stable DNS, a deliberate cost tradeoff --
see terraform/ecs.tf) and pulls recent traffic over plain HTTPS from
/predictions/recent, rather than connecting to the private RDS instance or
needing a NAT gateway/VPC endpoints just to reach CloudWatch/SNS.
"""
import io
import os
from datetime import datetime, timezone

import boto3
import httpx
import pandas as pd
from PIL import Image

from pathml.db.schema import prediction_image_key
from pathml.monitoring.drift import compute_drift, extract_features

ECS_CLUSTER = os.environ["ECS_CLUSTER"]
ECS_SERVICE = os.environ["ECS_SERVICE"]
S3_BUCKET = os.environ["S3_ARTIFACTS_BUCKET"]
BASELINE_KEY = os.environ.get("DRIFT_BASELINE_KEY", "monitoring/baseline.csv")
RECENT_LIMIT = int(os.environ.get("DRIFT_RECENT_LIMIT", "100"))
MIN_SAMPLES = int(os.environ.get("DRIFT_MIN_SAMPLES", "10"))
CLOUDWATCH_NAMESPACE = os.environ.get("CLOUDWATCH_NAMESPACE", "PathML/Monitoring")


def _resolve_api_base_url() -> str | None:
    ecs = boto3.client("ecs")
    task_arns = ecs.list_tasks(cluster=ECS_CLUSTER, serviceName=ECS_SERVICE).get("taskArns", [])
    if not task_arns:
        return None

    task = ecs.describe_tasks(cluster=ECS_CLUSTER, tasks=task_arns[:1])["tasks"][0]
    eni_id = next(
        d["value"] for d in task["attachments"][0]["details"] if d["name"] == "networkInterfaceId"
    )
    eni = boto3.client("ec2").describe_network_interfaces(NetworkInterfaceIds=[eni_id])
    public_ip = eni["NetworkInterfaces"][0]["Association"]["PublicIp"]
    return f"http://{public_ip}:8000"


def _load_baseline() -> pd.DataFrame:
    obj = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=BASELINE_KEY)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _load_recent_features(api_base_url: str) -> pd.DataFrame:
    response = httpx.get(f"{api_base_url}/predictions/recent", params={"limit": RECENT_LIMIT}, timeout=30)
    response.raise_for_status()
    recent = response.json()

    s3 = boto3.client("s3")
    rows = []
    for prediction in recent:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=prediction_image_key(prediction["input_hash"]))
        except s3.exceptions.NoSuchKey:
            continue
        image = Image.open(io.BytesIO(obj["Body"].read()))
        rows.append(extract_features(image))
    return pd.DataFrame(rows)


def handler(event, context) -> dict:
    api_base_url = _resolve_api_base_url()
    if api_base_url is None:
        print("no running API task found (service may be scaled to 0) -- skipping this check")
        return {"skipped": "no_running_task"}

    current_df = _load_recent_features(api_base_url)
    if len(current_df) < MIN_SAMPLES:
        print(f"only {len(current_df)} recent predictions with stored images -- need >= {MIN_SAMPLES}, skipping")
        return {"skipped": "insufficient_samples", "count": len(current_df)}

    reference_df = _load_baseline()
    result = compute_drift(reference_df, current_df)

    boto3.client("cloudwatch").put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[{"MetricName": "DriftShare", "Value": result.share, "Unit": "None"}],
    )

    report_key = f"monitoring/reports/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.html"
    local_path = "/tmp/drift_report.html"
    result.snapshot.save_html(local_path)
    boto3.client("s3").upload_file(local_path, S3_BUCKET, report_key)

    print(f"drift share={result.share:.3f} over {len(current_df)} samples, report at s3://{S3_BUCKET}/{report_key}")
    return {"drift_share": result.share, "sample_count": len(current_df), "report_key": report_key}
