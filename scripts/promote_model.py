#!/usr/bin/env python3
"""Gate and promote a registered model version from 'staging' to 'production'.

Loads the model version currently aliased `--alias` (default: staging) from
the MLflow registry -- not a local checkpoint file, since the point is to
evaluate exactly what's registered -- runs the same evaluation suite as
scripts/evaluate.py against it, and only re-aliases it to `production` (in
both the MLflow registry and the Postgres model_versions audit table) if it
clears the AUC and sensitivity gates. Otherwise it exits nonzero and leaves
the registry untouched.

If --s3-uri is given, also uploads the promoted checkpoint's state_dict to
that fixed S3 key, overwriting whatever was there -- this is the file the
deployed inference API downloads on startup (see MODEL_S3_URI in
pathml.api.main), so promoting a model here is what ships it.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import boto3
import mlflow
import torch
from mlflow import MlflowClient

from pathml.db.schema import DEFAULT_DSN, init_schema, record_model_version
from pathml.training.evaluation import evaluate_checkpoint
from pathml.training.train import DEFAULT_TRACKING_URI, REGISTERED_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", default="staging", help="registry alias to evaluate for promotion")
    parser.add_argument("--data-dir", default="data/pcam")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-sensitivity", type=float, default=0.95,
                         help="minimum tumor recall the operating threshold must guarantee")
    parser.add_argument("--min-auc", type=float, default=0.90,
                         help="minimum test AUC required to promote to production")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="Postgres DSN for the model_versions audit table")
    parser.add_argument("--s3-uri", default=None,
                         help="e.g. s3://bucket/models/pcam_resnet18.pt -- if set, uploads the promoted "
                              "checkpoint here after the gate passes, for the API to pick up")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    version_info = client.get_model_version_by_alias(args.model_name, args.alias)
    print(f"evaluating {args.model_name} v{version_info.version} (alias={args.alias!r}, "
          f"run_id={version_info.run_id})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(f"models:/{args.model_name}/{version_info.version}", map_location=device)

    result = evaluate_checkpoint(
        model, Path(args.data_dir), batch_size=args.batch_size,
        min_sensitivity=args.min_sensitivity, device=device,
    )
    print(f"  test: {result.test_metrics}")

    failures = []
    if result.test_metrics.auc < args.min_auc:
        failures.append(f"AUC {result.test_metrics.auc:.4f} < required {args.min_auc}")
    if result.test_metrics.sensitivity < args.min_sensitivity:
        failures.append(f"sensitivity {result.test_metrics.sensitivity:.4f} < required {args.min_sensitivity}")

    if failures:
        print("NOT promoting -- failed gate(s):")
        for reason in failures:
            print(f"  - {reason}")
        sys.exit(1)

    # Upload the artifact the API actually serves *before* the registry claims
    # this version is production -- if the upload fails, we want to exit here
    # with the registry still untouched, not have MLflow/Postgres say
    # "production" while the deployed checkpoint is still the old one.
    if args.s3_uri:
        bucket, _, key = args.s3_uri.removeprefix("s3://").partition("/")
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(model.state_dict(), tmp.name)
            boto3.client("s3").upload_file(tmp.name, bucket, key)
        print(f"uploaded checkpoint to {args.s3_uri}")

    client.set_registered_model_alias(args.model_name, "production", version_info.version)
    init_schema(args.dsn)
    record_model_version(
        args.dsn, args.model_name, version_info.version, "production", version_info.run_id,
    )
    print(f"promoted {args.model_name} v{version_info.version} to production "
          f"(MLflow registry + Postgres model_versions)")


if __name__ == "__main__":
    main()
