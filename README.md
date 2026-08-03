# PathML Pipeline

A containerized MLOps pipeline that fine-tunes an open-source vision model on the [PatchCamelyon](https://github.com/basveeling/pcam) (PCam) histopathology dataset, serves predictions via FastAPI on AWS, and automates the full model lifecycle: experiment tracking, model registry, CI/CD-triggered retraining, and post-deployment drift monitoring.

**This is an infrastructure project, not a research project.** The goal is to demonstrate responsible, production-grade ML engineering in a high-stakes domain — not to advance pathology research or claim clinical validity.

Full design doc, week-by-week plan, and the longer-term Phase 2 vision (an ontology platform for pathology research data) live in [`docs/pathml-pipeline-project.md`](docs/pathml-pipeline-project.md).

## Status

**Weeks 1-5 complete.** Data pipeline, model, training loop, real evaluation suite (AUC/sensitivity/specificity, confusion matrix, PR curve, temperature-scaling calibration), MLflow tracking + model registry with a gated promotion script, Postgres prediction/feedback schema, and the inference API are all wired up and tested end-to-end. The API is deployed on AWS: S3 (model artifacts), ECR (image registry), ECS Fargate Spot (serving), and RDS Postgres (prediction logging), all provisioned via Terraform in `terraform/`. CI/CD (GitHub Actions, OIDC-authenticated) builds and redeploys on every merge; a `/feedback` loop lets retraining learn from corrections instead of the static dataset; a scheduled Lambda compares live traffic against the training distribution via Evidently and alerts through CloudWatch/SNS on drift or an error-rate spike.

**First baseline (ResNet18, 5 epochs, ImageNet-pretrained, no augmentation) on held-out test:**

| Metric | Value |
|---|---|
| AUC | 0.935 |
| Accuracy (at 0.5 threshold) | 76.5% |
| Sensitivity (at selected threshold) | 95.3% |
| Specificity (at that threshold) | 57.6% |

The operating threshold is chosen on the valid split to guarantee ≥95% sensitivity (missing a tumor is far costlier than a false alarm here — see the design doc). At that sensitivity, specificity is 57.6%, i.e. over 40% of non-tumor patches are false alarms. That's an honest baseline number, not a tuned final result — no augmentation, LR scheduling, or calibration has been applied yet.

## Stack

| Layer | Tool |
|---|---|
| Dataset | PatchCamelyon (PCam) |
| Model | Fine-tuned ResNet/ViT (PyTorch/timm) |
| Database | PostgreSQL (local → AWS RDS) |
| Experiment tracking | MLflow |
| Inference API | FastAPI + Docker |
| Cloud | AWS (ECR, ECS Fargate/Lambda, S3) |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | Prediction logging + drift detection (Evidently) + CloudWatch |

## Layout

```
├── docs/                 # design doc, architecture notes
├── src/pathml/
│   ├── data/             # dataset loading, preprocessing
│   ├── models/           # model definitions
│   ├── training/         # training scripts, config
│   ├── api/              # FastAPI inference service
│   ├── db/               # Postgres schema (predictions, model_versions, feedback)
│   └── monitoring/       # drift detection, logging
├── tests/
├── notebooks/            # exploration, EDA
├── scripts/              # data download, one-off utilities
├── docker/               # Dockerfiles, compose
├── terraform/            # AWS infrastructure as code
├── .github/workflows/    # CI/CD
└── data/                 # local dataset cache (gitignored)
```

## Setup

```
# download and verify the dataset (~8.5GB)
python scripts/download_pcam.py --data-dir data/pcam

# train
python -m pathml.training.train --data-dir data/pcam --out models/pcam_resnet18.pt

# evaluate on the held-out test split (picks a sensitivity-floor threshold on valid)
python scripts/evaluate.py --data-dir data/pcam --checkpoint models/pcam_resnet18.pt

# build and run the inference API in Docker
docker build -f docker/Dockerfile -t pathml-api .
docker run -d --name pathml-api -p 8000:8000 -v $(pwd)/models:/app/models:ro pathml-api
curl -X POST http://localhost:8000/predict -F "file=@path/to/patch.png"
```

## Frontend (operations console)

A Foundry-style ops dashboard (React + TypeScript + Tailwind, Blueprint icons, TanStack Query/Table, React Flow) lives in `frontend/`. Design tokens and conventions come from [`docs/frontend-design-system.md`](docs/frontend-design-system.md). Four views, each backed by read-only endpoints on the same API:

| View | Shows | Endpoints |
|---|---|---|
| Console | Health, recent predictions, latency chart, upload → predict → correct-label | `/health`, `/predictions/recent`, `/predict`, `/feedback` |
| Pipeline | Lifecycle DAG with live status per stage | `/health`, `/predictions/recent`, `/feedback/export` |
| Registry | Model versions, aliases, predictions served | `/models` |
| Drift | Drift share vs. alarm threshold, Evidently report links | `/monitoring/drift` |

```
# API must be running on :8000 (or set VITE_API_URL in frontend/.env)
cd frontend
npm install
npm run dev   # http://localhost:5173
```

The API's CORS allow-list defaults to the Vite dev origins; set `ALLOWED_ORIGINS` on the deployed service for anything else. Without `DATABASE_URL` the console still works — prediction logging/feedback panels degrade with explicit messages.

## Deploying to AWS

Requires `aws configure` already set up locally with a user that can create S3/ECR/RDS/ECS/IAM resources, plus Terraform and Docker.

```
# 1. provision S3, ECR, RDS, and the ECS Fargate service
cd terraform
terraform init
terraform plan -out=tfplan   # review before applying -- this creates real, billed resources
terraform apply tfplan

# 2. build the image (the Dockerfile installs CPU-only torch -- see the comment
#    in docker/Dockerfile; installing the default PyPI torch wheel pulls in
#    ~6GB of unused CUDA libraries) and push it to the ECR repo just created
ECR_URL=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ECR_URL"
docker build -f ../docker/Dockerfile -t pathml-api:latest ..
docker tag pathml-api:latest "$ECR_URL:latest"
docker push "$ECR_URL:latest"

# 3. upload the trained checkpoint to S3 -- the API downloads this at container
#    startup (see MODEL_S3_URI in the ECS task definition / pathml.api.main)
BUCKET=$(terraform output -raw s3_artifacts_bucket)
aws s3 cp ../models/pcam_resnet18.pt "s3://$BUCKET/models/pcam_resnet18.pt"

# 4. the first deployment fails to pull :latest since ECR was empty at apply time --
#    kick off a fresh one now that the image and model both exist
aws ecs update-service --cluster pathml-cluster --service pathml-api --force-new-deployment

# 5. find the running task's public IP and hit it
TASK_ARN=$(aws ecs list-tasks --cluster pathml-cluster --service-name pathml-api --query 'taskArns[0]' --output text)
ENI_ID=$(aws ecs describe-tasks --cluster pathml-cluster --tasks "$TASK_ARN" --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI_ID" --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
curl -X POST "http://$PUBLIC_IP:8000/predict" -F "file=@path/to/patch.png"
```

**Cost note:** the ECS service runs on Fargate Spot (not on-demand) and RDS defaults to `db.t4g.micro`, both chosen to stay close to the project's own ~$10/mo budget rule -- see the cost comments in `terraform/ecs.tf` and `terraform/rds.tf`. The public IP changes on every redeploy since there's no load balancer (deliberately, to avoid its fixed monthly cost); re-run step 5 after any `force-new-deployment`.

**Tearing down:** `terraform destroy` from `terraform/` removes everything (RDS is the piece that keeps billing while idle). To pause without destroying, set `desired_count = 0` (`terraform apply -var desired_count=0`) to stop the Fargate task while keeping RDS/S3/ECR intact.

## Monitoring & drift

A scheduled Lambda (`terraform/monitoring.tf`) compares live traffic against the training distribution using Evidently, publishes a `DriftShare` CloudWatch metric, and alerts via SNS. It deliberately runs outside the VPC -- see the docstring in `pathml.monitoring.lambda_handler` -- so it costs a few cents a month, not a NAT gateway.

```
# one-time setup after the first `terraform apply` (needs alert_email, which has no
# default so it's never committed -- pass via TF_VAR_alert_email)
python scripts/compute_drift_baseline.py --data-dir data/pcam --s3-uri s3://$BUCKET/monitoring/baseline.csv

# check your email and confirm the SNS subscription -- alarms fire either way,
# but nothing is delivered to an unconfirmed subscription
```

To run a check on demand instead of waiting for the schedule (default `rate(6 hours)`): `aws lambda invoke --function-name pathml-drift-monitor out.json`. Each run's full HTML report lands in `s3://$BUCKET/monitoring/reports/`.

The API exposes this read-only at `GET /monitoring/drift` (drift history from the CloudWatch metric, plus presigned links to those reports) for the console's Drift view. It needs `S3_ARTIFACTS_BUCKET` set — Terraform wires it, along with the alarm's own threshold so the chart draws the line that actually fires.

## Roadmap (Phase 1, ~6 weeks)

1. **Week 1 (done)** — dataloader, first fine-tuned model, FastAPI + Docker end-to-end
2. **Week 2 (done)** — real evaluation suite, MLflow tracking, model registry, Postgres
3. **Week 3 (done)** — AWS deployment (S3/ECR/ECS/RDS) via Terraform
4. **Week 4 (done)** — CI/CD, retraining loop off the feedback table
5. **Week 5 (done)** — prediction logging, drift detection, alerting
6. **Week 6** — polish, docs, teardown

See the [design doc](docs/pathml-pipeline-project.md) for full detail, scope rules, and Phase 2.
