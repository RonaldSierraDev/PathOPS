# PathML Pipeline

A containerized MLOps pipeline that fine-tunes an open-source vision model on the [PatchCamelyon](https://github.com/basveeling/pcam) (PCam) histopathology dataset, serves predictions via FastAPI on AWS, and automates the full model lifecycle: experiment tracking, model registry, CI/CD-triggered retraining, and post-deployment drift monitoring.

**This is an infrastructure project, not a research project.** The goal is to demonstrate responsible, production-grade ML engineering in a high-stakes domain — not to advance pathology research or claim clinical validity.

Full design doc, week-by-week plan, and the longer-term Phase 2 vision (an ontology platform for pathology research data) live in [`docs/pathml-pipeline-project.md`](docs/pathml-pipeline-project.md).

## Status

**Week 1 complete.** Data pipeline, model, training loop, evaluation, threshold selection, and inference API are all wired up and tested end-to-end on real data, and the API runs in Docker: `curl` a running container and get a real prediction back, verified against a trained checkpoint.

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

## Roadmap (Phase 1, ~6 weeks)

1. **Week 1 (done)** — dataloader, first fine-tuned model, FastAPI + Docker end-to-end
2. **Week 2** — real evaluation suite, MLflow tracking, model registry, Postgres
3. **Week 3** — AWS deployment (ECR/ECS/RDS) via Terraform
4. **Week 4** — CI/CD, retraining loop off the feedback table
5. **Week 5** — prediction logging, drift detection, alerting
6. **Week 6** — polish, docs, teardown

See the [design doc](docs/pathml-pipeline-project.md) for full detail, scope rules, and Phase 2.
