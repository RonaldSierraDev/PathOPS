# PathML Pipeline

A containerized MLOps pipeline that fine-tunes an open-source vision model on the [PatchCamelyon](https://github.com/basveeling/pcam) (PCam) histopathology dataset, serves predictions via FastAPI on AWS, and automates the full model lifecycle: experiment tracking, model registry, CI/CD-triggered retraining, and post-deployment drift monitoring.

**This is an infrastructure project, not a research project.** The goal is to demonstrate responsible, production-grade ML engineering in a high-stakes domain — not to advance pathology research or claim clinical validity.

Full design doc, week-by-week plan, and the longer-term Phase 2 vision (an ontology platform for pathology research data) live in [`docs/pathml-pipeline-project.md`](docs/pathml-pipeline-project.md).

## Status

Groundwork stage — repo structure is in place; dataset download and the first training/inference pass haven't started yet.

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

Not yet — dependencies, environment, and the first dataloader are the next milestone.

## Roadmap (Phase 1, ~6 weeks)

1. **Week 1** — dataloader, first fine-tuned model, FastAPI + Docker end-to-end
2. **Week 2** — real evaluation suite, MLflow tracking, model registry, Postgres
3. **Week 3** — AWS deployment (ECR/ECS/RDS) via Terraform
4. **Week 4** — CI/CD, retraining loop off the feedback table
5. **Week 5** — prediction logging, drift detection, alerting
6. **Week 6** — polish, docs, teardown

See the [design doc](docs/pathml-pipeline-project.md) for full detail, scope rules, and Phase 2.
