# PathML — From MLOps Pipeline to Pathology Research Platform

This document has two phases.

**Phase 1 (this summer, ~6 weeks):** a production MLOps pipeline for histopathology classification. Self-contained, resume-ready, finishable before Fall 2026. This is the part you build now.

**Phase 2 (post-semester, multi-quarter):** evolve that pipeline into a Foundry-style *ontology platform* for pathology research data — a system that models entities and their relationships (patients, slides, findings, models) and lets researchers query, annotate, and trace them. This is the flagship's long arc, and it's deliberately open-ended.

**The relationship between them matters:** Phase 2 is only credible *because* Phase 1 exists. An ontology built before you have real data and a real user is the classic platform failure mode. Build Phase 1 for yourself, find a lab, then let their actual workflow drive Phase 2. Don't invert that order.

---

## PHASE 1 — Production MLOps Pipeline

**Goal:** Build an end-to-end MLOps system that trains, deploys, monitors, and retrains an open-source vision model for cancer-patch detection on histopathology data — demonstrating production ML engineering in a high-stakes, real-world domain.

**Timeline:** ~6 weeks (finish before Fall 2026 semester)
**Resume gap this fills:** Production/cloud infrastructure — deployment, IaC, CI/CD, monitoring — complementing existing ML fundamentals (NumPy NN) and systems work (kernel module).

---

## Project Summary (resume-ready framing)

A containerized ML pipeline that fine-tunes an open-source vision model on the PatchCamelyon histopathology dataset, serves predictions through a FastAPI inference service on AWS, and automates the full model lifecycle: experiment tracking, model registry, CI/CD-triggered retraining and redeployment, and post-deployment drift monitoring.

**Key claim:** This is an infrastructure project, not a research project. The pitch is "I can productionize ML responsibly in a domain where mistakes matter" — not "I advanced pathology research."

---

## Core Stack

| Layer | Tool | Why |
|---|---|---|
| Dataset | PatchCamelyon (PCam) — ~327k labeled 96×96 patches | Well-documented, tractable size, binary classification (tumor / no tumor) |
| Model | Fine-tuned open-source vision model (e.g., ResNet or ViT via PyTorch/timm) | Target ~0.95+ AUC — competent and rigorously evaluated, not SOTA-chasing |
| Database | PostgreSQL (local container in dev → AWS RDS in prod) | Prediction audit trail, model version registry, feedback loop for retraining |
| Training | Local GPU or Google Colab | Keeps AWS costs near zero — only inference lives in the cloud |
| Experiment tracking | MLflow (or Weights & Biases) | Runs, params, metrics, model registry |
| Inference API | FastAPI + Docker | Already know FastAPI — leverage it. **Stays Python in Phase 1 on purpose** (see note below); Rust enters in Phase 2 where it earns its place. |
| Cloud | AWS: ECR (images), ECS Fargate or Lambda (serving), S3 (data/artifacts) | Industry-standard; free-tier friendly for inference |
| IaC | Terraform | Infrastructure-as-code is a major interview signal |
| CI/CD | GitHub Actions | Test → build → push image → deploy on merge; retrain trigger on data change |
| Monitoring | Prediction logging + drift detection (e.g., Evidently) + CloudWatch | The piece most student projects skip — differentiator |
| Frontend (optional) | Simple upload UI (TypeScript) | Upload a patch → prediction + confidence |

---

## Architecture (target end state)

```
                ┌─────────────┐
                │  PCam data   │──► S3 (versioned raw + processed)
                └─────────────┘
                       │
              train.py (local/Colab)
                       │
              ┌────────▼────────┐
              │ MLflow tracking  │──► Model Registry (versioned)
              └────────┬────────┘
                       │  promote model
        GitHub Actions CI/CD (test → build → deploy)
                       │
              ┌────────▼────────┐
              │  Docker image    │──► ECR
              └────────┬────────┘
                       │
              ┌────────▼────────┐      ┌──────────────────┐
              │ FastAPI on ECS/  │─────►│ Prediction logs   │
              │ Lambda (AWS)     │      │ + drift monitor   │
              └────────┬────────┘      │ + CloudWatch      │
                       │               └──────────────────┘
              ┌────────▼────────┐
              │ Upload UI (opt.) │
              └─────────────────┘
```

---

## Week-by-Week Plan

### Week 1 — Ugly but end-to-end (the most important week)
- [ ] Download PCam, write a data loader, sanity-check class balance
- [ ] Fine-tune a small pretrained model locally — target "good enough" (~85%+ AUC), not SOTA
- [ ] Wrap inference in FastAPI: `POST /predict` takes an image, returns label + confidence
- [ ] Dockerize the API and run it locally
- **Exit criteria:** `curl` a local container with a patch image and get a prediction back

### Week 2 — Rigorous evaluation, experiment tracking & model registry
- [ ] **Build a real evaluation suite:** AUC + sensitivity/specificity at a chosen operating threshold, confusion matrix, PR curve. Accuracy alone is near-meaningless for a screening task.
- [ ] **Choose and justify the operating threshold explicitly** — false negatives (missed metastasis) are far costlier than false positives. Document the tradeoff.
- [ ] Add temperature scaling (Guo et al.) and plot reliability diagrams before/after
- [ ] Integrate MLflow: log params, metrics, artifacts for every training run
- [ ] Set up model registry with staging/production stages
- [ ] Refactor training into a reproducible script (config-driven, seeded)
- [ ] Store dataset + model artifacts in S3
- [ ] Stand up PostgreSQL locally (Docker container) with schema: `predictions`, `model_versions`, `feedback`

### Week 3 — Cloud deployment
- [ ] Push Docker image to ECR
- [ ] Deploy inference service on ECS Fargate (or Lambda if latency/size permits)
- [ ] Provision RDS PostgreSQL; point the inference service at it for prediction logging
- [ ] Write Terraform for all AWS resources (S3, ECR, ECS, RDS, IAM roles, security groups)
- [ ] **Set a billing alarm BEFORE provisioning RDS** — a running instance bills continuously and is the most likely source of a surprise charge. Verify current AWS free tier terms directly; they have changed recently.
- **Exit criteria:** public (or auth-gated) endpoint serving predictions from AWS, fully reproducible via `terraform apply`

### Week 4 — CI/CD
- [ ] GitHub Actions: lint + tests on PR; on merge to main → build image → push to ECR → redeploy
- [ ] Add a retraining workflow: manual dispatch (or data-change trigger) → retrain → log to MLflow → require manual promotion to production
- [ ] **Close the loop:** retraining pulls corrected labels from the Postgres `feedback` table, not a static dataset. This is what separates a demo pipeline from one that models how production ML actually evolves.
- **Exit criteria:** merging a PR ships a new model/API version with zero manual steps

### Week 5 — Monitoring & drift
- [ ] Log every prediction (input hash, output, confidence, latency) to S3/CloudWatch
- [ ] Add drift detection (Evidently or custom): compare live input distribution vs. training distribution
- [ ] Alarm/alert on drift or error-rate spike
- **Exit criteria:** demonstrable drift alert (e.g., feed it non-pathology images and show the monitor firing)

### Week 6 — Polish & presentation
- [ ] Optional: minimal upload UI (TypeScript) → prediction + confidence display
- [ ] README with architecture diagram, setup instructions, and design decisions
- [ ] Short demo video or GIF
- [ ] Write resume bullets + prepare to defend every component in interviews
- [ ] Tear down / cost-check AWS resources; document teardown in Terraform

---

## Scope Rules (read when tempted to over-engineer)

1. **The model must be genuinely good, but not maximally good.** Hit ~0.95+ AUC and stop. A pipeline serving a weak model discredits the whole project — but chasing the leaderboard past "clearly competent" buys nothing. Rigor of evaluation > size of the metric. Cap training work at 1 week.
2. **End-to-end first, hardened second.** A working ugly pipeline in week 1 beats a perfect half-pipeline in week 6.
3. **Only inference lives on AWS.** Train locally/Colab. If a bill could exceed ~$10/mo, redesign.
4. **Cut list (in order, if behind schedule):** frontend UI → drift detection → automated retraining trigger. Never cut: Docker, Terraform, CI/CD, cloud deployment.
5. **Honest framing everywhere:** demo, README, and interviews all say "production ML infrastructure demonstrated on a medical dataset" — never claim clinical validity.
6. **Phase 1 stays Python — this is deliberate, not lazy.** The thesis is "productionize ML fast and responsibly." Rewriting the backend in Rust/C++ here means learning a new language's async/web ecosystem *on top of* Docker + Terraform + AWS (all already new) on a 6-week clock, to reimplement what FastAPI does well. That undercuts the ship-fast story. Robustness and security in Phase 1 come from architecture — validation (Pydantic), tests, access control, TLS, audit logging — not from language choice. Rust is scheduled for Phase 2, where it earns its keep (see "Language & Architecture" there).

---

## Learning Outcomes (interview talking points)

- Containerization and image-based deployment (Docker, ECR)
- Infrastructure-as-code (Terraform) and reproducible cloud environments
- ML lifecycle management: experiment tracking, model registry, versioning (MLflow)
- CI/CD for ML systems — how deploying models differs from deploying code
- Post-deployment concerns: prediction logging, data drift, monitoring, alerting
- Cost-aware architecture decisions on AWS
- Working with real-world medical imaging data: class imbalance, evaluation metrics (AUC vs. accuracy), and why calibration/confidence matters in high-stakes domains

## Risks & Mitigations

- **AWS cost surprise** → billing alarm at $5 on day one; train off-cloud; Fargate spot or Lambda for serving
- **PCam download/size friction** → it's ~7GB; start the download day one, keep a small dev subset for iteration
- **Terraform learning curve** → start with S3+ECR only, expand incrementally; it's fine if week 3 spills into week 4
- **Semester starts early / life happens** → the cut list exists for a reason; weeks 1–4 alone are still a complete, resume-worthy project

---

**Repo name suggestion:** `pathml-pipeline` or `histopath-mlops`
**First action this week:** Week 1, task 1 — download PCam and get a dataloader running tonight.

**Phase 1 is done when:** the pipeline is deployed, reproducible via `terraform apply`, and you can defend every component. Do not start Phase 2 until this is true. A finished Phase 1 is already a strong flagship on its own.

---
---

# PHASE 2 — Pathology Research Ontology Platform

**What this is:** the evolution of the Phase 1 pipeline into something closer to Palantir Foundry — not a classifier with a dashboard, but a *data operating system* for pathology research. The core idea you described: a platform that builds an **ontology** of heterogeneous research data (patients, cases, slides, regions, annotations, findings, models, cohorts, studies) so researchers can see and query the relationships between them, not just the raw files.

**What makes it "an ontology" and not just a database:** three things, and all three are the actual engineering.
1. **A typed object/relationship model** — real-world entities as first-class objects with defined links, not tables you join by hand.
2. **Actions that write back** — a pathologist correcting a model call isn't a row update; it's a typed, attributed, timestamped event. State changes through actions, and actions are recorded.
3. **End-to-end lineage** — anything on screen traces back to its source: this finding → this region → this slide → this model version → this training run. Reproducibility is the product, not a nice-to-have.

**Honest framing (carry this everywhere):** even at full scale this is a *research-infrastructure* contribution, not a scientific one. You're not out-modeling the field; you're building the platform the field mostly lacks. Academic labs are notorious for brilliant modeling sitting on unreproducible pipelines — folders of TIFFs, notebooks with hardcoded paths, models nobody can re-derive. Being the person who fixes *that* is a genuine and attractive value proposition. Say it that way.

**Timeline:** deliberately open-ended, multi-quarter, done alongside coursework. This is not a sprint and shouldn't be scoped like one.

---

## The critical precondition: a real user

Foundry's ontology works because it was forged against actual deployments, not designed in the abstract. Yours needs the same. **Before building the ontology layer, get a real user with real data.**

- FIU has research labs; the Miami area has medical institutions. The pitch practically writes itself: *"I built this MLOps pipeline for histopathology — here's the deployed repo. You have data and grad students but no infrastructure engineer. Can I help productionize your lab's models?"*
- A PI who says yes gives you the two things you can't manufacture: real data and a real workflow to model. That workflow *is* your ontology spec.
- This is also the bridge to your stated long-term goal of developing for pathology research — it's a far more plausible route in than trying to compete on modeling.

**Governance note:** the moment real patient data is involved, de-identification, PHI handling, IRB/data-use agreements, and access control stop being optional. Some of Phase 2's "boring" work (below) is what makes a lab able to say yes at all.

---

## Language & Architecture: a polyglot core (Rust + Python)

Phase 2 is where a systems language earns its place — but as a **polyglot architecture**, not a wholesale rewrite. The boundary is drawn by *which property each component needs*, not by preference.

**Decide by requirement, not by category:**
- **Data accuracy** comes from schema constraints, validation, and tests — a design property, largely language-independent.
- **Patient-data security** comes from auth, RBAC, encryption in transit/at rest, injection prevention, and audit logging — architecture (see 2E), not language.
- **What a systems language actually adds:** compile-time memory safety, data-race freedom, a strong type system, and exhaustive error handling — i.e., *robustness* and the elimination of a class of security bugs.

**Rust, not C++.** For a greenfield, security-sensitive, patient-data system, C++ is close to self-defeating — choosing a memory-*unsafe* language for the one system where memory-safety bugs are security vulnerabilities. Rust gives memory safety and data-race freedom at compile time with no GC. C++ would only make sense if tied to an existing C++ codebase or a specific library with no Rust equivalent. (Note: service-level Rust is a different skillset than the kernel C on your resume — budget for that.)

**The split:**

| Component | Language | Why |
|---|---|---|
| WSI tiling / image pipeline (gigapixel, CPU-bound) | **Rust** | Performance-critical; Rust's sweet spot |
| Ontology core + data services | **Rust** | Correctness and concurrency matter most here |
| High-throughput embedding serving | **Rust** | Latency/throughput-sensitive |
| Model training, experiment tracking, MLOps | **Python / PyTorch** | The entire ML ecosystem (MLflow etc.) lives here |
| Foundation-model inference | **Python** (or Rust via `tch-rs`/LibTorch if latency demands) | Default to Python; move to Rust only if profiling justifies it |
| Glue / orchestration / web API | **Python** initially, migrate hot paths to Rust | Don't rewrite what works until it's a bottleneck |

**Interop:** either **PyO3** bindings (Rust core called from Python — the pattern behind Polars, Pydantic v2, and HF tokenizers) or **separate services over gRPC** (tonic on the Rust side). Prefer gRPC when the components have independent lifecycles; prefer PyO3 when Rust is a library inside a Python process.

**This is a real industry pattern, and a strong interview story:** "Rust for the performance- and correctness-critical paths, Python for ML orchestration" is a sophisticated, defensible design decision — and Rust alongside your kernel contribution tells a rare systems-plus-ML narrative. The failure mode to avoid is reaching for Rust *before* profiling shows you need it; migrate hot paths deliberately, don't rewrite on principle.

---

## Phase 2 Sub-Phases

Sequence these; don't parallelize. Each is a milestone that stands on its own.

### 2A — Whole-Slide Images (the real unlock)
Everything downstream depends on this. Move from 96×96 patches to gigapixel WSIs.
- OpenSlide for reading; a tiling pipeline; pyramidal/tiled storage; OpenSeadragon for deep-zoom viewing in the browser.
- Slide-level aggregation via MIL — now *in scope* where it wasn't in Phase 1. Start from CLAM (attention-MIL) or TransMIL; see the reading list.
- This single change makes the project resemble real computational pathology instead of a benchmark exercise, and it forces genuinely hard engineering: storage, tiling throughput, lazy loading.
- **Foundation-model option:** instead of training slide models from scratch, use a pathology foundation model (UNI/CONCH for tiles, Prov-GigaPath for slides) as a frozen encoder and build an embedding store. Serving embeddings at scale is a more research-relevant infrastructure problem than serving a ResNet.

### 2B — The Ontology Layer
The Foundry-style core. The domain hands you the schema:

```
Patient ─< Case ─< Slide ─< Region ─< Annotation ─< Finding
                     │
                     └─ InferenceRun >─ ModelVersion >─ Model
Cohort ─< Study      (Cohort groups Cases; Study groups analyses)
```

- Model these as typed objects with defined relationships, backed by Postgres (relational) and, where relationship-traversal is the point, a graph store (e.g., Neo4j) or graph queries over Postgres.
- Implement **actions**: `correctPrediction`, `annotateRegion`, `promoteModel`, `assignToCohort` — each a typed, attributed, logged event, not a raw write.
- Implement **lineage**: every object carries provenance; any finding is walkable back to source tile + model version + training run.
- **Node-link graph views now become legitimate.** (In Phase 1 they'd have been decoration — you have no relationships worth exploring. Here, exploring patient/case/slide/model relationships visually *is* the point.)

### 2C — Cohort Query & Human-in-the-Loop
Where the ontology pays off.
- **Cohort queries** the raw-file world can't answer: *"every slide from patients with subtype X where model confidence was high but the pathologist disagreed."* Trivial with a good object model; near-impossible over folders of TIFFs. This is the single most compelling demo of why the ontology exists.
- **Annotation tooling:** pathologists draw regions directly on slides; model-assisted pre-annotation; inter-annotator agreement metrics. This is the piece research groups most visibly lack, and it feeds the Phase 1 feedback loop you already built.

### 2D — Embeddings, Search & Similarity
- Embedding store over tiles/slides (from the 2A foundation-model encoders); vector search.
- *"Find morphologically similar regions across the cohort"* — a genuinely useful research primitive, and a much more relevant infrastructure problem than classification.
- Connects to CONCH-style text-queryable retrieval if you go vision-language.

### 2E — Multi-Site & Governance (unglamorous, mandatory)
- Site-stratified evaluation (the demographic-bias and hospital-shift problem made concrete — see Vaidya et al.).
- De-identification / PHI handling, role-based access control, full audit logging.
- Building this early is a large part of what makes a lab able to adopt your platform at all.

---

## Phase 2 Architecture (target)

```
   Slide ingest (OpenSlide) ──► Tiling ──► Pyramidal tile store (S3)
          │                                        │
          │                              Foundation-model encoder
          │                              (UNI / Prov-GigaPath, frozen)
          │                                        │
          │                                 Embedding store + vector search
          ▼                                        │
   ┌──────────────────────── ONTOLOGY CORE ───────────────────────┐
   │  Typed objects: Patient·Case·Slide·Region·Annotation·Finding  │
   │                 Model·ModelVersion·InferenceRun·Cohort·Study  │
   │  Actions (typed, attributed, logged)   Lineage (source-traced)│
   │  Backing: Postgres (+ graph store / graph queries)            │
   └───────────────┬───────────────────────────────┬──────────────┘
                   │                                │
        Cohort query engine              Annotation + HITL tooling
                   │                                │
   ┌───────────────▼────────────────────────────────▼─────────────┐
   │  Web app: slide viewer (OpenSeadragon) · object explorer ·     │
   │  node-link relationship graph · cohort builder · audit views  │
   └───────────────────────────────────────────────────────────────┘
                   │
        Governance layer: de-id · RBAC · audit log · site-stratified eval
```

---

## Phase 2 Scope Rules (the platform failure modes)

1. **No ontology before a real user.** Building the abstract object model in advance is exactly how platform projects die. Get a lab and real data first; their workflow is the spec.
2. **Each sub-phase must stand alone.** 2A (WSI support) is a complete, demoable milestone by itself. Never let "the whole platform" be the smallest shippable unit.
3. **Governance is not optional once real data appears.** De-id and access control before patient data touches the system, not after.
4. **Resist re-implementing the model layer.** Use foundation models as frozen encoders. The contribution is the platform, not a new architecture.
5. **"Foundry-like" is about the object/action/lineage model, not about cloning the UI.** Spend effort on the data model and provenance, not on visual mimicry.
6. **This runs alongside coursework — protect the cadence.** A slow, correct platform beats a burned-out sprint. If a quarter is heavy, sub-phases pause; they don't get rushed.

---

## Phase 2 Learning Outcomes / Interview Value

- Data modeling and ontology/knowledge-graph design against a real domain
- **Polyglot systems architecture: Rust for performance/correctness-critical paths, Python for ML orchestration, connected via PyO3 or gRPC**
- **Memory-safe systems programming in Rust (with kernel-C background, a rare and strong systems+ML narrative)**
- Graph databases and relationship-centric query
- Gigapixel image infrastructure: tiling, pyramidal storage, deep-zoom serving
- Foundation-model embedding pipelines and vector search at scale
- Provenance/lineage systems and reproducibility engineering
- Human-in-the-loop ML and annotation systems
- Data governance: de-identification, RBAC, audit, PHI handling
- Working directly with a research lab — turning a real workflow into software

---

## Phase 2 Risks & Mitigations

- **Scope explosion (the main risk)** → sub-phases are hard gates; 2A must fully ship before 2B starts.
- **No lab partner materializes** → 2A + embedding search is still a strong standalone project on public TCGA/CAMELYON data; the ontology layer waits until a real user appears rather than being faked.
- **Governance/compliance complexity** → start with fully public, already-de-identified datasets (TCGA, CAMELYON); only take on PHI once you understand the obligations and have institutional cover.
- **Graph-store learning curve** → begin with graph *queries* over Postgres; adopt Neo4j only if relationship traversal genuinely demands it.
- **It becomes a second thesis and swallows the semester** → it's *supposed* to be long and slow; treat it as a durable background project, and never let it jeopardize coursework or applications.
- **Rust learning curve / premature rewrite** → learn Rust on the isolated, well-bounded WSI tiling pipeline first (pure compute, no web/async), not by rewriting the API. Migrate hot paths only after profiling; never rewrite on principle.

---

**Phase 2 repo (separate from Phase 1):** `pathml-foundry` or `pathos-ontology`
**Phase 2 first action (only after Phase 1 ships):** draft a one-page pitch for an FIU/Miami pathology or bioinformatics lab, and get WSI reading working on a single public TCGA slide.
