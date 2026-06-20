# MLDevOps Learning Roadmap

> Platform: **GitHub** (repo, Actions, Container Registry)
> Focus: Learning MLOps as if working in a team at a large company
> Budget: **$0** — All tools are free/open-source

### Tools & Cost

| Tool | Cost | Notes |
|------|------|-------|
| MLflow | Free (open-source) | Tracking, Registry, Serving |
| Docker Desktop | Free (personal use) | Containers locally |
| PostgreSQL | Free (in Docker) | MLflow backend store |
| GitHub | Free (public repos) | Unlimited Actions minutes for public repos |
| GitHub Actions | Free (public repos) | CI/CD pipelines |
| DVC | Free (open-source) | Data versioning (use local/free remote) |
| FastAPI | Free (open-source) | Model serving |
| Python + scikit-learn | Free | ML framework |

---

## Phase 1: MLflow — Experiment Tracking & Model Registry (Current Focus)

### 1.1 Understanding MLflow Components

MLflow has 4 main components:

| Component | What It Does | Team Use Case |
|-----------|-------------|---------------|
| **Tracking** | Logs parameters, metrics, artifacts per run | Everyone sees experiment results |
| **Projects** | Packages ML code for reproducibility | Anyone can re-run your experiment |
| **Models** | Standard format for packaging models | Deploy anywhere (REST, batch, edge) |
| **Model Registry** | Version & stage models (Staging → Production) | Team approves model before deploy |

### 1.2 Local vs. Remote MLflow (Why It Matters)

```
LOCAL (what you have now):
  Your PC → mlflow.db (SQLite) + mlartifacts/ (local folder)
  ❌ Only you can see it
  ❌ Lost if your machine dies

REMOTE / TEAM SETUP:
  Everyone → MLflow Server → PostgreSQL + Cloud Storage (S3/Azure Blob)
  ✅ Everyone sees all experiments
  ✅ Durable, backed up
  ✅ Model Registry is shared
```

### 1.3 Setting Up Remote MLflow with Docker

We'll run MLflow server locally in Docker (simulates a team server):

```
Architecture (all running locally in Docker — FREE):
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Your Python  │────▶│  MLflow Server   │────▶│   PostgreSQL    │
│   Scripts    │     │  (Docker :5000)  │     │  (Docker :5432) │
└──────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Artifact Store  │
                     │ (Docker volume)  │
                     │  (local — free)  │
                     └──────────────────┘
```

> 💡 In a real company this would use cloud storage (S3/Azure Blob),
> but Docker volumes simulate the same architecture for free.

**Files to create:**
- `infrastructure/docker-compose.yml` — defines MLflow + PostgreSQL services
- `infrastructure/Dockerfile.mlflow` — custom MLflow image

### 1.4 Experiment Tracking Best Practices (Enterprise)

| Practice | Why |
|----------|-----|
| Use `mlflow.set_experiment("project/task")` | Organize experiments by project |
| Log ALL hyperparameters | Reproducibility |
| Log dataset version/hash | Know which data produced which model |
| Use tags for metadata | `mlflow.set_tag("developer", "erfan")` |
| Register good models in Model Registry | Team can review & approve |
| Use model stages: `None → Staging → Production` | Controlled rollout |
| Never commit artifacts to git | Use remote storage |

### 1.5 MLflow Model Registry Workflow (Team Style)

```
1. Data Scientist trains model → logs to MLflow
2. Good model? → Register in Model Registry (version 1, 2, 3...)
3. Team reviews metrics → Transitions model to "Staging"
4. Validation passes → Transitions to "Production"
5. CI/CD picks up "Production" model → Deploys automatically
```

### 1.6 Learning Tasks for Phase 1

- [ ] Set up Docker-based MLflow server (PostgreSQL backend)
- [ ] Refactor training script to use remote tracking URI
- [ ] Log parameters, metrics, and model artifacts properly
- [ ] Register a model in the Model Registry
- [ ] Transition model through stages (None → Staging → Production)
- [ ] Compare multiple experiment runs in MLflow UI
- [ ] Create a script that loads the "Production" model and serves predictions

---

## Phase 2: Data & Code Versioning

- **DVC** — version large datasets without git (stored in cloud)
- **Git branching** — feature branches for experiments, PRs for reviews
- Reproducible pipelines with `dvc.yaml`

---

## Phase 3: CI/CD with GitHub Actions

- Auto-train model on PR
- Run model validation tests (accuracy thresholds)
- Auto-register model if tests pass
- Deploy model on merge to main

---

## Phase 4: Model Deployment

- Serve model via **FastAPI** + **Docker** (free, runs locally)
- Deploy to cloud using **GitHub Actions** + free-tier options (e.g., Render, Railway, or just Docker locally)
- Blue/green deployment strategy

---

## Phase 5: Monitoring & Retraining

- Monitor data drift & prediction drift
- Set up alerts when model degrades
- Automated retraining triggers

---

## Project Structure (Target)

```
MLDevOps/
├── src/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Model evaluation
│   ├── preprocess.py         # Data preprocessing
│   └── serve.py              # Model serving (FastAPI)
├── tests/                    # Unit & integration tests
├── infrastructure/
│   ├── docker-compose.yml    # MLflow + DB services
│   └── Dockerfile.mlflow     # MLflow server image
├── pipelines/                # GitHub Actions workflows
├── configs/
│   └── params.yaml           # Hyperparameters
├── data/                     # DVC-tracked (Phase 2)
├── docs/
│   └── learning-roadmap.md   # This file
├── requirements.txt
├── .gitignore
└── README.md
```
