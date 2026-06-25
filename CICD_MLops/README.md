# MLOps CI/CD Pipeline — Learning Project

## Purpose of This Document

This README serves **two purposes**:
1. **For the developer (Erfan):** A step-by-step roadmap to build and understand the full MLOps pipeline incrementally.
2. **For AI agents:** A full picture of the architecture so each agent can develop a specific phase without losing context.

---

## Project Context

The model being used is a **LogisticRegression on the Iris dataset**, already developed in `../MLflow_local/`. That folder contains:
- `run_log_model.py` — trains models, logs to MLflow, saves test data to `test_data.pkl`
- `load_model.py` — loads best model from MLflow, evaluates on hold-out test set
- `MLflow_Guide.md` — MLflow API reference

The goal of **this folder (CICD_MLops)** is to take that local work and wire it into a real CI/CD pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  DEVELOPER MACHINE                                              │
│                                                                 │
│  Runs experiments freely                                        │
│  → All runs logged to DEV MLflow Server (DagsHub)              │
│  → Developer compares runs in MLflow UI                        │
│  → Picks best model, promotes to "Staging" in Model Registry   │
│  → Pushes training CODE (not model) to GitHub                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ git push
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  CI PIPELINE (GitHub Actions — free)                           │
│                                                                 │
│  Triggered on: push to main / pull request                     │
│                                                                 │
│  Step 1: Checkout code                                         │
│  Step 2: Install dependencies                                  │
│  Step 3: Re-train model (same script as developer used)        │
│          → Logs to PROD MLflow Server (separate DagsHub repo)  │
│  Step 4: Evaluate on SECRET test data (stored in GitHub        │
│          Actions secret / separate secure location)            │
│  Step 5: Compare new model accuracy vs current Production      │
│  Step 6: If new model wins → promote to "Production"           │
│          in PROD MLflow Model Registry                         │
│          Else → pipeline FAILS, blocks merge                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ model promoted to Production
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  CD / DEPLOYMENT (Render or HuggingFace Spaces — free)         │
│                                                                 │
│  Pulls model from PROD MLflow Model Registry                   │
│  Serves predictions via FastAPI REST endpoint                  │
│  /predict → returns class label + probability                  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  MONITORING (future phase)                                      │
│                                                                 │
│  Track prediction drift over time                              │
│  Alert if accuracy degrades                                    │
│  Trigger retraining pipeline automatically                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Two-Server Strategy

| | DEV MLflow Server | PROD MLflow Server |
|---|---|---|
| **Who writes** | Developers freely | Only CI/CD pipeline |
| **Who reads** | All team members | Developers (read-only) |
| **Purpose** | Experimentation, comparison | Official, auditable runs |
| **Cleanup** | Periodically cleaned | Kept permanently |
| **Hosted on** | DagsHub (free) | DagsHub (separate repo, free) |
| **Access** | Dev credentials | CI/CD secrets only |

---

## Secret Test Data Strategy

```
Developer sees:
  → train data (70%)
  → validation data (20%)   ← used to pick best model

Developer NEVER sees:
  → secret_test.csv (10%)   ← stored as GitHub Actions secret
                               or in a private repo

Pipeline uses secret test data to:
  1. Verify model generalizes (Quality Gate)
  2. Compare against current Production model
  3. Block deployment if score drops
```

This prevents developers from overfitting to the test set (even unintentionally).

---

## Free Services Used

| Purpose | Service | Why Free |
|---------|---------|---------|
| Code repository | **GitHub** | Free for public/private repos |
| CI/CD pipeline | **GitHub Actions** | 2000 min/month free |
| DEV MLflow server | **DagsHub** | Free MLflow + DVC hosting |
| PROD MLflow server | **DagsHub** | Separate repo, also free |
| Model serving API | **Render** or **HuggingFace Spaces** | Free tier available |
| Artifact storage | **DagsHub built-in** | Included with DagsHub |

---

## Development Phases (Step-by-Step Roadmap)

Each phase builds on the previous. Develop one phase at a time.

---

### ✅ Phase 0 — Local MLflow (DONE)
**Location:** `../MLflow_local/`

- [x] Train LogisticRegression on Iris dataset
- [x] Log parameters, metrics, model to local MLflow
- [x] 3-way split: 70% train / 20% validation / 10% test
- [x] Save test data to `test_data.pkl`
- [x] Load best model and evaluate on test set

---

### 🔲 Phase 1 — Connect to Remote DEV MLflow Server (DagsHub)
**Goal:** Replace `http://127.0.0.1:5000` with a real remote server that persists.

**Tasks:**
- [ ] Create a DagsHub account and a repo called `mlops-dev-tracking`
- [ ] Get DagsHub MLflow tracking URI and credentials
- [ ] Store credentials as environment variables (never hardcode)
- [ ] Update `run_log_model.py` to use DagsHub URI
- [ ] Run training and verify runs appear in DagsHub MLflow UI
- [ ] Run `load_model.py` and verify it loads from remote

**Key files to create:**
- `phase1_remote_tracking/run_log_model_remote.py` — copy of run_log_model.py with remote URI
- `.env.example` — template showing required env vars (no real values)

**Expected result:** Runs visible at `https://dagshub.com/<username>/mlops-dev-tracking.mlflow`

---

### 🔲 Phase 2 — GitHub Actions CI Pipeline (Re-train + Log)
**Goal:** Every push to `main` triggers training and logs a new run to PROD MLflow server.

**Tasks:**
- [ ] Create second DagsHub repo: `mlops-prod-tracking`
- [ ] Store PROD MLflow credentials as GitHub Actions secrets
- [ ] Store secret test data as GitHub Actions secret (base64 encoded CSV)
- [ ] Create `.github/workflows/train.yml`
  - Checkout code
  - Install dependencies from `requirements.txt`
  - Run training script
  - Log to PROD MLflow server
- [ ] Verify pipeline runs on push and run appears in PROD MLflow UI

**Key files to create:**
- `.github/workflows/train.yml`
- `requirements.txt`
- `phase2_ci_pipeline/train_pipeline.py` — training script adapted for CI

**Expected result:** Push to GitHub → GitHub Actions runs → new MLflow run appears in PROD server

---

### 🔲 Phase 3 — Quality Gate (Compare + Promote)
**Goal:** CI pipeline compares new model against current Production. Only promotes if better.

**Tasks:**
- [ ] Create `compare_and_promote.py` script:
  - Load new run metrics from PROD MLflow
  - Load current Production model metrics from Model Registry
  - If new accuracy > production accuracy → promote to Production
  - Else → exit with error code (blocks CI)
- [ ] Add secret test evaluation step (evaluate on secret test data before comparing)
- [ ] Add `compare_and_promote.py` as a step in `train.yml`
- [ ] Test: manually make a worse model, verify pipeline blocks it
- [ ] Test: make a better model, verify it gets promoted

**Key files to create:**
- `phase3_quality_gate/compare_and_promote.py`
- Update `.github/workflows/train.yml`

**Expected result:** Only better models make it to Production in MLflow Registry

---

### 🔲 Phase 4 — Model Serving API
**Goal:** Wrap the Production model in a FastAPI endpoint. Deploy for free.

**Tasks:**
- [ ] Create `app.py` with FastAPI:
  - `GET /health` → returns status
  - `POST /predict` → accepts 4 Iris features, returns predicted class + probability
  - On startup: loads `models:/iris_logistic_regression_model/Production` from PROD MLflow
- [ ] Test locally with `uvicorn app:app`
- [ ] Create `Dockerfile` for containerized deployment
- [ ] Deploy to **Render** (free tier) or **HuggingFace Spaces**
- [ ] Add deployment step to `train.yml` — redeploy after model promotion

**Key files to create:**
- `phase4_serving/app.py`
- `phase4_serving/Dockerfile`
- Update `.github/workflows/train.yml`

**Expected result:** Live public endpoint that returns Iris predictions

---

### 🔲 Phase 5 — Monitoring (Future)
**Goal:** Detect when the deployed model starts performing poorly.

**Tasks:**
- [ ] Log each prediction (input + output) to a database or file
- [ ] Periodically compare recent predictions to known labels (if available)
- [ ] Alert (GitHub Issue / email) if accuracy drops below threshold
- [ ] Optionally: auto-trigger retraining pipeline on drift detection

**Note:** This phase is exploratory. Tools to consider: Evidently AI (free), Grafana (free OSS).

---

## Key Concepts Reference

### Model Lifecycle in MLflow Registry
```
None → Staging → Production → Archived
```
- **None:** Just registered, not reviewed
- **Staging:** Candidate, passed dev review
- **Production:** Live, serving traffic
- **Archived:** Replaced by newer version

### What Lives Where
```
GitHub:       Training code, pipeline config, API code
DagsHub DEV:  Developer experiment runs, dev model versions
DagsHub PROD: Official CI runs, production model versions
Render:       Running FastAPI server (loads model from DagsHub PROD)
```

### Environment Variables Needed
```bash
# DEV MLflow
MLFLOW_DEV_TRACKING_URI=https://dagshub.com/<user>/mlops-dev-tracking.mlflow
MLFLOW_DEV_USERNAME=<dagshub_username>
MLFLOW_DEV_PASSWORD=<dagshub_token>

# PROD MLflow (stored as GitHub Actions secrets, never locally)
MLFLOW_PROD_TRACKING_URI=https://dagshub.com/<user>/mlops-prod-tracking.mlflow
MLFLOW_PROD_USERNAME=<dagshub_username>
MLFLOW_PROD_PASSWORD=<dagshub_token>

# Secret test data (stored as GitHub Actions secret)
SECRET_TEST_DATA_B64=<base64 encoded CSV>
```

---

## Folder Structure (Final State)

```
CICD_MLops/
├── README.md                        ← this file
├── requirements.txt
├── .env.example                     ← template, no real secrets
├── .github/
│   └── workflows/
│       └── train.yml                ← CI/CD pipeline definition
├── phase1_remote_tracking/
│   └── run_log_model_remote.py
├── phase2_ci_pipeline/
│   └── train_pipeline.py
├── phase3_quality_gate/
│   └── compare_and_promote.py
├── phase4_serving/
│   ├── app.py
│   └── Dockerfile
└── phase5_monitoring/               ← future
```

---

## How to Use This README as an Agent

When asking an AI agent to implement a phase:

> "I am working on Phase X of my MLOps pipeline. The full architecture is described in `CICD_MLops/README.md`. Please implement Phase X only. The existing local code is in `../MLflow_local/`. Ask me before making assumptions about credentials or service choices."

Each phase is self-contained. An agent should:
1. Read this README fully first
2. Check the relevant `phase_X/` folder for existing files
3. Implement only the tasks listed under that phase
4. Not modify files from other phases unless explicitly asked
