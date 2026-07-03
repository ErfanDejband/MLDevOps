# MLOps CI/CD Pipeline — Learning Project

## Purpose of This Document

This README serves **two purposes**:
1. **For the developer:** A step-by-step roadmap to build and understand the full MLOps pipeline incrementally.
2. **For AI agents:** A full picture of the architecture so each agent can develop a specific phase without losing context.

---

## Project Context

### Dataset: MNIST Handwriting Detection
- 60,000 training images, 10,000 test images
- 28×28 grayscale images, **10 classes** (digits 0–9)
- Loaded via `tensorflow.keras.datasets.mnist` — no manual download needed
- Split strategy: **70% train / 20% validation / 10% secret test**

### Two Teams, Two Models

| Team | Model | Folder | Status |
|------|-------|--------|--------|
| **Team 1** | DNN (Dense Neural Network) | `team1_dnn/` | 🔲 Phase 1 |
| **Team 2** | CNN (Convolutional Neural Network) | `team2_cnn/` | 🔲 Later |

Both teams share the same CI/CD pipeline and MLflow server.  
Each team uses a **separate MLflow experiment** so runs never mix.

### Team 1 — Suggested DNN Architecture
```
Input: 784 (28×28 flattened)
  ↓
Dense(512, relu)
  ↓
Dropout(0.2)
  ↓
Dense(256, relu)
  ↓
Dropout(0.2)
  ↓
Dense(10, softmax)   ← probability per digit
```
Metrics to track: `val_accuracy`, `val_loss`, `test_accuracy`

### Team 2 — Suggested CNN Architecture (implement later)
```
Input: (28, 28, 1)
  ↓
Conv2D(32, 3×3, relu) → MaxPooling2D
  ↓
Conv2D(64, 3×3, relu) → MaxPooling2D
  ↓
Flatten → Dense(128, relu) → Dropout(0.5)
  ↓
Dense(10, softmax)
```

### Reference: MLflow_local
`../MLflow_local/` contains the Iris/sklearn experiment used to learn MLflow basics.
All patterns there (3-way split, `infer_signature`, `log_model`, `compare_and_promote`)
apply here — just with TensorFlow instead of sklearn.

The goal of **this folder (CICD_MLops)** is to build a full production-grade MLOps pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  DEVELOPER MACHINE                                              │
│                                                                 │
│  python train.py                                                │
│  → Grid search: logs params + metrics for ALL runs (no artifact)│
│  → After all runs: saves artifact for BEST run only            │
│  → Auto-registers best model to "Staging" in Model Registry    │
│                                                                 │
│  Developer opens MLflow UI (DEV server):                       │
│  → Compares all runs (val_accuracy, loss, dropout...)          │
│  → If happy with auto-best → Staging stays as is               │
│  → If prefers another run → change Staging in UI manually      │
│                                                                 │
│  git push → open Pull Request                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ PR triggers CI
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  CI PIPELINE (GitHub Actions — free)                           │
│                                                                 │
│  Does NOT re-train. Model already lives in MLflow (DEV server) │
│                                                                 │
│  Step 1: Checkout code + install deps                          │
│  Step 2: Restore secret test data (from GitHub secret)         │
│  Step 3: Load Staging model from DEV MLflow Model Registry     │
│  Step 4: Evaluate on SECRET test data (developer never sees)   │
│  Step 5: Check minimum accuracy floor (e.g. 0.90)             │
│  Step 6: Compare vs current Production model                   │
│  Step 7: If Staging wins → promote to Production ✅            │
│          If Staging loses → block PR ❌                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ model promoted to Production
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  CD / DEPLOYMENT (Phase 4 — Render or HuggingFace Spaces)      │
│                                                                 │
│  Loads models:/mnist_classifier/Production from MLflow         │
│  Serves predictions via FastAPI REST endpoint                  │
│  POST /predict → returns predicted digit + confidence          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cross-Team Competition

Both teams use the **same registered model name** `mnist_classifier`.
They compete for the same Production slot:

```
Team 1 trains DNN → registers to Staging → PR → pipeline evaluates → Production if better
Team 2 trains CNN → registers to Staging → PR → pipeline evaluates → Production if better

Winner = highest secret test accuracy, regardless of team or architecture
```

Each team has its own:
- MLflow experiment (runs don't mix)
- CI workflow file (triggers only on their folder changes)

Both teams share:
- `compare_and_promote.py` (quality gate logic)
- `mnist_classifier` registered model name

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

- [x] Train LogisticRegression on Iris dataset (learning exercise)
- [x] Log parameters, metrics, model to local MLflow
- [x] 3-way split: 70% train / 20% validation / 10% test
- [x] Save test data to `test_data.pkl`
- [x] Load best model and evaluate on test set

> This phase was for learning MLflow locally. The real project starts at Phase 1 with MNIST + TensorFlow.

---

### ✅ Phase 1 — Team 1 DNN: Local Training + Remote DEV MLflow (DagsHub)
**Location:** `team1_dnn/`

- [x] Create DagsHub account + repo `mlops-dev-tracking`
- [x] Create `team1_dnn/train.py` — MNIST, 70/20/10 split, DNN, logs to DagsHub DEV
- [x] Create `team1_dnn/requirements.txt`
- [x] Create `CICD_MLops/.env.example`
- [x] Runs visible at `https://dagshub.com/e.dejband/mlops-dev-tracking.mlflow`

> **Note on manual Staging:** Developers do NOT need to manually register or stage models.
> The CI pipeline handles registration and promotion to Production automatically.
> Staging in MLflow UI is only useful for learning what stages mean — not needed in this workflow.

---

### ✅ Phase 2 — GitHub Actions CI Pipeline (Re-train + Log to PROD)
**Location:** `.github/workflows/train_team1_dnn.yml`

- [x] Create second DagsHub repo: `mlops-prod-tracking`
- [x] Create `.github/workflows/train_team1_dnn.yml`
  - Triggers only on changes to `team1_dnn/` files
  - Restores secret test data from GitHub secret
  - Re-trains model, logs to PROD MLflow
  - Calls shared `compare_and_promote.py`
- [x] pip cache keyed to `requirements.txt` — fast on repeated runs
- [ ] Add GitHub Secrets: `MLFLOW_PROD_TRACKING_URI`, `MLFLOW_PROD_USERNAME`, `MLFLOW_PROD_TOKEN`, `SECRET_TEST_DATA_B64`
- [ ] Create `team1_dnn/.env` locally (copy from `.env.example`, fill token)
- [ ] Test: push branch → open PR → verify pipeline runs in GitHub Actions

---

### ✅ Phase 3 — Quality Gate (Compare + Promote)
**Location:** `CICD_MLops/compare_and_promote.py` (shared across all teams)

- [x] Shared script — both Team 1 and Team 2 compete for same `mnist_classifier` Production slot
- [x] Evaluates new model on secret test data (developer never sees this data)
- [x] Hard accuracy floor (default 0.90) — blocks deploy even if "better than nothing"
- [x] Archives old Production version when new one wins
- [x] Logs `secret_test_accuracy` back to run for future comparisons
- [x] Each team's workflow passes `EXPERIMENT_NAME` + `MODEL_ARTIFACT` as env vars

---

### 🔲 Phase 4 — Model Serving API
**Goal:** Wrap the Production model in a FastAPI endpoint. Deploy for free.

**Tasks:**
- [ ] Create `serving/app.py` with FastAPI:
  - `GET /health` → returns status
  - `POST /predict` → accepts 28×28 pixel array (784 values), returns predicted digit + confidence
  - On startup: loads `models:/mnist_classifier/Production` from PROD MLflow
- [ ] Test locally with `uvicorn app:app`
- [ ] Create `serving/Dockerfile`
- [ ] Deploy to **Render** (free tier) or **HuggingFace Spaces**
- [ ] Add redeploy step to workflow — triggers after successful model promotion

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
├── .env.example                     ← template, no real secrets
├── .github/
│   └── workflows/
│       ├── train_dnn.yml            ← CI/CD pipeline for Team 1 (DNN)
│       └── train_cnn.yml            ← CI/CD pipeline for Team 2 (CNN)
├── team1_dnn/
│   ├── train.py                     ← DNN training script (Phase 1)
│   ├── compare_and_promote.py       ← quality gate (Phase 3)
│   ├── secret_test_data.npz         ← held-out test set (NOT committed to git)
│   └── requirements.txt
├── team2_cnn/                       ← Phase later
│   ├── train.py
│   ├── compare_and_promote.py
│   └── requirements.txt
├── serving/
│   ├── app.py                       ← FastAPI serving (Phase 4)
│   └── Dockerfile
└── monitoring/                      ← Phase 5 (future)
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
