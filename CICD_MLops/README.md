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

- [x] Train LogisticRegression on Iris dataset (learning exercise)
- [x] Log parameters, metrics, model to local MLflow
- [x] 3-way split: 70% train / 20% validation / 10% test
- [x] Save test data to `test_data.pkl`
- [x] Load best model and evaluate on test set

> This phase was for learning MLflow locally. The real project starts at Phase 1 with MNIST + TensorFlow.

---

### 🔲 Phase 1 — Team 1 DNN: Local Training + Remote MLflow (DagsHub)
**Location:** `team1_dnn/`  
**Goal:** Build the DNN training script for MNIST and log runs to DagsHub (remote MLflow).

**One-time setup (manual, you do this):**
- [ ] Create a [DagsHub](https://dagshub.com) account
- [ ] Create a new repo on DagsHub called `mlops-dev-tracking`
- [ ] Go to repo → **Remote** tab → **MLflow** → copy the tracking URI + token
- [ ] Run `pip install dagshub tensorflow mlflow`

**Tasks:**
- [ ] Create `team1_dnn/train.py`:
  - Load MNIST via `keras.datasets.mnist`
  - Apply 70/20/10 split
  - Save 10% test split to `team1_dnn/secret_test_data.npz` (for CI pipeline later)
  - Build DNN model (architecture above)
  - Train with different hyperparameters (epochs, learning rate, dropout)
  - Log params + metrics + model to DagsHub MLflow
  - Use `mlflow.tensorflow.log_model(...)` with `name=` (not deprecated `artifact_path`)
- [ ] Create `.env.example` showing required env vars
- [ ] Create `team1_dnn/requirements.txt`
- [ ] Verify: runs visible at `https://dagshub.com/<username>/mlops-dev-tracking.mlflow`

**Key env vars needed (store in `.env`, never commit):**
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/mlops-dev-tracking.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub_username>
MLFLOW_TRACKING_PASSWORD=<dagshub_token>
```

**Simplest DagsHub connection in code:**
```python
import dagshub
import mlflow

dagshub.init(repo_owner="<username>", repo_name="mlops-dev-tracking", mlflow=True)
mlflow.set_experiment("team1_dnn_mnist")
# Now mlflow.log_* goes to DagsHub automatically
```

**Expected result:** Training runs visible on DagsHub MLflow UI, model saved as artifact

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
  - `POST /predict` → accepts 28×28 pixel array (784 values), returns predicted digit + confidence
  - On startup: loads `models:/mnist_dnn/Production` from PROD MLflow
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
