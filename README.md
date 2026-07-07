# Erfan — MLDevOps

Two things live in this repo, in increasing order of realism:

| | What it is | What it teaches |
|---|---|---|
| [`MLflow_local/`](MLflow_local/) | MLflow running entirely on your machine — SQLite backend, local artifact store | The building blocks: experiments, runs, params/metrics, model registry, `infer_signature` |
| [`CICD_MLops/`](CICD_MLops/) | Two teams training competing MNIST models, tracked on hosted MLflow (DagsHub), shipped through GitHub Actions | The real thing: CI/CD around ML, promotion governance, multi-team competition for one production slot |

Start with `MLflow_local` if MLflow itself is new to you. Move to `CICD_MLops`
for the actual CI/CD project — that's where the interesting design decisions
live.

---

## `MLflow_local` — the playground

A local-only MLflow setup: no server to stand up, no cloud account, no CI.
Everything writes to a SQLite file (`mlflow.db`) and a local `mlartifacts/`
folder in `Local_setup/`. It exists to get the core MLflow vocabulary under
your fingers — experiment, run, param, metric, artifact, registered model,
model version — before any of that gets wired into a pipeline.

**Setup:**

```bash
cd MLflow_local
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cd Local_setup
python run_log_model.py       # trains + logs a run
python load_model.py          # loads it back and predicts
mlflow ui                     # browse http://127.0.0.1:5000
```

See `MLflow_local/README.md` for the full MLflow reference (every function
used, why, with examples) — that's the deep-dive doc for this half of the
repo.

---

## `CICD_MLops` — the real situation

Two teams (DNN and CNN) train competing models on MNIST, both targeting the
same registered model name, `mnist_classifier`, on hosted MLflow (DagsHub).
An automated quality gate — never a person — decides which one holds the
`@champion` alias, wired up end-to-end through GitHub Actions.

**Why this is the "real" half, not just a bigger toy:**

- **Two MLflow servers, split by role** — `mlops-dev-tracking` for
  experimentation, `mlops-prod-tracking` as the production system of record.
  Developers physically cannot write a new champion; only CI can.
- **CI and CD are separate jobs with separate triggers** — `evaluate` runs on
  every PR (read-only, gates the merge); `promote` runs only after a merge to
  `main`. A model can't reach production just because a branch got pushed.
- **One shared quality-gate script for both teams** — the promotion rule
  (accuracy floor + must-beat-current-champion, checked against a secret
  held-out test set) lives in one place, so it can't quietly diverge between
  teams and the competition stays fair.
- **Fully runnable locally before anything touches CI** — the same script CI
  runs can be dry-run against your own `.env`, so you see PASS/FAIL before you
  even open a PR.

**Setup (per team, same for both):**

```bash
cd CICD_MLops/team1_dnn        # or team2_cnn
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp ../WIKI/.env.example .env   # fill in your mlops-dev-tracking DagsHub credentials
python train.py                # trains, registers the best run, sets @team{n}-candidate
```

Then, from `CICD_MLops/shared`, `python quality_gate.py --mode evaluate` dry-runs
the exact check your PR will get. Full walkthrough, including how to generate
and register the secret test-data GitHub secret, is in each team's own
`README.md`.

For the design reasoning behind all of this — why two servers, why CI/CD had
to split, exactly how a candidate becomes champion — see
`CICD_MLops/README.md` and `CICD_MLops/WIKI/ARCHITECTURE.md`.
