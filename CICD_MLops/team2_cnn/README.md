# Team 2 — CNN

Convolutional entry in the `mnist_classifier` competition. Architecture:
`Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout
→ Dense(10, softmax)` on 28×28×1 MNIST images (not flattened, unlike Team 1).

This folder is yours to change freely (architecture, hyperparameters,
`train.py` internals). Everything that must stay identical across teams —
the quality gate, the promotion rule, the workflow shape — lives in
`CICD_MLops/shared/` and `.github/workflows/`, not here.

## 1. One-time setup

```bash
cd CICD_MLops/team2_cnn
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy the template and fill in your **dev-tracking** DagsHub credentials:

```bash
cp ../WIKI/.env.example .env
```

Edit `.env`:

```
MLFLOW_DEV_TRACKING_URI=https://dagshub.com/<you>/mlops-dev-tracking.mlflow
MLFLOW_DEV_USERNAME=<you>
MLFLOW_DEV_PASSWORD=<your DagsHub token>
```

You only ever need **dev-tracking** credentials locally. Prod-tracking is
written to only by CI, after a merge.

## 2. Train

```bash
python train.py
```

This runs the full hyperparameter grid (edit `param_grid` in `train.py` to
change it), logs params+metrics for every combination, then — for the single
best run by validation accuracy — logs the model artifact and registers it
under the shared model name `mnist_classifier`, moving the alias
`@team2-candidate` to point at it. Team 1's `@team1-candidate` alias is
completely separate; you can't clobber each other.

Re-running `train.py` again later just moves `@team2-candidate` to whichever
run wins that time — there is never more than one live Team 2 candidate.

Check the run in the MLflow UI on your `mlops-dev-tracking` DagsHub repo.

## 3. Generate the secret test set (once)

`train.py` already writes `secret_test_data.npz` (a 10% held-out split you
never train or tune against) as a side effect of `load_data()`. Base64-encode
it once and store it as the `SECRET_TEST_DATA_B64_TEAM2` GitHub Actions
secret — this is what CI evaluates every candidate against. It's a **separate
secret from Team 1's** because the CNN's `X_test` is shaped `(N, 28, 28, 1)`
while the DNN's is flattened to `(N, 784)` — they aren't interchangeable
files.

```bash
# Windows PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("secret_test_data.npz")) | Set-Clipboard
```

Paste the clipboard contents as the secret's value in
**GitHub repo → Settings → Secrets and variables → Actions**. Never commit
the `.npz` file itself (already `.gitignore`d).

## 4. Check the quality gate locally, before pushing

```bash
cd ../shared
python quality_gate.py --mode evaluate --team team2_cnn
```

`--team` only matters locally — it's how one shared script finds *your*
`.env` and secret test data instead of Team 1's (its default). This reuses
your `.env` credentials (no separate CI identity needed locally), pulls your
`@team2-candidate` from dev-tracking, compares it against the current
`@champion` on prod-tracking, and prints PASS/FAIL — the exact same check CI
runs on your PR.

## 5. Ship it

```bash
git add .
git commit -m "..."
git push
# open a PR
```

- **On the PR** — the `evaluate` job runs automatically (read-only, no
  promotion) and becomes the PR's status check.
- **On merge to `main`** — the `promote` job re-runs the same comparison and,
  only on pass, registers the winning version into `mlops-prod-tracking` and
  moves `@champion` there.

See `CICD_MLops/WIKI/ARCHITECTURE.md` and `plan_to_impliment.md` for the full
design reasoning (why two servers, why CI and CD are separate jobs, how the
quality gate decides pass/fail).
