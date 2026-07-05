# Implementation Plan — Pattern B (from ARCHITECTURE.md)

This is a plan only — nothing described here is built yet. Review it, then tell me to go
ahead (or correct anything first).

**Scope:** Team 1 (DNN) only, end-to-end. Team 2 (CNN) is intentionally left for a later
pass — it will reuse everything built here unchanged, since the shared quality-gate script
and workflow template don't know or care which architecture produced a candidate.

## Decisions locked in (from your answers)

| Decision | Choice |
|---|---|
| Team 2 scope | Not built in this pass — team1_dnn only |
| Candidate alias naming | Per-team: `@team1-candidate` (later `@team2-candidate`) |
| Hard accuracy floor | 0.90 on the secret test set |
| Server topology | **Two servers**, reusing your existing DagsHub repos: `mlops-dev-tracking` (experimentation + candidate registration) and `mlops-prod-tracking` (the actual production system of record, written only by CI) |
| Promotion trigger | **After merge to `main`**, not on every PR push — see §5 for why this matters |
| CI identity | Reuse your personal DagsHub token (different repo than your local `.env` uses, so it's fine for now) |
| Branch protection | Not enabled for now — single developer, would just add friction |

---

## 1. Why CI and CD had to split (the gap you found)

Originally I had one script triggered on `pull_request` that both evaluated *and*
promoted. That's wrong: it means production could change the moment you push to a PR
branch, before you've reviewed or merged anything. The fix is to split evaluation from
promotion, and give them different triggers:

| | Trigger | Reads | Writes | Purpose |
|---|---|---|---|---|
| **CI — evaluate** | `pull_request` (path-filtered) | Candidate from `mlops-dev-tracking`, current champion from `mlops-prod-tracking` | Nothing (dry run) | Shows pass/fail as the PR's status check — this is what blocks/allows merge |
| **CD — promote** | `push` to `main` (i.e. *after* you approve + merge) | Same candidate + champion | Registers the winning model into `mlops-prod-tracking`, sets `@champion` there | The actual production update |

**Why re-evaluate on merge instead of trusting the PR's result:** the champion baseline
can change between when your PR was approved and when it actually merges — e.g. Team 2's
PR merges first and moves `@champion`. Re-running the comparison at merge time means
you're always compared against the *current* champion, not a stale one from when the PR
was opened. This isn't redundant — it's a legitimate second check with a different
purpose than the pre-merge one.

One consequence worth stating plainly: if the post-merge CD step fails, the **code is
already merged** — CD failing doesn't undo the merge, it just means production doesn't
move. The pre-merge CI check is what's supposed to catch a bad model before merge; the
post-merge check is a safety net for baseline drift between approval and merge, not the
primary gate.

## 2. Why two servers instead of one

You already have both `mlops-dev-tracking` and `mlops-prod-tracking` provisioned, so we
use both rather than inventing a single-server design and leaving one idle:

- **`mlops-dev-tracking`**: hosts `team1_dnn` and `team2_cnn` experiments (every
  hyperparameter-search run), and is where developers register their best run under
  `mnist_classifier` with a `@team{N}-candidate` alias.
- **`mlops-prod-tracking`**: a separate MLflow instance holding the *official* production
  history of `mnist_classifier`. Only the CD step ever writes here. The `@champion` alias
  lives on this server, not on dev-tracking. This is a physically separate system of
  record — closer to the real-world "compliance-driven separate environment" pattern
  than an alias flip on a shared server would be.
- A third logical space, `quality_gate_evaluations`, records every evaluation (from both
  the PR-time and merge-time runs) — living on `mlops-prod-tracking` alongside the
  registry it's judging, so the audit trail and the thing it's auditing sit together.

**Open question this raises (see §8):** you mentioned both existing repos use the same
DagsHub user/token. That means a truly separate "CI service identity" isn't available
today without provisioning a second DagsHub account. For now the plan assumes CI uses
your existing token but against two different tracking URIs — see §8 for the tradeoff.

## 3. Prerequisites — things only you can do

- [ ] **Add GitHub Actions secrets** (repo → Settings → Secrets and variables → Actions):
  - `MLFLOW_DEV_TRACKING_URI` — `https://dagshub.com/e.dejband/mlops-dev-tracking.mlflow`
  - `MLFLOW_PROD_TRACKING_URI` — `https://dagshub.com/e.dejband/mlops-prod-tracking.mlflow`
  - `MLFLOW_CI_USERNAME` / `MLFLOW_CI_PASSWORD` — credentials CI uses against *both*
    servers (see §8 for whether this is your personal token or a future dedicated one)
  - `SECRET_TEST_DATA_B64` — base64-encoded secret test set (see §5)
- [ ] Confirm both repo URIs above are correct/current.

## 4. Config changes: `.env.example` and `team1_dnn/.env`

- `.env.example` gets rewritten to document exactly the four GitHub secrets above, plus
  the **local dev** variables a developer actually needs
  (`MLFLOW_DEV_TRACKING_URI`/`MLFLOW_DEV_USERNAME`/`MLFLOW_DEV_PASSWORD` — dev-tracking
  only; developers never need prod-tracking credentials locally).
- Your local `team1_dnn/.env` needs updating to match — I'll flag exactly what changes
  when we get there, without reading/printing its current secret contents.
- `train.py`'s `setup_mlflow()` currently reads generic `MLFLOW_TRACKING_URI/USERNAME/
  PASSWORD` — needs renaming to the `MLFLOW_DEV_*` names so it's unambiguous it only
  ever talks to dev-tracking.

## 5. `team1_dnn/train.py` — restore training logic

Currently only `load_data()` + `setup_mlflow()` remain. Plan to rebuild it as:

1. **Fix `setup_mlflow()`**: read `MLFLOW_DEV_*` env vars (see §4); rename experiment
   from `team1_dnn_mnist_Experiment3` to `team1_dnn`.
2. **Uncomment + wire up secret-test-data export**: `load_data()` already computes the
   held-out 10% (`X_test`/`y_test`) — restore `np.savez(...)` so a developer can generate
   `secret_test_data.npz` locally, then base64-encode it once for the
   `SECRET_TEST_DATA_B64` GitHub secret. Stays out of git (already `.gitignore`d).
3. **`build_model(dropout_rate)`**: same DNN shape as before (Dense512 → Dropout →
   Dense256 → Dropout → Dense10 softmax) — restoring, not redesigning.
4. **`train(...)`**: fits one model for one hyperparameter combination, returns
   `(model, val_accuracy, val_loss)`.
5. **Grid search loop in `__main__`**: iterate the hyperparameter grid, log
   params+metrics only per run (no artifact), track the best run in memory.
6. **After the loop — log the winner only**: re-open the best run, log the model +
   `infer_signature` + input example, `registered_model_name="mnist_classifier"`
   (registered on **dev-tracking**).
7. **Set the alias**: `client.set_registered_model_alias("mnist_classifier",
   "team1-candidate", version)` — on dev-tracking.
8. Print a clear next-step message (git push → open PR).

## 6. Shared quality-gate script (new): `CICD_MLops/shared/quality_gate.py`

One script, shared by both teams' workflows, with a `--mode {evaluate,promote}` flag (or
`GATE_MODE` env var) controlling behavior:

**Common to both modes:**
1. Connect to `mlops-dev-tracking` (read candidate) and `mlops-prod-tracking` (read
   current champion) using the CI credentials.
2. Load the candidate via `models:/mnist_classifier@{CANDIDATE_ALIAS}` from dev-tracking.
3. Load/decode the secret test set.
4. Evaluate candidate accuracy on the secret test set.
5. **Hard floor check**: accuracy < 0.90 → fail immediately.
6. **Champion comparison**: resolve `models:/mnist_classifier@champion` from
   prod-tracking.
   - No champion yet (bootstrap case) → candidate passes on floor alone.
   - Champion exists → candidate must strictly beat champion's accuracy (re-evaluated
     fresh, not read from old logged metrics).
7. Log the evaluation to a `quality_gate_evaluations` run on **prod-tracking**: mode,
   candidate version, candidate accuracy, champion accuracy (if any), pass/fail.

**`evaluate` mode** (called from the `pull_request` trigger): exit non-zero on failure so
the PR check goes red; exit zero on pass. **No writes to the registry either way** — this
mode only ever reports.

**`promote` mode** (called from the `push`-to-`main` trigger): does everything above,
and additionally, **only on pass**, actually copies the model into prod-tracking (a
registry version is tied to the server it was logged on, so this isn't just flipping a
pointer):

1. Reads the candidate's original run from dev-tracking via
   `client.get_run(candidate.run_id)` — its params (`epochs`, `batch_size`,
   `dropout_rate`, ...) and training metrics (`val_accuracy`, `val_loss`).
2. Switches the MLflow client to `mlops-prod-tracking` and starts a **new run** there.
3. Logs those same params + training metrics into the new run, plus this evaluation's
   `secret_test_accuracy` — so prod-tracking is self-contained and doesn't require
   cross-referencing dev-tracking to see why a version was promoted.
4. Logs the model artifact + signature into that new run
   (`mlflow.tensorflow.log_model(...)`), registers it under `mnist_classifier` on
   prod-tracking, and sets `@champion` to the new version.

On fail: logs the attempt (mode, scores, verdict) to `quality_gate_evaluations` on
prod-tracking, exits non-zero (surfaces in the Actions run even though nothing blocks a
merge that already happened), no registry changes.

## 7. GitHub Actions workflow (new): `.github/workflows/team1_dnn_quality_gate.yml`

One workflow file, two jobs, both path-filtered to `CICD_MLops/team1_dnn/**`:

- **`evaluate` job** — trigger: `pull_request`.
  1. Checkout, setup Python, cache/install deps.
  2. Decode `SECRET_TEST_DATA_B64` → `secret_test_data.npz` (discarded after the job).
  3. Placeholder check (trivial, always-passing step for now — e.g. an import
     sanity-check) — not investing in real unit tests yet.
  4. `python CICD_MLops/shared/quality_gate.py --mode evaluate` with env vars:
     `MLFLOW_DEV_TRACKING_URI`, `MLFLOW_PROD_TRACKING_URI`, `MLFLOW_CI_USERNAME`,
     `MLFLOW_CI_PASSWORD`, `CANDIDATE_ALIAS=team1-candidate`,
     `REGISTERED_MODEL_NAME=mnist_classifier`, `SECRET_TEST_PATH=...`,
     `ACCURACY_FLOOR=0.90`.
  5. Non-zero exit → PR check shows red, merge blocked (assuming branch protection
     requires this check — worth turning on if not already).

- **`promote` job** — trigger: `push`, `branches: [main]`.
  1. Same setup steps as above.
  2. `python CICD_MLops/shared/quality_gate.py --mode promote` with the same env vars.
  3. Non-zero exit just fails the Actions run for visibility — nothing to "block" since
     the merge already happened.

## 8. Bootstrap note (first run ever)

The first time this ever runs, there's no existing `@champion` on prod-tracking — the
"no champion yet" branch (§6) handles it by promoting on floor-clearance alone. Worth
testing deliberately once since it's an edge case that only happens once.

## 9. Verification checklist (before calling this done)

- [ ] `train.py` run locally produces a `@team1-candidate` alias visible on
      **dev-tracking**'s MLflow UI, with params+metrics for every grid combination and an
      artifact only on the winning run.
- [ ] `secret_test_data.npz` generated once, base64-encoded, stored as
      `SECRET_TEST_DATA_B64`.
- [ ] Opening a PR against `CICD_MLops/team1_dnn/**` triggers only the `evaluate` job (not
      `promote`), and a PR against `docs/` triggers neither.
- [ ] `evaluate` job passing/failing correctly reflects candidate quality, and **does not
      change anything on prod-tracking** either way (check the registry before/after).
- [ ] Merging that PR to `main` triggers the `promote` job.
- [ ] First-ever merge promotes to `@champion` on prod-tracking (bootstrap case).
- [ ] A deliberately worse second candidate: `evaluate` job fails on its PR; if
      hypothetically merged anyway, `promote` job also fails and does not move
      `@champion`.
- [ ] A deliberately better second candidate: both jobs pass, `@champion` moves on
      prod-tracking, and the old version is still visible in registry history (not
      deleted).
- [ ] `quality_gate_evaluations` on prod-tracking shows one run per evaluation (both PR
      and merge-time runs), human-readable enough to audit later.

## 10. Resolved

- **CI identity**: reusing your personal DagsHub token is fine — it's a different repo
  (`mlops-prod-tracking`) than your local `.env` points at (`mlops-dev-tracking`), so
  there's a real boundary even without a second account. Revisit true service-account
  separation only if this ever stops being a single-developer project.
- **Branch protection**: skipped for now — you're the only developer, so a required
  status check would only add friction, not safety, at this stage.

---

Nothing left open. Once you say go, I'll implement in this order: `.env.example` →
`team1_dnn/train.py` → `CICD_MLops/shared/quality_gate.py` → the workflow file → walk
through the verification checklist together.
