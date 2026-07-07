# CICD_MLops

A working MLflow + GitHub Actions pipeline where **two teams train competing
models for the same production slot**, and a shared, automated quality gate —
not a human clicking "promote" — decides which one actually ships.

Team 1 trains a DNN, Team 2 trains a CNN, both on MNIST, both registering
under the same model name, `mnist_classifier`. Only one can hold the
`@champion` alias at a time. This mirrors how real ML platforms run multiple
teams against one production endpoint: the serving layer never needs to know
which architecture is currently winning, and neither team can promote its own
model — a common, versioned script does that, applying the same bar to both.

## Structure

```
CICD_MLops/
├── team1_dnn/     Team 1's model code — theirs to change freely
├── team2_cnn/     Team 2's model code — theirs to change freely
├── shared/        quality_gate.py — one script, used by both teams, never duplicated
└── WIKI/          design docs: ARCHITECTURE.md, plan_to_impliment.md, .env.example
```

## Why it's built this way

- **Two DagsHub MLflow servers, not one.** `mlops-dev-tracking` is where
  developers experiment and register candidates; `mlops-prod-tracking` is the
  production system of record — its `@champion` alias and `quality_gate_evaluations`
  audit trail are written only by CI, never by a developer's laptop. A
  developer physically cannot promote their own model, even by accident.

- **CI evaluates; CD promotes — these are two different jobs with two different
  triggers.** `evaluate` runs on `pull_request` and only reads — it's what
  turns the PR check green or red. `promote` runs on `push` to `main`, i.e.
  strictly after a human has reviewed and merged, and is the only code path
  allowed to write a new champion. A model can't reach production just because
  someone pushed to a branch.

- **One shared quality-gate script, not one per team.** Both workflows call
  the exact same `shared/quality_gate.py` with different parameters
  (`CANDIDATE_ALIAS`, `SECRET_TEST_PATH`, `ARTIFACT_NAME`). If the promotion
  rule changes, it changes once, for both teams — there's no way for Team 1's
  gate and Team 2's gate to quietly drift apart.

- **The quality gate itself is a real gate, not a rubber stamp:** a hard
  accuracy floor (nothing bad ships even as the only candidate), a secret
  held-out test set neither team ever trains or tunes against, and a strict
  "must beat the current champion" comparison — re-checked at merge time, not
  just at PR time, since the champion can move between approval and merge if
  the other team merges first.

- **Alias-based promotion, not file copying.** `@team{n}-candidate` and
  `@champion` are pointers MLflow moves atomically; the full version history
  of every run from every team stays in the registry, so "who's winning and by
  how much" is always inspectable, and a regression can be traced back to a
  specific run.

- **Runnable locally before it ever touches CI.** `quality_gate.py` reads the
  same `.env` a developer already has — there's no separate credential setup
  just to dry-run the exact check CI will perform on your PR.

## Where to go next

- New to a team's folder? Start with that team's own `README.md`
  (`team1_dnn/README.md` or `team2_cnn/README.md`) — setup, training, and how
  to ship a candidate.
- Want the full design reasoning — why two servers, why CI/CD had to split,
  how promotion actually works step by step? See `WIKI/ARCHITECTURE.md` and
  `WIKI/plan_to_impliment.md`.
