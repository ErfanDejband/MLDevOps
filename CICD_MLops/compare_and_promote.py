"""
Shared Quality Gate — used by ALL teams' CI pipelines.

IMPORTANT: The pipeline does NOT re-train the model.
The developer trains locally (many combinations), picks the best run,
tags it with approved=true, then pushes to trigger this pipeline.

This script:
  1. Finds the run tagged approved=true in this team's experiment
  2. Evaluates it on secret test data
  3. Compares against current Production model
  4. If better → promotes to Production
  5. If worse  → blocks the PR

Both Team 1 (DNN) and Team 2 (CNN) call this same script.
They compete for the SAME Production slot: registered model "mnist_classifier".
"""
import sys
import os
import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.client import MlflowClient

# ── Config from env vars (set by each team's CI workflow) ────────────────────
EXPERIMENT_NAME  = os.environ.get("EXPERIMENT_NAME")
MODEL_ARTIFACT   = os.environ.get("MODEL_ARTIFACT")
REGISTERED_NAME  = os.environ.get("REGISTERED_NAME", "mnist_classifier")
SECRET_TEST_PATH = os.environ.get("SECRET_TEST_PATH", "secret_test_data.npz")
MIN_ACCURACY     = float(os.environ.get("MIN_ACCURACY", "0.90"))

if not EXPERIMENT_NAME or not MODEL_ARTIFACT:
    print("❌ Missing required env vars: EXPERIMENT_NAME, MODEL_ARTIFACT")
    sys.exit(1)

client = MlflowClient()

print(f"🔍 Quality Gate")
print(f"   Team experiment : {EXPERIMENT_NAME}")
print(f"   Competing for   : '{REGISTERED_NAME}' Production slot")
print(f"   Minimum accuracy: {MIN_ACCURACY}")
print()

# ── 1. Load secret test data ─────────────────────────────────────────────────
print("Loading secret test data...")
data   = np.load(SECRET_TEST_PATH)
X_test = data["X_test"].astype("float32")
y_test = data["y_test"]
print(f"  {X_test.shape[0]} samples, classes: {sorted(set(y_test.tolist()))}")

# ── 2. Find the developer-approved run ───────────────────────────────────────
# Developer tagged the best run with approved=true before pushing
print(f"\nSearching for approved run in '{EXPERIMENT_NAME}'...")
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    filter_string="tags.approved = 'true'",
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    print("❌ No approved run found!")
    print("   Developer must tag the best run before pushing:")
    print("   mlflow.set_tag('approved', 'true')  ← in MLflow UI or in code")
    sys.exit(1)

approved_run    = runs.iloc[0]
approved_run_id = approved_run["run_id"]
print(f"  Approved run ID : {approved_run_id}")
print(f"  Val accuracy    : {approved_run.get('metrics.val_accuracy', 'N/A')}")

# ── 3. Evaluate approved model on secret test data ───────────────────────────
print("\nLoading approved model and evaluating on secret test data...")
new_model    = mlflow.tensorflow.load_model(f"runs:/{approved_run_id}/{MODEL_ARTIFACT}")
y_pred       = new_model.predict(X_test, verbose=0).argmax(axis=1)
new_accuracy = float((y_pred == y_test).mean())
print(f"  Secret test accuracy: {new_accuracy:.4f}")

# ── 4. Hard floor check ──────────────────────────────────────────────────────
if new_accuracy < MIN_ACCURACY:
    print(f"\n❌ FAILED: {new_accuracy:.4f} < minimum floor {MIN_ACCURACY}")
    sys.exit(1)

# ── 5. Get current Production model ─────────────────────────────────────────
prod_accuracy = 0.0
prod_versions = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])

if prod_versions:
    prod_run      = mlflow.get_run(prod_versions[0].run_id)
    prod_accuracy = prod_run.data.metrics.get("secret_test_accuracy", 0.0)
    prod_team     = prod_run.data.tags.get("team", "unknown")
    print(f"\n  Current Production: v{prod_versions[0].version} (team: {prod_team})")
    print(f"  Production accuracy: {prod_accuracy:.4f}")
else:
    print("\n  No Production model yet — this will be the first.")

# ── 6. Compare ────────────────────────────────────────────────────────────────
print(f"\n  Approved model : {new_accuracy:.4f}")
print(f"  Production     : {prod_accuracy:.4f}")

if new_accuracy <= prod_accuracy:
    print(f"\n❌ FAILED: approved model ({new_accuracy:.4f}) does not beat Production ({prod_accuracy:.4f})")
    sys.exit(1)

# ── 7. Promote ───────────────────────────────────────────────────────────────
print("\n✅ Approved model wins! Promoting to Production...")

with mlflow.start_run(run_id=approved_run_id):
    mlflow.log_metric("secret_test_accuracy", new_accuracy)
    mlflow.set_tag("approved", "false")  # clear tag so it can't be re-promoted accidentally

all_versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
new_version  = next((v for v in all_versions if v.run_id == approved_run_id), None)

if new_version is None:
    print(f"❌ No registered version found for run {approved_run_id}")
    print(f"   Make sure train.py uses registered_model_name='{REGISTERED_NAME}'")
    sys.exit(1)

if prod_versions:
    client.transition_model_version_stage(
        name=REGISTERED_NAME, version=prod_versions[0].version, stage="Archived"
    )
    print(f"  Archived old Production (v{prod_versions[0].version})")

client.transition_model_version_stage(
    name=REGISTERED_NAME, version=new_version.version, stage="Production"
)
print(f"  🚀 '{REGISTERED_NAME}' v{new_version.version} is now Production!")
print(f"     Secret test accuracy: {new_accuracy:.4f}")
