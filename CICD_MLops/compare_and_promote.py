"""
Shared Quality Gate — used by ALL teams' CI pipelines.

IMPORTANT: The pipeline does NOT re-train the model.
Developer trains locally (grid search), train.py auto-saves artifact for
the best run and registers it to Staging in the Model Registry.
Developer reviews MLflow UI — can change which version is Staging if preferred.
Developer pushes code → this pipeline runs.

This script:
  1. Loads the current Staging model from the Model Registry
  2. Evaluates it on secret test data (developer never sees this)
  3. Compares against current Production model
  4. If better → promotes Staging to Production
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
REGISTERED_NAME  = os.environ.get("REGISTERED_NAME", "mnist_classifier")
SECRET_TEST_PATH = os.environ.get("SECRET_TEST_PATH", "secret_test_data.npz")
MIN_ACCURACY     = float(os.environ.get("MIN_ACCURACY", "0.90"))

client = MlflowClient()

print(f"🔍 Quality Gate")
print(f"   Registered model: '{REGISTERED_NAME}'")
print(f"   Minimum accuracy: {MIN_ACCURACY}")
print()

# ── 1. Load secret test data ─────────────────────────────────────────────────
print("Loading secret test data...")
data   = np.load(SECRET_TEST_PATH)
X_test = data["X_test"].astype("float32")
y_test = data["y_test"]
print(f"  {X_test.shape[0]} samples, classes: {sorted(set(y_test.tolist()))}")

# ── 2. Find the Staging model (developer set this in MLflow UI) ──────────────
print(f"\nLooking for Staging version of '{REGISTERED_NAME}'...")
staging_versions = client.get_latest_versions(REGISTERED_NAME, stages=["Staging"])

if not staging_versions:
    print("❌ No model in Staging!")
    print("   Developer must run train.py locally first.")
    print("   train.py auto-registers the best run to Staging.")
    print("   Developer can also change which version is Staging in MLflow UI.")
    sys.exit(1)

staging_version = staging_versions[0]
staging_run_id  = staging_version.run_id
staging_run     = mlflow.get_run(staging_run_id)
staging_team    = staging_run.data.tags.get("team", "unknown")
print(f"  Staging version : v{staging_version.version}")
print(f"  Team            : {staging_team}")
print(f"  Val accuracy    : {staging_run.data.metrics.get('val_accuracy', 'N/A')}")

# ── 3. Evaluate Staging model on secret test data ────────────────────────────
print("\nLoading Staging model and evaluating on secret test data...")
staging_model = mlflow.tensorflow.load_model(f"models:/{REGISTERED_NAME}/Staging")
y_pred        = staging_model.predict(X_test, verbose=0).argmax(axis=1)
new_accuracy  = float((y_pred == y_test).mean())
print(f"  Secret test accuracy: {new_accuracy:.4f}")

# ── 4. Hard floor check ──────────────────────────────────────────────────────
if new_accuracy < MIN_ACCURACY:
    print(f"\n❌ FAILED: {new_accuracy:.4f} < minimum floor {MIN_ACCURACY}")
    print("   Model is not good enough. Blocking deployment.")
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
print(f"\n  Staging model  : {new_accuracy:.4f}")
print(f"  Production     : {prod_accuracy:.4f}")

if new_accuracy <= prod_accuracy:
    print(f"\n❌ FAILED: Staging ({new_accuracy:.4f}) does not beat Production ({prod_accuracy:.4f})")
    print("   Keeping current Production. Blocking merge.")
    sys.exit(1)

# ── 7. Promote Staging → Production ──────────────────────────────────────────
print("\n✅ Staging model wins! Promoting to Production...")

# Log secret test accuracy to the run for future comparisons
with mlflow.start_run(run_id=staging_run_id):
    mlflow.log_metric("secret_test_accuracy", new_accuracy)

# Archive old Production
if prod_versions:
    client.transition_model_version_stage(
        name=REGISTERED_NAME, version=prod_versions[0].version, stage="Archived"
    )
    print(f"  Archived old Production (v{prod_versions[0].version}, team: {prod_team})")

# Promote Staging to Production
client.transition_model_version_stage(
    name=REGISTERED_NAME, version=staging_version.version, stage="Production"
)
print(f"  🚀 '{REGISTERED_NAME}' v{staging_version.version} is now Production!")
print(f"     Team: {staging_team} | Secret test accuracy: {new_accuracy:.4f}")
