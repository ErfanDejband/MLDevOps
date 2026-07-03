"""
Shared Quality Gate — used by ALL teams' CI pipelines.

Two-server architecture (real-world pattern):
  DEV server  — where developers experiment and register models with 'staging' alias
  PROD server — where only CI writes; holds the official 'champion' model

Flow:
  1. Load 'staging' model FROM DEV server (developer set this alias)
  2. Evaluate on secret test data (developer never sees this)
  3. Check minimum accuracy floor
  4. Compare against current 'champion' on PROD server
  5. If staging wins → register model on PROD + set 'champion' alias
  6. If staging loses → block PR

Both Team 1 (DNN) and Team 2 (CNN) call this same script.
They compete for the SAME registered model name: 'mnist_classifier'.
"""
import sys
import os
import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.client import MlflowClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
REGISTERED_NAME  = os.environ.get("REGISTERED_NAME", "mnist_classifier")
SECRET_TEST_PATH = os.environ.get("SECRET_TEST_PATH", "secret_test_data.npz")
MIN_ACCURACY     = float(os.environ.get("MIN_ACCURACY", "0.90"))

# DEV server credentials (read staging model from here)
DEV_URI      = os.environ["MLFLOW_DEV_TRACKING_URI"]
DEV_USERNAME = os.environ["MLFLOW_DEV_USERNAME"]
DEV_PASSWORD = os.environ["MLFLOW_DEV_PASSWORD"]

# PROD server credentials (write champion model here)
PROD_URI      = os.environ["MLFLOW_PROD_TRACKING_URI"]
PROD_USERNAME = os.environ["MLFLOW_PROD_USERNAME"]
PROD_PASSWORD = os.environ["MLFLOW_PROD_PASSWORD"]

dev_client  = MlflowClient(tracking_uri=DEV_URI,
                            registry_uri=DEV_URI)
prod_client = MlflowClient(tracking_uri=PROD_URI,
                            registry_uri=PROD_URI)

print(f"🔍 Quality Gate")
print(f"   Registered model: '{REGISTERED_NAME}'")
print(f"   DEV  server: {DEV_URI}")
print(f"   PROD server: {PROD_URI}")
print(f"   Minimum accuracy: {MIN_ACCURACY}")
print()

# ── 1. Load secret test data ─────────────────────────────────────────────────
print("Loading secret test data...")
data   = np.load(SECRET_TEST_PATH)
X_test = data["X_test"].astype("float32")
y_test = data["y_test"]
print(f"  {X_test.shape[0]} samples, classes: {sorted(set(y_test.tolist()))}")

# ── 2. Find 'staging' model on DEV server ────────────────────────────────────
print(f"\nLooking for '{REGISTERED_NAME}@staging' on DEV server...")
try:
    staging_mv = dev_client.get_model_version_by_alias(REGISTERED_NAME, "staging")
except Exception:
    print("❌ No 'staging' alias found on DEV server!")
    print("   Developer must run train.py locally first.")
    print("   train.py auto-registers best run and sets 'staging' alias.")
    sys.exit(1)

staging_run  = mlflow.tracking.MlflowClient(tracking_uri=DEV_URI).get_run(staging_mv.run_id)
staging_team = staging_run.data.tags.get("team", "unknown")
print(f"  Version  : v{staging_mv.version}")
print(f"  Team     : {staging_team}")
print(f"  Val acc  : {staging_run.data.metrics.get('val_accuracy', 'N/A')}")

# ── 3. Evaluate staging model on secret test data ────────────────────────────
print("\nLoading staging model from DEV and evaluating on secret test data...")
mlflow.set_tracking_uri(DEV_URI)
os.environ["MLFLOW_TRACKING_URI"]      = DEV_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = DEV_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DEV_PASSWORD
staging_model = mlflow.tensorflow.load_model(f"models:/{REGISTERED_NAME}@staging")

y_pred       = staging_model.predict(X_test, verbose=0).argmax(axis=1)
new_accuracy = float((y_pred == y_test).mean())
print(f"  Secret test accuracy: {new_accuracy:.4f}")

# ── 4. Hard floor check ──────────────────────────────────────────────────────
if new_accuracy < MIN_ACCURACY:
    print(f"\n❌ FAILED: {new_accuracy:.4f} < minimum floor {MIN_ACCURACY}")
    sys.exit(1)

# ── 5. Get current champion from PROD server ─────────────────────────────────
champion_accuracy = 0.0
champion_mv       = None
try:
    champion_mv      = prod_client.get_model_version_by_alias(REGISTERED_NAME, "champion")
    champion_run     = mlflow.tracking.MlflowClient(tracking_uri=PROD_URI).get_run(champion_mv.run_id)
    champion_accuracy = champion_run.data.metrics.get("secret_test_accuracy", 0.0)
    champion_team    = champion_run.data.tags.get("team", "unknown")
    print(f"\n  Current champion (PROD): v{champion_mv.version} (team: {champion_team})")
    print(f"  Champion accuracy: {champion_accuracy:.4f}")
except Exception:
    print("\n  No champion on PROD yet — this will be the first.")

# ── 6. Compare ────────────────────────────────────────────────────────────────
print(f"\n  Staging (DEV)    : {new_accuracy:.4f}")
print(f"  Champion (PROD)  : {champion_accuracy:.4f}")

if new_accuracy <= champion_accuracy:
    print(f"\n❌ FAILED: staging ({new_accuracy:.4f}) does not beat champion ({champion_accuracy:.4f})")
    sys.exit(1)

# ── 7. Register model on PROD and set 'champion' alias ───────────────────────
print("\n✅ Staging model wins! Registering on PROD and setting as champion...")

# Explicitly switch MLflow module to PROD (os.environ alone is not enough)
mlflow.set_tracking_uri(PROD_URI)
os.environ["MLFLOW_TRACKING_URI"]      = PROD_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = PROD_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = PROD_PASSWORD

# Ensure the promotion experiment exists on PROD before creating a run
PROD_EXPERIMENT = "quality_gate_promotions"
try:
    prod_exp = mlflow.get_experiment_by_name(PROD_EXPERIMENT)
    prod_exp_id = prod_exp.experiment_id if prod_exp else mlflow.create_experiment(PROD_EXPERIMENT)
except Exception:
    prod_exp_id = None  # falls back to Default experiment

with mlflow.start_run(
    experiment_id=prod_exp_id,
    tags={"team": staging_team, "promoted_from_dev_run": staging_mv.run_id}
):
    mlflow.log_metric("secret_test_accuracy", new_accuracy)
    mlflow.log_metric("val_accuracy", staging_run.data.metrics.get("val_accuracy", 0))
    mlflow.log_params(staging_run.data.params)
    mlflow.set_tag("team", staging_team)

    prod_model_info = mlflow.tensorflow.log_model(
        model=staging_model,
        name="mnist_classifier_model",
        registered_model_name=REGISTERED_NAME
    )

# Set 'champion' alias on PROD, remove old one if exists
all_prod_versions = prod_client.search_model_versions(f"name='{REGISTERED_NAME}'")
new_prod_version  = max(all_prod_versions, key=lambda v: int(v.version))
prod_client.set_registered_model_alias(REGISTERED_NAME, "champion", new_prod_version.version)

# Clean up 'staging' alias on DEV (prevents accidental re-promotion)
dev_client.delete_registered_model_alias(REGISTERED_NAME, "staging")

print(f"  🚀 '{REGISTERED_NAME}' v{new_prod_version.version} is champion on PROD!")
print(f"     Team: {staging_team} | Secret test accuracy: {new_accuracy:.4f}")