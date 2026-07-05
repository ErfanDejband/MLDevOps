import argparse
import logging
import os
import sys

import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException

# Resolved relative to this file, not the caller's CWD, so this script works
# the same whether you run it from the repo root or from CICD_MLops/shared.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM1_DNN_DIR = os.path.join(SCRIPT_DIR, "..", "team1_dnn")

# Local dry runs: load team1_dnn/.env explicitly. No-op in CI — there's no
# .env file there, only real GitHub Actions secrets as env vars.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(TEAM1_DNN_DIR, ".env"))
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

QUALITY_GATE_EXPERIMENT = "quality_gate_evaluations"
PRODUCTION_EXPERIMENT = "production"
CHAMPION_ALIAS = "champion"


def env(name, default=None, required=True):
    value = os.environ.get(name, default)
    if required and not value:
        raise EnvironmentError(f"{name} is not set")
    return value


def load_secret_test_set(path):
    data = np.load(path)
    return data["X_test"], data["y_test"]


def evaluate_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = predictions.argmax(axis=1)
    return float((predicted_labels == y_test).mean())


def get_champion_accuracy(prod_client, model_name):
    try:
        champion_version = prod_client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return None, None
    champion_run = prod_client.get_run(champion_version.run_id)
    return champion_version, champion_run.data.metrics.get("secret_test_accuracy")


def log_gate_result(mode, candidate_version, candidate_accuracy, champion_version, champion_accuracy, passed, floor):
    mlflow.set_experiment(QUALITY_GATE_EXPERIMENT)
    with mlflow.start_run(run_name=f"{mode}_v{candidate_version.version}"):
        mlflow.log_param("mode", mode)
        mlflow.log_param("candidate_version", candidate_version.version)
        mlflow.log_param("accuracy_floor", floor)
        mlflow.log_metric("candidate_accuracy", candidate_accuracy)
        if champion_accuracy is not None:
            mlflow.log_param("champion_version", champion_version.version)
            mlflow.log_metric("champion_accuracy", champion_accuracy)
        mlflow.log_param("passed", passed)


def promote_candidate(dev_client, prod_client, model_name, candidate_version, candidate_model, candidate_accuracy, X_test, team_tag):
    source_run = dev_client.get_run(candidate_version.run_id)

    mlflow.set_experiment(PRODUCTION_EXPERIMENT)
    with mlflow.start_run(run_name=f"promoted_{team_tag}_v{candidate_version.version}") as run:
        mlflow.log_params(source_run.data.params)
        mlflow.log_metrics(source_run.data.metrics)
        mlflow.log_metric("secret_test_accuracy", candidate_accuracy)
        mlflow.set_tag("source_team", team_tag)
        mlflow.set_tag("promoted_from_dev_version", candidate_version.version)

        sample_input = X_test[:5].astype("float32")
        sample_output = candidate_model.predict(sample_input, verbose=0)
        signature = infer_signature(sample_input, sample_output)

        mlflow.tensorflow.log_model(
            model=candidate_model,
            name="mnist_dnn_model",
            signature=signature,
            input_example=sample_input,
            registered_model_name=model_name,
        )
        promoted_run_id = run.info.run_id

    new_version = next(
        v for v in prod_client.search_model_versions(f"name='{model_name}'")
        if v.run_id == promoted_run_id
    )
    prod_client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version.version)
    logger.info(f"Champion promoted → {model_name} v{new_version.version} on prod-tracking")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "promote"], required=True)
    args = parser.parse_args()

    dev_uri           = env("MLFLOW_DEV_TRACKING_URI")
    prod_uri          = env("MLFLOW_PROD_TRACKING_URI")
    # In CI, MLFLOW_CI_USERNAME/PASSWORD are real GitHub secrets. Locally,
    # there's no separate CI identity (per our decision to reuse the personal
    # token) — fall back to the same MLFLOW_DEV_* credentials already in .env,
    # so a local dry run needs nothing beyond what train.py already uses.
    username          = env("MLFLOW_CI_USERNAME", default=os.environ.get("MLFLOW_DEV_USERNAME"))
    password          = env("MLFLOW_CI_PASSWORD", default=os.environ.get("MLFLOW_DEV_PASSWORD"))
    candidate_alias   = env("CANDIDATE_ALIAS", default="team1-candidate", required=False)
    model_name        = env("REGISTERED_MODEL_NAME", default="mnist_classifier", required=False)
    secret_test_path  = env("SECRET_TEST_PATH", default=os.path.join(TEAM1_DNN_DIR, "secret_test_data.npz"), required=False)
    accuracy_floor    = float(env("ACCURACY_FLOOR", default="0.90", required=False))

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    team_tag = candidate_alias.replace("-candidate", "")

    X_test, y_test = load_secret_test_set(secret_test_path)

    # ── Read the candidate from dev-tracking ──────────────────────────────
    mlflow.set_tracking_uri(dev_uri)
    dev_client = MlflowClient(tracking_uri=dev_uri)
    candidate_version = dev_client.get_model_version_by_alias(model_name, candidate_alias)
    candidate_model = mlflow.tensorflow.load_model(f"models:/{model_name}@{candidate_alias}")
    candidate_accuracy = evaluate_accuracy(candidate_model, X_test, y_test)
    logger.info(f"Candidate v{candidate_version.version} secret-test accuracy: {candidate_accuracy:.4f}")

    # ── Read the current champion from prod-tracking ──────────────────────
    prod_client = MlflowClient(tracking_uri=prod_uri)
    champion_version, champion_accuracy = get_champion_accuracy(prod_client, model_name)
    if champion_accuracy is not None:
        logger.info(f"Current champion v{champion_version.version} secret-test accuracy: {champion_accuracy:.4f}")
    else:
        logger.info("No champion registered yet on prod-tracking — bootstrap case")

    # ── Decide pass/fail ───────────────────────────────────────────────────
    clears_floor = candidate_accuracy >= accuracy_floor
    beats_champion = champion_accuracy is None or candidate_accuracy > champion_accuracy
    passed = clears_floor and beats_champion

    if not clears_floor:
        logger.info(f"FAIL — {candidate_accuracy:.4f} is below the {accuracy_floor:.2f} floor")
    elif not beats_champion:
        logger.info(f"FAIL — {candidate_accuracy:.4f} does not beat champion's {champion_accuracy:.4f}")
    else:
        logger.info("PASS")

    # ── Log the evaluation itself on prod-tracking (both modes do this) ───
    mlflow.set_tracking_uri(prod_uri)
    log_gate_result(args.mode, candidate_version, candidate_accuracy, champion_version, champion_accuracy, passed, accuracy_floor)

    if args.mode == "promote" and passed:
        promote_candidate(dev_client, prod_client, model_name, candidate_version, candidate_model, candidate_accuracy, X_test, team_tag)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
