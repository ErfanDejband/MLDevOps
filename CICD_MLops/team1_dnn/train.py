import os
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras # type: ignore
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from itertools import product
import logging
import warnings

# Load .env file when running locally (ignored if vars already set by CI)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "mnist_classifier"
CANDIDATE_ALIAS = "team1-candidate"


# ── MLflow setup — dev-tracking only; CI/CD talks to both servers itself ────

def setup_mlflow():
    tracking_uri = os.environ.get("MLFLOW_DEV_TRACKING_URI")
    username     = os.environ.get("MLFLOW_DEV_USERNAME")
    password     = os.environ.get("MLFLOW_DEV_PASSWORD")

    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_DEV_TRACKING_URI not set.\n"
            "  Copy CICD_MLops/.env.example → CICD_MLops/team1_dnn/.env and fill in\n"
            "  your mlops-dev-tracking DagsHub values."
        )

    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username or ""
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password or ""

    mlflow.set_experiment("team1_dnn")
    logger.info(f"MLflow connected → {tracking_uri} ✅")


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    (X_all, y_all), (X_holdout, y_holdout) = keras.datasets.mnist.load_data()

    # Combine all data, then do our own 70/20/10 split
    X_full = np.concatenate([X_all, X_holdout], axis=0)
    y_full = np.concatenate([y_all, y_holdout], axis=0)

    # Normalize pixel values to [0, 1] and flatten 28x28 → 784
    X_full = X_full.astype("float32") / 255.0
    X_flat = X_full.reshape(-1, 784)

    # Split 1: 90% temp, 10% secret test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_flat, y_full, test_size=0.10, random_state=42, stratify=y_full
    )

    # Split 2: 70% train, 20% validation (from the 90%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp
    )

    logger.info(f"Data split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Secret test set — used by the CI/CD quality gate only, never committed
    np.savez("secret_test_data.npz", X_test=X_test, y_test=y_test)
    logger.info("Secret test data saved to secret_test_data.npz (do NOT commit this)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(dropout_rate: float = 0.2) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(512, activation="relu", input_shape=(784,)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation="softmax"),
    ])
    return model


# ── Training ─────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val, params: dict) -> tuple:
    model = build_model(dropout_rate=params["dropout_rate"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_data=(X_val, y_val),
        verbose=0
    )

    val_accuracy = max(history.history["val_accuracy"])
    val_loss     = min(history.history["val_loss"])
    logger.info(f"\tVal Accuracy: {val_accuracy:.4f}  |  Val Loss: {val_loss:.4f}")
    return model, val_accuracy, val_loss


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_mlflow()
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Parameter grid — add/change values here to experiment
    param_grid = {
        "epochs": [20],
        "batch_size": [ 128],
        "learning_rate": [0.001],
        "dropout_rate": [0.2, 0.4]
    }

    keys, values = zip(*param_grid.items())
    all_param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    logger.info(f"Total hyperparameter combinations to try: {len(all_param_combinations)}")
    best_run_id       = None
    best_val_accuracy = -1.0
    best_model        = None

    for i, params in enumerate(all_param_combinations):
        logger.info("=" * 55)
        logger.info(f"Run {i+1}/{len(all_param_combinations)} | Params: {params}")

        model, val_accuracy, val_loss = train(X_train, y_train, X_val, y_val, params)

        run_name = (
            f"dnn_e{params['epochs']}"
            f"_bs{params['batch_size']}"
            f"_lr{params['learning_rate']}"
            f"_dr{params['dropout_rate']}"
        )
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_param("architecture", "Dense512-Drop-Dense256-Drop-Softmax10")
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.set_tag("team", "team1_dnn")
            mlflow.set_tag("dataset", "mnist")
            # No artifact saved here — only params + metrics, to keep the
            # experiment cheap to browse across the whole grid search

            logger.info("\tRun logged ✅ (params + metrics only)")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_run_id       = run.info.run_id
                best_model        = model

    # ── After all runs: save artifact ONLY for the best run ──────────────────
    logger.info("=" * 55)
    logger.info(f"Best run: {best_run_id} | Val accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Saving artifact for best run and registering as '{CANDIDATE_ALIAS}'...")

    with mlflow.start_run(run_id=best_run_id):
        sample_input  = X_val[:5].astype("float32")
        sample_output = best_model.predict(sample_input, verbose=0)
        signature     = infer_signature(sample_input, sample_output)

        mlflow.tensorflow.log_model(
            model=best_model,
            name="mnist_dnn_model",
            signature=signature,
            input_example=sample_input,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    # Set the team's candidate alias on the newly registered version
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    new_version  = next((v for v in all_versions if v.run_id == best_run_id), None)
    if new_version:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CANDIDATE_ALIAS, new_version.version)
        logger.info(f"✅ {REGISTERED_MODEL_NAME} v{new_version.version} → alias '{CANDIDATE_ALIAS}' set")
        logger.info("Review in MLflow UI. When ready: git push + open PR to trigger the quality gate.")

    logger.info("View runs on your MLFLOW_DEV_TRACKING_URI DagsHub repo")
