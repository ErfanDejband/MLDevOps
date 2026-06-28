import os
import numpy as np
import mlflow
import mlflow.tensorflow
import dagshub
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ── DagsHub / MLflow setup ───────────────────────────────────────────────────

def setup_mlflow():
    dagshub.init(
        repo_owner="e.dejband",
        repo_name="mlops-dev-tracking",
        mlflow=True
    )
    mlflow.set_experiment("team1_dnn_mnist")
    logger.info("MLflow connected to DagsHub ✅")


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

    # Save secret test set — used by CI pipeline only, do NOT commit this file
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
    param_grid = [
        {"epochs": 5,  "batch_size": 128, "learning_rate": 0.001, "dropout_rate": 0.2},
        {"epochs": 5,  "batch_size": 64,  "learning_rate": 0.001, "dropout_rate": 0.3},
        {"epochs": 10, "batch_size": 128, "learning_rate": 0.0005,"dropout_rate": 0.2},
    ]

    for i, params in enumerate(param_grid):
        logger.info("=" * 55)
        logger.info(f"Run {i+1}/{len(param_grid)} | Params: {params}")

        model, val_accuracy, val_loss = train(X_train, y_train, X_val, y_val, params)

        with mlflow.start_run(run_name=f"dnn_run_{i+1}"):
            # Log all hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("architecture", "Dense512-Drop-Dense256-Drop-Softmax10") #TODO automate this
            
            # Log validation metrics
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("val_loss", val_loss)

            mlflow.set_tag("team", "team1_dnn")
            mlflow.set_tag("dataset", "mnist")

            # Log model to MLflow (saved to DagsHub artifact storage)
            mlflow.tensorflow.log_model(
                model=model,
                name="mnist_dnn_model",
                registered_model_name="mnist_dnn"  # MLflow auto-versions: v1, v2, v3...
            )

            logger.info(f"\tRun logged ✅  →  {mlflow.get_artifact_uri()}")

    logger.info("=" * 55)
    logger.info("All runs complete. View at: https://dagshub.com/e.dejband/mlops-dev-tracking.mlflow")
