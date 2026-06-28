# Introduction 
Using ML devops tools

# Getting Started
0. cd to mlflow_local 
1.	python -m venv .venv
2.	pip install requirements.
3. cd to Local_setup and run the scripts (1- run and log | 2- load model)
4. see the database and mlartifacts


# Contribute
MLflow website: https://mlflow.org/


# MLflow Complete Guide: Basics to Advanced

## Table of Contents
1. [What is MLflow?](#what-is-mlflow)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Key Functions & Methods](#key-functions--methods)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Common Workflows](#common-workflows)

---

## What is MLflow?

**MLflow** is an open-source platform for managing machine learning workflows. It helps you:
- **Track experiments** - Log parameters, metrics, and outputs
- **Reproduce runs** - Save code versions and model snapshots
- **Package models** - Save models in a standardized format
- **Deploy models** - Serve models in production
- **Compare models** - Visualize performance across runs

Think of it as **Git for machine learning models** + **TensorBoard** + **Model registry**.

---

## Installation & Setup

```bash
# Install MLflow
pip install mlflow

# Install with extra dependencies (for serving models)
pip install mlflow[extras]

# Verify installation
mlflow --version
```

---

## Core Concepts

### 1. **Experiment**
A collection of related runs. Think of it as a project or task.
- Example: "Iris Classification", "Customer Churn Prediction"

### 2. **Run**
A single training execution with specific parameters and results.
- Each run has unique ID, start time, end time

### 3. **Parameters**
Input values for your model (fixed before training)
- Examples: `max_iter=100`, `learning_rate=0.01`

### 4. **Metrics**
Numerical values that measure model performance (computed during/after training)
- Examples: `accuracy=0.95`, `loss=0.23`

### 5. **Artifacts**
Files saved during training (models, plots, data)
- Examples: model pickle file, confusion matrix plot, feature importance CSV

### 6. **Run ID**
Unique identifier for each run (auto-generated)
- Used to retrieve run data later

---

## Basic Usage

### Step 1: Set Tracking URI (where MLflow stores data)

```python
import mlflow

# Local directory storage (SQLite database + artifacts folder)
mlflow.set_tracking_uri("./mlruns")

# Or use MLflow server (remote tracking)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

### Step 2: Create/Set Experiment

```python
# Create a new experiment
experiment_id = mlflow.create_experiment("My_Experiment")
mlflow.set_experiment("My_Experiment")

# Or if experiment already exists, just set it
mlflow.set_experiment("My_Experiment")
```

### Step 3: Log Parameters, Metrics, and Artifacts

```python
import mlflow

with mlflow.start_run():
    # Log parameters (before training)
    mlflow.log_param("max_iter", 100)
    mlflow.log_param("learning_rate", 0.01)
    
    # Train model...
    accuracy = 0.95
    
    # Log metrics (after training)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("loss", 0.23)
    
    # Log artifacts (files)
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("confusion_matrix.png")
```

### Step 4: View Results

```bash
# Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# Open browser at http://127.0.0.1:5000
```

---

## Key Functions & Methods

### `mlflow.set_tracking_uri(uri)`
Sets where MLflow stores tracking data.

```python
# Local storage (creates ./mlruns folder)
mlflow.set_tracking_uri("./mlruns")

# Remote server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Databricks workspace
mlflow.set_tracking_uri("databricks")
```

---

### `mlflow.create_experiment(name, artifact_location=None)`
Creates a new experiment.

```python
experiment_id = mlflow.create_experiment(
    name="Iris_Classification",
    artifact_location="./artifacts"  # Optional
)
print(experiment_id)  # Returns experiment ID
```

---

### `mlflow.set_experiment(experiment_name)`
Selects an experiment for logging runs.

```python
# If experiment exists, use it
mlflow.set_experiment("Iris_Classification")

# If doesn't exist, creates it
mlflow.set_experiment("New_Experiment")
```

---

### `mlflow.start_run(run_name=None, run_id=None, experiment_id=None, tags=None, ...)`
Starts a new run context. All logging happens inside this context.

```python
# Basic
with mlflow.start_run():
    mlflow.log_param("x", 1)
    # Do training
    
# With custom name and tags
with mlflow.start_run(run_name="test_v1", tags={"model": "lr"}):
    mlflow.log_param("x", 1)

# Manual start/end (no context manager)
mlflow.start_run()
mlflow.log_param("x", 1)
mlflow.end_run()

# Run ID available after start
run = mlflow.active_run()
print(run.info.run_id)
```

---

### `mlflow.log_param(key, value)`
Logs a single parameter (before/after training).

```python
mlflow.log_param("max_iter", 100)
mlflow.log_param("solver", "lbfgs")
mlflow.log_param("learning_rate", 0.001)
```

---

### `mlflow.log_params(params)`
Logs multiple parameters at once.

```python
model_parameters = {
    "max_iter": 100,
    "solver": "lbfgs",
    "random_state": 42
}
mlflow.log_params(model_parameters)
```

---

### `mlflow.log_metric(key, value, step=None)`
Logs a metric value. Can log multiple times to track progression.

```python
# Single metric
mlflow.log_metric("accuracy", 0.95)

# Log progression (e.g., during training)
for epoch in range(10):
    loss = train_one_epoch()
    mlflow.log_metric("loss", loss, step=epoch)
```

---

### `mlflow.log_metrics(metrics, step=None)`
Logs multiple metrics at once.

```python
metrics = {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.91,
    "f1": 0.92
}
mlflow.log_metrics(metrics)
```

---

### `mlflow.log_artifact(local_path, artifact_path=None)`
Saves a file to MLflow.

```python
# Save single file
mlflow.log_artifact("confusion_matrix.png")

# Save with subdirectory in MLflow
mlflow.log_artifact("model.pkl", artifact_path="models")

# After saving, view in UI under "Artifacts" tab
```

---

### `mlflow.log_artifacts(local_dir, artifact_path=None)`
Saves an entire directory.

```python
mlflow.log_artifacts("./plots/")  # Saves all files in plots/
mlflow.log_artifacts("./models/", artifact_path="model_snapshots")
```

---

### `infer_signature(model_input, model_output)` ⭐ IMPORTANT

This creates a **model signature** - a description of expected input/output types. Essential for model serving.

```python
from mlflow.models import infer_signature
import numpy as np

# After training
X_test = np.array([[5.1, 3.5, 1.4, 0.2], ...])  # Input shape: (n_samples, 4)
y_pred = model.predict(X_test)  # Output shape: (n_samples,)

# Infer signature
signature = infer_signature(X_test, y_pred)
# Returns: inputs=[ColSpec(type=numpy_array, shape=[-1, 4])],
#          outputs=[ColSpec(type=numpy_array, shape=[-1])]

# Use when logging model
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature
)
```

**Why is this important?**
- When serving the model, MLflow validates input/output types
- Prevents errors from wrong input shapes or types
- Documents model interface clearly

---

### `mlflow.sklearn.log_model(sk_model, name, signature=None, input_example=None, registered_model_name=None)`
Saves a scikit-learn model in MLflow format.

⚠️ **NOTE:** `artifact_path` is **DEPRECATED** - use `name` instead (as of MLflow 2.8+)

```python
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

signature = infer_signature(X_test, y_pred)
input_example = X_test[:5]  # Show example input

mlflow.sklearn.log_model(
    sk_model=model,
    name="iris_model",  # ✅ NEW - replaces artifact_path
    signature=signature,
    input_example=input_example
    # Note: registered_model_name auto-creates versions, don't use with iterations
)
```

**About `registered_model_name`:**
- If you use `registered_model_name="MyModel"` in multiple runs, MLflow automatically creates versions (v1, v2, v3...)
- You DON'T need to add `_v1`, `_v2` to the name - MLflow handles versioning
- Simpler: just use the same name across runs

```python
# ✅ GOOD - MLflow auto-manages versions
for i in range(3):
    with mlflow.start_run(run_name=f"trial_{i}"):
        model = LogisticRegression(max_iter=i*50)
        model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model",
            registered_model_name="Iris_Classifier"  # Same name every time
            # MLflow creates v1, v2, v3 automatically
        )

# ❌ OLD WAY - Don't do this
for i in range(3):
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        registered_model_name=f"Iris_Classifier_v{i}"  # Unnecessary
    )
```

---

### `mlflow.tensorflow.log_model()` / `mlflow.pytorch.log_model()` / etc.
Similar to `sklearn.log_model`, but for different frameworks.

```python
# TensorFlow
mlflow.tensorflow.log_model(tf_model, artifact_path="tf_model")

# PyTorch
mlflow.pytorch.log_model(torch_model, artifact_path="torch_model")

# XGBoost
mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_model")

# Generic (for custom models)
mlflow.pyfunc.log_model(pyfunc_model, artifact_path="custom_model")
```

---

### `mlflow.set_tag(key, value)` / `mlflow.set_tags(tags)`
Adds metadata tags to a run.

```python
# Single tag
mlflow.set_tag("team", "data_science")
mlflow.set_tag("model_type", "classification")

# Multiple tags
mlflow.set_tags({
    "version": "v1.0",
    "env": "production",
    "author": "John"
})
```

---

### `mlflow.search_runs(experiment_names=None, filter_string=None, order_by=None)`
Queries previous runs.

```python
import mlflow

# Search runs in current experiment
runs = mlflow.search_runs()

# Search across multiple experiments
runs = mlflow.search_runs(
    experiment_names=["Iris_v1", "Iris_v2"]
)

# Filter by metrics/params
runs = mlflow.search_runs(
    filter_string="metrics.accuracy > 0.9 AND params.max_iter = 100"
)

# Get best run (highest accuracy)
best_run = runs.sort_values("metrics.accuracy", ascending=False).iloc[0]
print(f"Best run ID: {best_run.run_id}")
print(f"Best accuracy: {best_run['metrics.accuracy']}")
```

---

### `mlflow.get_run(run_id)`
Retrieves a specific run's details.

```python
run_id = "abc123"
run = mlflow.get_run(run_id)

# Access run info
print(run.info.run_id)
print(run.info.start_time)
print(run.info.end_time)

# Access logged data
print(run.data.params)  # All parameters
print(run.data.metrics)  # All metrics
```

---

### `mlflow.load_model(model_uri)`
Loads a previously saved model.

```python
# Load by run ID and artifact path
model = mlflow.sklearn.load_model(
    "runs/abc123/iris_model"
)

# Use loaded model for predictions
y_pred = model.predict(X_test)
```

---

## Advanced Features

### 1. **Model Registry** - Centralized Model Management

The Model Registry lets you manage model lifecycle: staging, production, archiving.

```python
from mlflow.client import MlflowClient

# Log model with registered name
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="Iris_Classifier"  # Creates model in registry
)

client = MlflowClient()

# Get model details
model_details = client.get_registered_model("Iris_Classifier")
print(model_details.latest_versions)

# Transition to production
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

client.transition_model_version_stage(
    name="Iris_Classifier",
    version=1,
    stage="Production"  # "Staging", "Production", "Archived"
)

# Get production model version
for version in model_details.latest_versions:
    if version.current_stage == "Production":
        print(f"Production version: {version.version}")
```

---

### 2. **Nested Runs** - Runs within Runs

Useful for hyperparameter tuning where each parameter combo is a nested run.

```python
with mlflow.start_run(run_name="hyperparameter_tuning"):
    mlflow.log_param("search_space", "grid")
    
    for max_iter in [50, 100, 200]:
        with mlflow.start_run(nested=True, run_name=f"max_iter_{max_iter}"):
            mlflow.log_param("max_iter", max_iter)
            
            model = LogisticRegression(max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            
            mlflow.log_metric("accuracy", accuracy)
```

---

### 3. **Autolog** - Automatic Logging

MLflow can automatically log parameters, metrics, and models for supported frameworks.

```python
# Enable autolog for sklearn
mlflow.sklearn.autolog()

# Now just train normally - everything is logged automatically!
with mlflow.start_run():
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    # Params and metrics logged automatically!

# Available for other frameworks too
mlflow.tensorflow.autolog()
mlflow.pytorch.autolog()
mlflow.xgboost.autolog()
```

---

### 4. **Context-Aware Logging**

Set experiment before starting run, it auto-associates.

```python
mlflow.set_experiment("Iris_Experiment")

# This run automatically goes to "Iris_Experiment"
with mlflow.start_run():
    mlflow.log_param("x", 1)
    # Run is in "Iris_Experiment"
```

---

## Best Practices

### 1. **Always use Context Managers**

```python
# ✅ GOOD - Automatic cleanup
with mlflow.start_run():
    mlflow.log_param("x", 1)
    # Run automatically ends here

# ❌ BAD - Manual cleanup required
mlflow.start_run()
mlflow.log_param("x", 1)
mlflow.end_run()  # Easy to forget!
```

---

### 2. **Log Parameters Before Training**

```python
# ✅ GOOD
with mlflow.start_run():
    params = {"max_iter": 100, "solver": "lbfgs"}
    mlflow.log_params(params)
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

# ❌ CONFUSING - Hard to track what params were used
with mlflow.start_run():
    model = LogisticRegression(max_iter=100)
    mlflow.log_param("max_iter", 100)  # After training
```

---

### 3. **Use Meaningful Run Names**

```python
# ✅ GOOD - Clear what this run tests
with mlflow.start_run(run_name="lr_lbfgs_100iter"):
    ...

# ❌ BAD - Non-descriptive
with mlflow.start_run(run_name="run_1"):
    ...
```

---

### 4. **Always Log Model Signature**

```python
# ✅ GOOD - Serves model reliably
signature = infer_signature(X_test, y_pred)
mlflow.sklearn.log_model(model, "model", signature=signature)

# ⚠️ OK but risky - No validation on serving
mlflow.sklearn.log_model(model, "model")
```

---

### 5. **Use Tags for Organization**

```python
with mlflow.start_run():
    mlflow.set_tags({
        "team": "data_science",
        "project": "iris_classification",
        "env": "experiment",
        "model_type": "logistic_regression"
    })
```

---

### 6. **Log Artifacts for Reproducibility**

```python
with mlflow.start_run():
    # Save preprocessing steps
    mlflow.log_artifact("preprocessing.pkl")
    
    # Save feature engineering code
    mlflow.log_artifact("feature_engineering.py")
    
    # Save model evaluation plots
    mlflow.log_artifact("plots/confusion_matrix.png")
```

---

## Common Workflows

### Workflow 1: Basic Model Training with Logging

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

# Setup
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Iris_Classification")

# Training
with mlflow.start_run(run_name="baseline_lr"):
    # Log params
    params = {"max_iter": 200, "solver": "lbfgs", "random_state": 42}
    mlflow.log_params(params)
    
    # Train
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model (using 'name' not deprecated 'artifact_path')
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",  # ✅ Use 'name' parameter
        signature=signature
    )
    
    print(f"✅ Run ID: {mlflow.active_run().info.run_id}")
```

---

### Workflow 2: Hyperparameter Tuning

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Iris_HPTuning")

# Grid search with auto-versioning
with mlflow.start_run(run_name="grid_search"):
    mlflow.log_param("search_type", "grid_search")
    
    best_accuracy = 0
    best_params = None
    
    for max_iter in [50, 100, 200]:
        for solver in ["lbfgs", "saga"]:
            for C in [0.1, 1.0, 10.0]:
                with mlflow.start_run(nested=True):
                    params = {
                        "max_iter": max_iter,
                        "solver": solver,
                        "C": C
                    }
                    mlflow.log_params(params)
                    
                    model = LogisticRegression(**params)
                    model.fit(X_train, y_train)
                    accuracy = model.score(X_test, y_test)
                    
                    mlflow.log_metric("accuracy", accuracy)
                    
                    # ✅ Use same 'registered_model_name' - MLflow auto-versions
                    mlflow.sklearn.log_model(
                        model, 
                        name="model",
                        registered_model_name="Iris_Tuned"  # v1, v2, v3... auto-created
                    )
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = params
    
    print(f"Best accuracy: {best_accuracy}")
    print(f"Best params: {best_params}")
```

---

### Workflow 3: Comparing Models

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_experiment("Iris_ModelComparison")

# Model 1: Logistic Regression
with mlflow.start_run(run_name="logistic_regression"):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))

# Model 2: Random Forest
with mlflow.start_run(run_name="random_forest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))

# Compare in UI at http://127.0.0.1:5000
# mlflow ui
```

---

### Workflow 4: Loading and Using Previous Model

```python
import mlflow

# Find best run
runs = mlflow.search_runs(
    experiment_names=["Iris_Classification"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

best_run = runs.iloc[0]
print(f"Best run: {best_run.run_id}")
print(f"Accuracy: {best_run['metrics.accuracy']}")

# Load model
model = mlflow.sklearn.load_model(f"runs/{best_run.run_id}/model")

# Use for predictions
y_pred = model.predict(new_data)
```

---

## Quick Reference

| Function | Purpose |
|----------|---------|
| `mlflow.set_tracking_uri(uri)` | Set where to save MLflow data |
| `mlflow.set_experiment(name)` | Select/create experiment |
| `mlflow.start_run()` | Begin a run |
| `mlflow.log_param(key, value)` | Log single parameter |
| `mlflow.log_params(params)` | Log multiple parameters |
| `mlflow.log_metric(key, value)` | Log single metric |
| `mlflow.log_metrics(metrics)` | Log multiple metrics |
| `mlflow.log_artifact(path)` | Save file |
| `mlflow.log_artifacts(dir)` | Save directory |
| `infer_signature(input, output)` | Create model signature |
| `mlflow.sklearn.log_model()` | Save sklearn model |
| `mlflow.search_runs()` | Query runs |
| `mlflow.get_run(run_id)` | Get run details |
| `mlflow.load_model(uri)` | Load saved model |
| `mlflow.set_tag(key, value)` | Add metadata |

---

## Troubleshooting

### Issue: "No experiments found"
```python
# Solution: Create an experiment first
mlflow.create_experiment("My_Experiment")
mlflow.set_experiment("My_Experiment")
```

### Issue: "Cannot write to artifact storage"
```python
# Solution: Ensure directory is writable
import os
os.makedirs("./mlruns", exist_ok=True)
mlflow.set_tracking_uri("./mlruns")
```

### Issue: Model signature mismatch
```python
# Solution: Ensure input/output shapes match
X_for_sig = X_test.iloc[:10]  # Use same type/shape as training
y_for_sig = model.predict(X_for_sig)
signature = infer_signature(X_for_sig, y_for_sig)
```

---

## Next Steps

1. **Explore MLflow UI**: `mlflow ui` and navigate experiments
2. **Try Autolog**: Enable automatic logging for your framework
3. **Use Model Registry**: Register and manage model versions
4. **Deploy Model**: Use MLflow to serve models in production
5. **Set Remote Tracking**: Connect to MLflow server for team collaboration

Happy experiment tracking! 🚀
