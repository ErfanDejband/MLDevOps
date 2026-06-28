'''
Load model with highest accuracy and lowest max_iter as best run
Use the input and output to check it and then test it with the test data
'''
import json

import mlflow
import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Only show the message, no logger name or level
)

def get_best_run_and_model_uri():
    # 1. This part functions perfectly for you to find the best run
    runs = mlflow.search_runs(
        experiment_names=["MLflow_Local_Experiment"],
        output_format="pandas"
    )
    logger.info(f"{'='*150}")
    logger.info("just for learning purpose: runs DataFrame columns:")
    logger.info(f"Total runs found: {len(runs)}")
    logger.info(f"Runs DataFrame:\n{runs[['run_id', 'metrics.validation_accuracy', 'params.max_iter']]}")
    logger.info(f"{'='*150}")
    runs["params.max_iter"] = pd.to_numeric(runs["params.max_iter"])
    best_run = runs.sort_values(
        by=["metrics.validation_accuracy", "params.max_iter"],
        ascending=[False, True]
    ).iloc[0] # here we can any run that we need based on the requirements
    
    best_run_id = best_run["run_id"]
    logger.info(f"Best run ID: {best_run_id},Max Iter: {best_run['params.max_iter']}, Validation Accuracy: {best_run['metrics.validation_accuracy']}")

    # Query MLflow for the exact model version linked to this run
    # This searches the metadata directly on the server side (instantaneous)
    mv = mlflow.search_model_versions(filter_string=f"run_id = '{best_run_id}'")
    
    if not mv:
        raise ValueError(f"No logged models found for run_id: {best_run_id}")
        
    # The 'source' attribute contains the complete, direct URI to that 'm-...' folder 
    model_uri = mv[0].source
    logger.info(f"Direct Model URI fetched from MLflow metadata: {model_uri}")
    
    return model_uri


def test_best_model(model_uri):
    # Load the model directly from the flawless URI string returned by MLflow
    best_model = mlflow.sklearn.load_model(model_uri)
    logger.info("Best model loaded successfully!")

    # Load test data
    with open("test_data.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)
    logger.info("Test data loaded from test_data.pkl")

    # Make predictions and evaluate
    y_pred = best_model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    logger.info(f"Test Accuracy of the best model: {accuracy}")
    logger.info(f"Predicted labels: {y_pred}")
    logger.info(f"Actual labels: {y_test}")

# use pyfunc method to load the model and test it with the test data
def test_best_model_pyfunc(model_uri):
    # Load the model using pyfunc
    best_model_pyfunc = mlflow.pyfunc.load_model(model_uri)
    logger.info("Best model loaded successfully using pyfunc!")

    # Load test data
    with open("test_data.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)
    logger.info("Test data loaded from test_data.pkl")

    # show the results in the dataframe format
    y_pred = best_model_pyfunc.predict(X_test)
    results_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    results_df["predicted_label"] = y_pred
    results_df["actual_label"] = y_test
    logger.info(f"Results DataFrame:\n{results_df[:10]}")  # Show only the first 10 rows for brevity
    

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    best_model_uri = get_best_run_and_model_uri()
    test_best_model(best_model_uri)
    logger.info(f"{'='*150}")
    test_best_model_pyfunc(best_model_uri)