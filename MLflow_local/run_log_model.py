import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
import logging
import warnings
import pickle

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Only show the message, no logger name or level
)


def setup_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("MLflow_Local_Experiment")
    logger.info("MLflow setup complete. Tracking URI: http://127.0.0.1:5000")

def load_data(validation_size):
    X, y = datasets.load_iris(return_X_y=True)
    random_state = 42
    
    # First split: 90% train+val, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state
    )
    
    # Second split: 70% train, 20% validation (from the 90%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_size, random_state=random_state 
    )
    
    logger.info(f"Data loaded and split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def define_and_train_model(X_train, y_train, X_val, y_val, model_parameters, show_labels=False):
    model = LogisticRegression(**model_parameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    logger.info(f"\tValidation Accuracy: {accuracy}")
    if show_labels:
        logger.info(f"\tPredicted labels: {y_pred}")
        logger.info(f"\tActual labels: {y_val}")
    return model, y_pred, accuracy

if __name__ == "__main__":
    setup_mlflow()
    random_state = 42
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(validation_size=0.2)

    with open("test_data.pkl", "wb") as f:
        pickle.dump((X_test, y_test), f)
    logger.info("Test data saved to test_data.pkl (this will save in your CI/CD storage)")
    
    for iter in range(2, 15):
        model_parameters = {"solver": "lbfgs",
                            "random_state": random_state,
                            "max_iter": iter,
                            "l1_ratio": 0}
        logger.info("=" * 50)
        logger.info(f"Training model with parameters: {model_parameters}")
        model, y_pred, accuracy = define_and_train_model(X_train, y_train, X_val, y_val, model_parameters)
        with mlflow.start_run():
            mlflow.log_params(model_parameters)
            mlflow.log_metric("validation_accuracy", accuracy)
            mlflow.set_tag("model_type", "Logistic Regression")
            signature = infer_signature(X_train, y_pred)
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="iris_model",
                signature=signature,
                input_example=X_train[:5],
                registered_model_name="iris_logistic_regression_model")

    