import mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature


mlflow.set_tracking_uri("http://127.0.0.1:5000")
if not mlflow.get_experiment_by_name("ML_Experiment"):
    exp = mlflow.create_experiment("ML_Experiment")
    print(f"Experiment created with id: {exp}")
else:
    exp = mlflow.get_experiment_by_name("ML_Experiment")
    mlflow.set_experiment(exp.name)
    print(f"Experiment already exists with id: {exp.experiment_id}")


if __name__ == "__main__":
    X,y= datasets.load_iris(return_X_y=True)
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
    #define logestic regression model parameters
    for iter in range(2, 4):
        model_parameters = {"solver": "lbfgs","random_state": random_state,"max_iter": iter, "l1_ratio": 0}
        model = LogisticRegression(**model_parameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        print(f"predicted labels: {y_pred}")
        print(f"actual labels: {y_test}")

        with mlflow.start_run():
            mlflow.log_params(model_parameters)
            mlflow.log_metric("accuracy", accuracy)
            signature = infer_signature(X_train, y_pred)
            model_info=mlflow.sklearn.log_model(
                sk_model=model,
                name="iris_model",
                signature=signature,
                input_example=X_train,
                registered_model_name=f"Iris_Logistic_Regression_Model_{iter}"
            )
    