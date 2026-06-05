import mlflow
from mlflow import validate_evaluation_results


mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = mlflow.get_experiment_by_name("ML_Experiment")
print(f"Experiment already exists with id: {exp.experiment_id}")
models = mlflow.search_runs(experiment_ids=exp.experiment_id)
print(models[["run_id", "metrics.accuracy", "params.max_iter"]])
# max accuracy model
best_model = models.sort_values("metrics.accuracy", ascending=False).iloc[0]
print(f"Best model run id: {best_model.run_id}, accuracy: {best_model['metrics.accuracy']}")  
best_model_info = mlflow.get_run(best_model.run_id)
print(f"Best model parameters: {best_model_info.data.params}")

# use input_example.jason of artifact of the best model to validate the model
input_example_path = f"mlruns/{exp.experiment_id}/{best_model.run_id}/artifacts/iris_model/input_example.json"
input_example = mlflow.artifacts.load_artifact(input_example_path)
print(f"Input example: {input_example}")
