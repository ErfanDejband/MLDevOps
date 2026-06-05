import mlflow
import argparse

# Delete experiment by name, we need delete it form the mlflow.db directly
# get the experiment name as argument and delete it from mlflow.db
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete MLflow experiment permanently")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to delete")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is not None:
        mlflow.delete_experiment(experiment.experiment_id)
        print(f"Experiment '{args.experiment_name}' deleted permanently.")
    else:
        print(f"Experiment '{args.experiment_name}' not found.")
    # now use sqlite3 to delete the experiment from mlflow.db
    import sqlite3
    conn = sqlite3.connect("mlflow.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE name=?", (args.experiment_name,))
    conn.commit()
    conn.close()
    print(f"Experiment '{args.experiment_name}' deleted from mlflow.db.")