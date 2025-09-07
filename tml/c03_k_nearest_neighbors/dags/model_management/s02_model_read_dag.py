"""
Purpose: Serve or validate predictions from an existing model.

Tasks:

Pull latest (or specific) model artifact from the archive.

Load a batch of scoring data (e.g., new house sales).

Run predictions.

Store predictions in a predictions table.

Optionally run drift checks (compare current input distributions vs training).
"""

from datetime import datetime
import os
import pickle
import pandas as pd
import numpy as np

from airflow import DAG
from airflow.decorators import task
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from tml.c03_k_nearest_neighbors.regression import KNNModel

# Paths
TMP_DIR = "/tmp/king_county"
SCORING_DATA_PATH = f"{TMP_DIR}/king_county_data.csv"
MODEL_DIR = "/mnt/SSD1/mrepos/github.com/wilsonify/ThoughtfulMachineLearning/tml/c03_k_nearest_neighbors/data/models"
POSTGRES_CONN_ID = "my_postgres"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="model_read_pipeline",
    default_args=default_args,
    description="Serve or validate predictions from an existing model",
    schedule=None,  # run on demand
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def get_latest_model():
        """Find the most recent pickle model in the archive."""
        archive_path = os.path.join(MODEL_DIR, "archive")
        if not os.path.exists(archive_path):
            raise FileNotFoundError("No model archive found. Run model_create_dag first.")

        versions = sorted(os.listdir(archive_path))
        if not versions:
            raise FileNotFoundError("No model versions found in archive.")

        latest_version = versions[-1]
        pickle_path = os.path.join(archive_path, latest_version, f"knn_model_{latest_version}.pkl")
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle model not found: {pickle_path}")

        return {"version": latest_version, "pickle_path": pickle_path}

    @task
    def load_scoring_data(limit: int = 50):
        """Load batch of data for scoring (new sales or sample)."""
        if not os.path.exists(SCORING_DATA_PATH):
            raise FileNotFoundError(f"{SCORING_DATA_PATH} does not exist. Run transform DAG first.")

        df = pd.read_csv(SCORING_DATA_PATH, nrows=limit)
        ids = df["id"]
        houses = df.drop(columns=["id", "AppraisedValue"], errors="ignore")
        return {"ids": ids.to_list(), "houses": houses.to_dict(orient="list")}

    @task
    def run_predictions(model_info: dict, data: dict):
        """Load model and run predictions."""
        with open(model_info["pickle_path"], "rb") as f:
            model: KNNModel = pickle.load(f)

        houses = pd.DataFrame(data["houses"])
        predictions = []
        for _, row in houses.iterrows():
            predictions.append(float(model.predict(row)))

        return {
            "version": model_info["version"],
            "ids": data["ids"],
            "predictions": predictions,
        }

    insert_predictions = SQLExecuteQueryOperator(
        task_id="insert_predictions",
        conn_id=POSTGRES_CONN_ID,
        sql="""
        INSERT INTO predictions (model_version, house_id, prediction, created_at)
        SELECT '{{ ti.xcom_pull(task_ids="run_predictions")["version"] }}',
               unnest(array{{ ti.xcom_pull(task_ids="run_predictions")["ids"] }}),
               unnest(array{{ ti.xcom_pull(task_ids="run_predictions")["predictions"] }}),
               NOW();
        """,
    )

    @task
    def drift_check(data: dict, predictions: dict):
        """Naive drift check: compare mean feature values."""
        houses = pd.DataFrame(data["houses"])
        feature_means = houses.mean().to_dict()
        pred_mean = np.mean(predictions["predictions"])

        # For now, just log differences
        print("Feature means:", feature_means)
        print("Prediction mean:", pred_mean)

        # TODO: Persist drift metrics in a monitoring table
        return {"feature_means": feature_means, "prediction_mean": pred_mean}

    # DAG flow
    model_info = get_latest_model()
    data = load_scoring_data()
    preds = run_predictions(model_info, data)
    preds >> insert_predictions
    drift_check(data, preds)
