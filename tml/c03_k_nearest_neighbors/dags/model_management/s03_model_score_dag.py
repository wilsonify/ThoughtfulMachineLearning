"""
model_score_dag.py

Purpose: Scheduled scoring of fresh data.

Tasks:

Download or receive new raw data.

Apply the transform DAG (s03_transform).

Load most recent model.

Score and insert results into a serving layer (e.g., Postgres or an API).
"""

from datetime import datetime
import os
import pickle
import pandas as pd

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
    dag_id="s03_model_score_pipeline",
    default_args=default_args,
    description="Scheduled scoring DAG using latest trained model",
    schedule="@daily",   # or @hourly, adjust as needed
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
    def load_new_data(limit: int = 100):
        """Load fresh data to score (e.g. new sales appended by ETL DAG)."""
        if not os.path.exists(SCORING_DATA_PATH):
            raise FileNotFoundError(f"{SCORING_DATA_PATH} does not exist. Run transform DAG first.")

        df = pd.read_csv(SCORING_DATA_PATH, nrows=limit)
        ids = df["id"]
        houses = df.drop(columns=["id", "AppraisedValue"], errors="ignore")
        return {"ids": ids.to_list(), "houses": houses.to_dict(orient="list")}

    @task
    def score_data(model_info: dict, data: dict):
        """Apply trained model to new data batch."""
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
        SELECT '{{ ti.xcom_pull(task_ids="score_data")["version"] }}',
               unnest(array{{ ti.xcom_pull(task_ids="score_data")["ids"] }}),
               unnest(array{{ ti.xcom_pull(task_ids="score_data")["predictions"] }}),
               NOW();
        """,
    )

    # DAG flow
    model_info = get_latest_model()
    data = load_new_data()
    preds = score_data(model_info, data)
    preds >> insert_predictions
