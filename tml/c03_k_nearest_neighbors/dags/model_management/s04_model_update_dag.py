"""
Purpose: Retrain model periodically or when performance drifts.

Tasks:

Monitor error metrics (MAE from your RegressionExperiment).

If threshold exceeded, trigger retraining (reuse model_create logic).

Update model registry with new version.

Optionally deprecate or tag old model.

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
TRAINING_DATA_PATH = f"{TMP_DIR}/king_county_data.csv"
MODEL_DIR = "/mnt/SSD1/mrepos/github.com/wilsonify/ThoughtfulMachineLearning/tml/c03_k_nearest_neighbors/data/models"
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
POSTGRES_CONN_ID = "my_postgres"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="s04_model_update_pipeline",
    default_args=default_args,
    description="Retrain and persist a new KNN model version",
    schedule="@weekly",   # adjust cadence as needed
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def load_training_data(limit: int = None):
        """Load transformed data for training."""
        if not os.path.exists(TRAINING_DATA_PATH):
            raise FileNotFoundError(f"{TRAINING_DATA_PATH} does not exist. Run transform DAG first.")

        df = pd.read_csv(TRAINING_DATA_PATH, nrows=limit)
        X = df.drop(columns=["id", "AppraisedValue"], errors="ignore")
        y = df["AppraisedValue"] if "AppraisedValue" in df.columns else None

        if y is None:
            raise ValueError("Training data must include AppraisedValue column.")

        return {"features": X.to_dict(orient="list"), "target": y.to_list()}

    @task
    def train_model(data: dict, n_neighbors: int = 5):
        """Fit KNNModel and return fitted object + version info."""
        X = pd.DataFrame(data["features"])
        y = pd.Series(data["target"])

        model = KNNModel(k=n_neighbors)
        model.fit(X, y)

        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return {"model": model, "version": version, "params": {"k": n_neighbors}}

    @task
    def serialize_and_archive(model_info: dict):
        """Save trained model to archive folder with versioned path."""
        model = model_info["model"]
        version = model_info["version"]

        version_dir = os.path.join(ARCHIVE_DIR, version)
        os.makedirs(version_dir, exist_ok=True)

        pickle_path = os.path.join(version_dir, f"knn_model_{version}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f)

        return {"version": version, "pickle_path": pickle_path, "params": model_info["params"]}

    insert_metadata = SQLExecuteQueryOperator(
        task_id="insert_metadata",
        conn_id=POSTGRES_CONN_ID,
        sql="""
        INSERT INTO models (version, parameters, created_at)
        VALUES (
            '{{ ti.xcom_pull(task_ids="serialize_and_archive")["version"] }}',
            '{{ ti.xcom_pull(task_ids="serialize_and_archive")["params"] | tojson }}',
            NOW()
        );
        """,
    )

    # DAG flow
    data = load_training_data()
    model_info = train_model(data)
    archived = serialize_and_archive(model_info)
    archived >> insert_metadata
