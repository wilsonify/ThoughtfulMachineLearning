"""
Purpose: Train and persist a new model version.

Tasks:

Load training data (from your transformed CSV or staging tables).

Train a KNNModel (or any model defined in regression.py).

Serialize model artifact (pickle or ONNX).

Push model metadata to a models table in Postgres (version, parameters, date).

Archive artifact to /data/models/{timestamp}/.
"""


from datetime import datetime
import os
import pickle
import shutil

from airflow import DAG
from airflow.decorators import task
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

import pandas as pd
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from tml.c03_k_nearest_neighbors.regression import DatasetManager, KNNModel

# Paths
TMP_DIR = "/tmp/king_county"
TRANSFORMED_PATH = f"{TMP_DIR}/king_county_data.csv"
MODEL_DIR = "/mnt/SSD1/mrepos/github.com/wilsonify/ThoughtfulMachineLearning/tml/c03_k_nearest_neighbors/data/models"
POSTGRES_CONN_ID = "my_postgres"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="model_create_pipeline",
    default_args=default_args,
    description="Train and persist a new KNN model version",
    schedule=None,  # run manually or triggered
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def load_training_data(limit: int = None):
        if not os.path.exists(TRANSFORMED_PATH):
            raise FileNotFoundError(f"{TRANSFORMED_PATH} does not exist. Run transform DAG first.")
        dataset = DatasetManager()
        dataset.create_from_csv(TRANSFORMED_PATH, limit=limit)
        return {
            "houses": dataset.houses.to_dict(),
            "values": dataset.values.to_dict(),
        }

    @task
    def train_and_serialize(data: dict, k: int = 5):
        # Rehydrate dataset
        houses = pd.DataFrame(data["houses"])
        values = pd.Series(data["values"])

        # Train model
        model = KNNModel(k=k)
        model.create_from_dataset(houses, values)

        # Serialize Pickle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(MODEL_DIR, exist_ok=True)
        pickle_path = os.path.join(MODEL_DIR, f"knn_model_{timestamp}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f)

        # Serialize ONNX (use a dummy sklearn KNN for compatibility)
        from sklearn.neighbors import KNeighborsRegressor
        sk_model = KNeighborsRegressor(n_neighbors=k)
        sk_model.fit(houses, values)
        initial_type = [("float_input", FloatTensorType([None, houses.shape[1]]))]
        onnx_model = convert_sklearn(sk_model, initial_types=initial_type)
        onnx_path = os.path.join(MODEL_DIR, f"knn_model_{timestamp}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        return {
            "version": timestamp,
            "k": k,
            "pickle_path": pickle_path,
            "onnx_path": onnx_path,
        }

    insert_metadata = SQLExecuteQueryOperator(
        task_id="insert_model_metadata",
        conn_id=POSTGRES_CONN_ID,
        sql="""
        INSERT INTO models (version, created_at, parameters, artifact_path)
        VALUES ('{{ ti.xcom_pull(task_ids="train_and_serialize")["version"] }}',
                NOW(),
                '{"k": {{ ti.xcom_pull(task_ids="train_and_serialize")["k"] }}}',
                '{{ ti.xcom_pull(task_ids="train_and_serialize")["pickle_path"] }}');
        """,
    )

    @task
    def archive_artifact(paths: dict):
        archive_dir = os.path.join(MODEL_DIR, "archive", paths["version"])
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copy2(paths["pickle_path"], archive_dir)
        shutil.copy2(paths["onnx_path"], archive_dir)
        return archive_dir

    # DAG flow
    data = load_training_data()
    artifacts = train_and_serialize(data)
    artifacts >> insert_metadata
    archive_artifact(artifacts)
