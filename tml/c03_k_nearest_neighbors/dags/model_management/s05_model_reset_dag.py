"""
Purpose: Reset to baseline state (useful for experiments or corruption recovery).

Tasks:

Drop current model tables/registry entries.

Re-initialize database schema for models/predictions.

Archive and move existing artifacts to reset/{timestamp}/.
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
EVAL_DATA_PATH = f"{TMP_DIR}/king_county_data.csv"
MODEL_DIR = "/mnt/SSD1/mrepos/github.com/wilsonify/ThoughtfulMachineLearning/tml/c03_k_nearest_neighbors/data/models"
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")
POSTGRES_CONN_ID = "my_postgres"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="s05_model_reset_pipeline",
    default_args=default_args,
    description="Compare current model vs baseline, reset if baseline is better",
    schedule="@weekly",
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def load_eval_data(limit: int = 200):
        """Load evaluation data (hold-out set or fresh sample)."""
        if not os.path.exists(EVAL_DATA_PATH):
            raise FileNotFoundError(f"{EVAL_DATA_PATH} not found. Run transform DAG first.")

        df = pd.read_csv(EVAL_DATA_PATH, nrows=limit)
        X = df.drop(columns=["id", "AppraisedValue"], errors="ignore")
        y = df["AppraisedValue"]
        return {"features": X.to_dict(orient="list"), "target": y.to_list()}

    @task
    def get_latest_model():
        """Load the most recent archived model."""
        if not os.path.exists(ARCHIVE_DIR):
            raise FileNotFoundError("No archived models found.")

        versions = sorted(os.listdir(ARCHIVE_DIR))
        if not versions:
            raise FileNotFoundError("No model versions in archive.")

        latest_version = versions[-1]
        pickle_path = os.path.join(ARCHIVE_DIR, latest_version, f"knn_model_{latest_version}.pkl")
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)

        return {"version": latest_version, "model": model, "pickle_path": pickle_path}

    @task
    def load_baseline_model():
        """Load baseline model (mean/median predictor or fixed reference)."""
        if not os.path.exists(BASELINE_PATH):
            raise FileNotFoundError(f"Baseline model missing at {BASELINE_PATH}")

        with open(BASELINE_PATH, "rb") as f:
            model = pickle.load(f)

        return {"model": model, "path": BASELINE_PATH}

    @task
    def evaluate_models(eval_data: dict, latest: dict, baseline: dict):
        """Compute RMSE for both latest and baseline models."""
        X = pd.DataFrame(eval_data["features"])
        y = np.array(eval_data["target"])

        # Latest model predictions
        preds_latest = np.array([latest["model"].predict(row) for _, row in X.iterrows()])
        rmse_latest = np.sqrt(np.mean((y - preds_latest) ** 2))

        # Baseline predictions
        preds_base = np.array([baseline["model"].predict(row) for _, row in X.iterrows()])
        rmse_base = np.sqrt(np.mean((y - preds_base) ** 2))

        print(f"Latest model RMSE: {rmse_latest:.3f}")
        print(f"Baseline model RMSE: {rmse_base:.3f}")

        return {
            "latest_version": latest["version"],
            "rmse_latest": float(rmse_latest),
            "rmse_baseline": float(rmse_base),
            "reset_needed": rmse_base < rmse_latest,
        }

    reset_metadata = SQLExecuteQueryOperator(
        task_id="reset_metadata",
        conn_id=POSTGRES_CONN_ID,
        sql="""
        INSERT INTO model_resets (reset_to, reason, created_at)
        VALUES (
            'baseline',
            'Baseline RMSE better than latest',
            NOW()
        );
        """,
        trigger_rule="none_failed_min_one_success",
    )

    @task(trigger_rule="all_success")
    def reset_model(evaluation: dict, baseline: dict):
        """If baseline is better, copy baseline into 'current' slot."""
        if evaluation["reset_needed"]:
            current_path = os.path.join(MODEL_DIR, "current_model.pkl")
            with open(baseline["path"], "rb") as fsrc, open(current_path, "wb") as fdst:
                fdst.write(fsrc.read())
            print("Reset: baseline promoted to current production model")
        else:
            print("No reset: latest model is still better")

    # DAG flow
    eval_data = load_eval_data()
    latest = get_latest_model()
    baseline = load_baseline_model()
    evaluation = evaluate_models(eval_data, latest, baseline)
    reset = reset_model(evaluation, baseline)
    evaluation >> reset_metadata
    evaluation >> reset
