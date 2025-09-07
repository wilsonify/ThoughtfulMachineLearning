import os
import shutil
from datetime import datetime

from airflow.decorators import task

POSTGRES_CONN_ID = "my_postgres"
OUTPUT_PATH = "/tmp/king_county/king_county_data.csv"
LOCAL_ARCHIVE_PATH = "/mnt/SSD1/mrepos/github.com/wilsonify/ThoughtfulMachineLearning/tml/c03_k_nearest_neighbors/data/archive"


def make_archive_task():
    @task
    def ensure_archive_dir():
        os.makedirs(LOCAL_ARCHIVE_PATH, exist_ok=True)
        return LOCAL_ARCHIVE_PATH

    @task
    def archive_csv(archive_dir: str):
        if not os.path.exists(OUTPUT_PATH):
            raise FileNotFoundError(f"{OUTPUT_PATH} does not exist. Run transform task first.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = os.path.join(archive_dir, f"king_county_data_{timestamp}.csv")
        shutil.copy2(OUTPUT_PATH, archive_file)
        return archive_file

    archive_dir = ensure_archive_dir()
    archive_task = archive_csv(archive_dir)
    return archive_task
