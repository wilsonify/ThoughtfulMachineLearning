from airflow import DAG
from datetime import datetime

from tml.c03_k_nearest_neighbors.dags.s01_download_datasets import make_download_tasks
from tml.c03_k_nearest_neighbors.dags.s02_load_stage import make_load_tasks
from tml.c03_k_nearest_neighbors.dags.s03_transform import make_transform_task
from tml.c03_k_nearest_neighbors.dags.s05_backfill_slowly import make_backfill_task

# Data source URLs and paths

PARCEL_URL = "https://data.kingcounty.gov/api/views/qzjh-2p79/rows.csv?accessType=DOWNLOAD"
RESBLDG_URL = "https://data.kingcounty.gov/api/views/f29f-zza5/rows.csv?accessType=DOWNLOAD"
ACCT_URL = "https://data.kingcounty.gov/api/views/i2sg-4vkb/rows.csv?accessType=DOWNLOAD"
SALE_URL = "https://data.kingcounty.gov/api/views/nu5z-2fpr/rows.csv?accessType=DOWNLOAD"

TMP_DIR = "/tmp/king_county"
PARCEL_PATH = f"{TMP_DIR}/EXTR_Parcel.csv"
RESBLDG_PATH = f"{TMP_DIR}/EXTR_ResBldg.csv"
ACCT_PATH = f"{TMP_DIR}/EXTR_RPAcct_NoName.csv"
SALE_PATH = f"{TMP_DIR}/EXTR_RPSale.csv"
OUTPUT_PATH = f"{TMP_DIR}/king_county_data.csv"

POSTGRES_CONN_ID = "my_postgres"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    "king_county_pipeline",
    default_args=default_args,
    description="ETL pipeline for King County Assessor data",
    schedule_interval="@monthly",
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    downloads = make_download_tasks()
    loads = make_load_tasks()
    transform = make_transform_task()
    backfill = make_backfill_task()

    # dependencies
    [downloads["parcel"], downloads["resbldg"], downloads["acct"], downloads["sale"]] >> [
        loads["parcel"], loads["resbldg"], loads["acct"], loads["sale"]
    ] >> transform >> backfill
