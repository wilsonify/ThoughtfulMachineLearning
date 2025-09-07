from airflow.providers.standard.operators.bash import BashOperator

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


def make_download_tasks():
    return {
        "parcel": BashOperator(
            task_id="download_parcel",
            bash_command=f"mkdir -p {TMP_DIR} && curl -L '{PARCEL_URL}' -o {PARCEL_PATH}",
        ),
        "resbldg": BashOperator(
            task_id="download_resbldg",
            bash_command=f"curl -L '{RESBLDG_URL}' -o {RESBLDG_PATH}",
        ),
        "acct": BashOperator(
            task_id="download_acct",
            bash_command=f"curl -L '{ACCT_URL}' -o {ACCT_PATH}",
        ),
        "sale": BashOperator(
            task_id="download_sale",
            bash_command=f"curl -L '{SALE_URL}' -o {SALE_PATH}",
        ),
    }
