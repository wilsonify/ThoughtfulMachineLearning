from airflow.providers.postgres.operators.postgres import PostgresOperator
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


def make_backfill_task():
    return PostgresOperator(
        task_id="backfill_sales",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql="""
        INSERT INTO house_sales (tax_id, sale_date, plat_id, sale_price, tax_value)
        SELECT hst."ExciseTaxNbr",
               hst."DocumentDate",
               NULL,
               hst."SalePrice",
               NULL
        FROM house_sales_tmp hst
        LEFT JOIN house_sales hs
          ON hs.tax_id = hst."ExciseTaxNbr"
        WHERE hs.tax_id IS NULL
        ORDER BY hst."DocumentDate" ASC
        LIMIT 1000;
        """,
    )
