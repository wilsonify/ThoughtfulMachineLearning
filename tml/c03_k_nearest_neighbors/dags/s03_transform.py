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


def make_transform_task():
    return PostgresOperator(
        task_id="transform",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=f"""
        COPY (
            SELECT
              CONCAT(ht."Major", ht."Minor") as "id",
              hl."Address",
              ht."ApprLandVal" + ht."ApprImpsVal" as "AppraisedValue",
              ht."TaxableLandVal" + ht."TaxableImpsVal" as "TaxableValue",
              hp."SqFtLot",
              hp."WaterSystem",
              hp."SewerSystem",
              hp."Access",
              hp."Topography",
              hp."StreetSurface"
            FROM house_taxes_tmp ht
            JOIN house_parcel_tmp hp USING ("Major", "Minor")
            JOIN house_location_tmp hl USING ("Major", "Minor")
            WHERE hp."PropType" = 'R'
        )
        TO '{OUTPUT_PATH}'
        WITH CSV HEADER;
        """,
    )
