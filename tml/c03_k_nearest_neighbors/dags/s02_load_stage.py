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


def make_load_tasks():
    return {
        "parcel": PostgresOperator(
            task_id="load_parcel",
            postgres_conn_id=POSTGRES_CONN_ID,
            sql=f"""
            DROP TABLE IF EXISTS house_parcel_tmp;
            CREATE TABLE house_parcel_tmp (
                "Major" integer,
                "Minor" integer,
                "PropType" varchar,
                "SqFtLot" varchar,
                "WaterSystem" varchar,
                "SewerSystem" varchar,
                "Access" varchar,
                "Topography" varchar,
                "StreetSurface" varchar
            );
            COPY house_parcel_tmp FROM '{PARCEL_PATH}' DELIMITER ',' CSV HEADER;
            """,
        ),
        "resbldg": PostgresOperator(
            task_id="load_resbldg",
            postgres_conn_id=POSTGRES_CONN_ID,
            sql=f"""
            DROP TABLE IF EXISTS house_location_tmp;
            CREATE TABLE house_location_tmp (
                "Major" integer,
                "Minor" integer,
                "Address" varchar,
                "ZipCode" varchar,
                "SqFtTotLiving" varchar,
                "Bedrooms" varchar,
                "BathFullCount" varchar,
                "YrBuilt" varchar,
                "Condition" varchar
            );
            COPY house_location_tmp FROM '{RESBLDG_PATH}' DELIMITER ',' CSV HEADER;
            """,
        ),
        "acct": PostgresOperator(
            task_id="load_acct",
            postgres_conn_id=POSTGRES_CONN_ID,
            sql=f"""
            DROP TABLE IF EXISTS house_taxes_tmp;
            CREATE TABLE house_taxes_tmp (
                "Major" integer,
                "Minor" integer,
                "ApprLandVal" numeric,
                "ApprImpsVal" numeric,
                "TaxableLandVal" numeric,
                "TaxableImpsVal" numeric,
                "BillYr" integer
            );
            COPY house_taxes_tmp FROM '{ACCT_PATH}' DELIMITER ',' CSV HEADER;
            DELETE FROM house_taxes_tmp WHERE "BillYr" < 2015;
            """,
        ),
        "sale": PostgresOperator(
            task_id="load_sale",
            postgres_conn_id=POSTGRES_CONN_ID,
            sql=f"""
            DROP TABLE IF EXISTS house_sales_tmp;
            CREATE TABLE house_sales_tmp (
                "ExciseTaxNbr" bigint,
                "Major" integer,
                "Minor" integer,
                "DocumentDate" date,
                "SalePrice" numeric
            );
            COPY house_sales_tmp FROM '{SALE_PATH}' DELIMITER ',' CSV HEADER;
            DELETE FROM house_sales_tmp WHERE "DocumentDate" < '2015-01-01';
            """,
        ),
    }
