from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

TMP_DIR = "/tmp/king_county"
OUTPUT_PATH = f"{TMP_DIR}/king_county_data.csv"
POSTGRES_CONN_ID = "my_postgres"


def make_transform_task():
    return SQLExecuteQueryOperator(
        task_id="transform",
        conn_id=POSTGRES_CONN_ID,
        sql=f"""
        COPY (
            SELECT
                CONCAT(ht."Major", ht."Minor") AS "id",
                hl."Address",
                ht."ApprLandVal" + ht."ApprImpsVal" AS "AppraisedValue",
                ht."TaxableLandVal" + ht."TaxableImpsVal" AS "TaxableValue",
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
        """
    )
