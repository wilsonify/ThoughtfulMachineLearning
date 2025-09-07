from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

TMP_DIR = "/tmp/king_county"
POSTGRES_CONN_ID = "my_postgres"


def make_backfill_task():
    return SQLExecuteQueryOperator(
        task_id="backfill_sales",
        conn_id=POSTGRES_CONN_ID,
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
        """
    )
