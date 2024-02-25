import pandas as pd

from io_library.write_to_s3 import write_csv_to_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX, columns_in_order


def save_progress(data, filename):
    result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
    ensure_dtypes(result)
    write_csv_to_s3(
        result[columns_in_order],
        bucket=OUTPUT_BUCKET,
        key=f"{OUTPUT_PREFIX}/{filename}"
    )
    result[columns_in_order].to_csv(filename, mode='a', header=False)


def save_initial(data, filename):
    result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
    ensure_dtypes(result)
    "wines_ship_to_pa.csv"
    write_csv_to_s3(
        result[columns_in_order],
        bucket=OUTPUT_BUCKET,
        key=f"{OUTPUT_PREFIX}/{filename}"
    )


def ensure_dtypes(result):
    result["price"] = pd.to_numeric(result["price"])
    result["WS"] = pd.to_numeric(result["WS"])
    result["JS"] = pd.to_numeric(result["JS"])
