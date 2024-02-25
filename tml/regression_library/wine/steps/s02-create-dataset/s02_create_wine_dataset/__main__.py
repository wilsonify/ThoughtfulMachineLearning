from pprint import pprint

import pandas as pd

from io_library.list_objects_s3 import list_objects_s3
from io_library.read_from_s3 import read_csv_from_s3
from io_library.write_to_s3 import write_parquet_to_s3, write_csv_to_s3
from s02_create_wine_dataset import INPUT_BUCKET, INPUT_PREFIX, OUTPUT_BUCKET, OUTPUT_PREFIX


def happy_path(context):
    objs = list_objects_s3(bucket=INPUT_BUCKET, prefix=f"{INPUT_PREFIX}/csv/pages", glob_pattern='*.csv')
    dfs_to_concat = []
    for obj in objs:
        df = read_csv_from_s3(bucket=INPUT_BUCKET, key=obj)
        dfs_to_concat.append(df)
    result_df = pd.concat(dfs_to_concat)
    result_df = result_df.reset_index(drop=True)
    result_df.index.name = "index"

    # Randomly shuffle the data
    result_df = result_df.sample(frac=1, random_state=42)

    # Calculate the sizes for train, test, and validation sets
    total_size = len(result_df)
    test_size = int(total_size * 0.2)  # 20% for testing
    validate_size = int(total_size * 0.05)  # 5% for validation
    train_size = total_size - test_size - validate_size

    # Split the dataset into train, test, and validation sets
    train_df = result_df[:train_size]
    test_df = result_df[train_size:train_size + test_size]
    validate_df = result_df[train_size + test_size:]

    # Write train data to S3
    write_csv_to_s3(df=train_df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/csv/train.csv")
    write_parquet_to_s3(df=train_df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/parquet/train.parquet")

    # Write test data to S3
    write_csv_to_s3(df=test_df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/csv/test.csv")
    write_parquet_to_s3(df=test_df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/parquet/test.parquet")

    # Write validation data to S3
    write_csv_to_s3(df=validate_df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/csv/validate.csv")
    write_parquet_to_s3(df=validate_df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/parquet/validate.parquet")


def lambda_handler(event, context):
    print("event")
    pprint(event)

    print("context")
    pprint(context)

    print("start main")
    happy_path(context)
    print("done main")

    response = {
        "statusCode": 200,
        "body": "Success from s02_create_wine_dataset lambda"
    }
    return response


if __name__ == "__main__":
    happy_path({})
