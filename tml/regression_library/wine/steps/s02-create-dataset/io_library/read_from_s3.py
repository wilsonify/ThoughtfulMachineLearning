import json
from io import BytesIO

import pandas as pd
from boto3 import Session


def read_csv_from_s3(bucket, key):
    s3_session = Session()
    s3_client = s3_session.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response['Body']
    body_bytes = body.read()
    buffer = BytesIO(body_bytes)
    df = pd.read_csv(buffer, index_col=0)
    return df


def read_parquet_from_s3(bucket, key):
    print(f"read_parquet_from_s3 {bucket}/{key}")
    s3_session = Session()
    s3_client = s3_session.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response['Body']
    body_bytes = body.read()
    buffer = BytesIO(body_bytes)
    df = pd.read_parquet(buffer)
    return df


def read_dict_from_json_s3(bucket: str, key: str):
    """
    Load a dictionary from a .json file in an S3 bucket.

    Parameters:
        bucket (str): S3 bucket name.
        key (str): S3 key (file path).
    """
    s3_session = Session()
    s3_client = s3_session.client('s3')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    json_bytes = obj['Body'].read()
    json_str = json_bytes.decode('utf-8')
    params = json.loads(json_str)
    return params
