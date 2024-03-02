from pprint import pprint

from io_library.read_from_s3 import read_csv_from_s3
from s03_wine_fit import INPUT_BUCKET, INPUT_PREFIX


def happy_path(event):
    df = read_csv_from_s3(bucket=INPUT_BUCKET, key=f"{INPUT_PREFIX}/csv/train.csv")

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

INPUT_BUCKET="064592191516-kaggle"
OUTPUT_BUCKET="064592191516-kaggle"
INPUT_PREFIX = "s02-create-dataset"
OUTPUT_PREFIX="s03-fit"

df = read_csv_from_s3(bucket=INPUT_BUCKET, key=f"{INPUT_PREFIX}/csv/train.csv")

def lambda_handler(event, context):
    print("event")
    pprint(event)

    print("context")
    pprint(context)

    print("start main")
    happy_path(event)
    print("done main")

    response = {
        "statusCode": 200,
        "body": "Success from s02_create_wine_dataset lambda"
    }
    return response
