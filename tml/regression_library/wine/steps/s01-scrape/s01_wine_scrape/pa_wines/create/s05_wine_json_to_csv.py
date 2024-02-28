import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import boto3
import pandas as pd
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from io_library.read_from_s3 import read_dict_from_json_s3
from io_library.write_to_s3 import write_csv_to_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX


def main_json_to_csv(page_key):
    page_root, page_ext = os.path.splitext(page_key)
    page_head, page_tail = os.path.split(page_root)
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    s3_session = Session()
    s3_client = s3_session.client('s3')
    count = 0
    resp = s3_client.get_object(
        Bucket=OUTPUT_BUCKET,
        Key=page_key
    )
    json_content = resp['Body'].read()
    wines_list = json.loads(json_content)
    dfs_to_concat = []
    for wine in wines_list:
        suffix = wine["suffix"]
        wine_key = f"{OUTPUT_PREFIX}/{today_date_str}/json/wines/{suffix}.json"
        print(f"wine_key = {wine_key}")
        try:
            data_dict = read_dict_from_json_s3(bucket=OUTPUT_BUCKET, key=wine_key)
            count += 1
            print(f"found wine_key = {wine_key}")
        except:
            print(f"could not find wine_key = {wine_key}")
            continue
        dfs_to_concat.append(pd.DataFrame(data_dict, index=[count]))
        print(f"length = {len(dfs_to_concat)}")
    df = pd.concat(dfs_to_concat)
    df = df.reset_index(drop=True)
    df.index.name = "index"
    print(f"df.shape = {df.shape}")
    write_csv_to_s3(df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/{today_date_str}/csv/pages/{page_tail}.csv")


def main_scrape_wine_parallel_pages_json_to_csv():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages",
        glob_pattern='*.json'
    )
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(main_json_to_csv, objs)


def enqueue_json_to_csv():
    sqs = boto3.client('sqs')
    queue_url = 'wine-sqs-try'
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages",
        glob_pattern='*.json'
    )
    for obj in objs:
        message_dict = {
            "strategy": "main_json_to_csv",
            "wine_key": obj
        }
        message_str = json.dumps(message_dict)
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_str
        )


if __name__ == "__main__":
    # main_scrape_wine_parallel_pages_json_to_csv()
    enqueue_json_to_csv()
