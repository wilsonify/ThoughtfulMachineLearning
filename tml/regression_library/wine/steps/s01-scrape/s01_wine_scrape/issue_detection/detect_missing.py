import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

import boto3
import pandas as pd
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from io_library.read_from_s3 import read_dict_from_json_s3, read_csv_from_s3, read_html_from_s3
from io_library.write_to_s3 import write_csv_to_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX


def check_for_csv(page_df, wine_key):
    is_wine_of_interest = page_df['wine_url'] == wine_key
    is_wine_of_interest_any = is_wine_of_interest.any()
    if is_wine_of_interest_any:
        has_csv = True
        print(f"found wine_key = {wine_key} in page csv")
    else:
        print(f"missing wine_key = {wine_key} in page csv")
        has_csv = False
    return has_csv


def check_for_json(wine_key_json):
    try:
        read_dict_from_json_s3(bucket=OUTPUT_BUCKET, key=wine_key_json)
        print(f"found wine_key_json = {wine_key_json}")
        has_json = True
    except:
        print(f"missing wine_key_json = {wine_key_json}")
        has_json = False
    return has_json


def check_for_html(wine_key_html):
    try:
        soup = read_html_from_s3(OUTPUT_BUCKET, wine_key_html)
        print(f"found wine_key_html = {wine_key_html}")
        has_html = True
    except:
        print(f"missing wine_key_html = {wine_key_html}")
        has_html = False
    return has_html


def main_detect_missing(page_key, today_date_str=None):
    print(f"page_key = {page_key}")
    if today_date_str is None:
        today_date_str = datetime.now().strftime('%Y-%m-%d')
    page_root, page_ext = os.path.splitext(page_key)
    page_head, page_tail = os.path.split(page_root)

    s3_session = Session()
    s3_client = s3_session.client('s3')
    count = 0
    resp = s3_client.get_object(
        Bucket=OUTPUT_BUCKET,
        Key=page_key
    )
    json_content = resp['Body'].read()
    wines_list = json.loads(json_content)

    page_key_csv = f"{OUTPUT_PREFIX}/{today_date_str}/csv/pages/{page_tail}.csv"
    page_df = read_csv_from_s3(OUTPUT_BUCKET, page_key_csv)

    dfs_to_concat = []
    for wine in wines_list:
        link_text = wine["link_text"]
        wine_url = wine["wine_url"]
        suffix = wine["suffix"]
        wine_key = wine["wine_key"]
        wine_key_html = f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        has_html = check_for_html(wine_key_html)
        wine_key_json = f"{OUTPUT_PREFIX}/{today_date_str}/json/wines/{suffix}.json"
        print(f"wine_key_json = {wine_key_json}")
        has_json = check_for_json(wine_key_json)
        wine_key_csv = f"{OUTPUT_PREFIX}/{today_date_str}/csv/wines/{suffix}.csv"
        has_csv = check_for_csv(page_df, wine_key)
        dfs_to_concat.append(pd.DataFrame(dict(
            page_key=[page_key],
            wine_url=[wine_url],
            link_text=[link_text],
            suffix=[suffix],
            wine_key=[wine_key_json],
            has_html=[has_html],
            has_json=[has_json],
            has_csv=[has_csv]
        )))
    df = pd.concat(dfs_to_concat)
    df = df.reset_index(drop=True)
    df.index.name = "index"
    print(f"df.shape = {df.shape}")
    write_csv_to_s3(
        df=df,
        bucket=OUTPUT_BUCKET,
        key=f"{OUTPUT_PREFIX}/{today_date_str}/csv/reports/{page_tail}_missing.csv"
    )


def main_detect_missing_parallel(today_date_str=None):
    if today_date_str is None:
        today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET, prefix=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages", glob_pattern='*.json'
    )
    main_detect_missing_part = partial(main_detect_missing, today_date_str=today_date_str)
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(main_detect_missing_part, objs)


def enqueue_detect_missing():
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
            "strategy": "main_detect_missing",
            "page_key": obj
        }
        message_str = json.dumps(message_dict)
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_str
        )


if __name__ == "__main__":
    # main_detect_missing(f"wine/s01-scrape/2024-03-02/json/pages/page_01.json")
    main_detect_missing_parallel()
    # enqueue_detect_missing()
