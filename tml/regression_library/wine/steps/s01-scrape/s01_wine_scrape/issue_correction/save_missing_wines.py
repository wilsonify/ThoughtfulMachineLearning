import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from io import BytesIO

import boto3
import pandas as pd
import requests
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX
from s01_wine_scrape.pa_wines.create.s04_wine_html_to_json import main_scrape_one_wine
from s01_wine_scrape.pa_wines.create.s05_wine_json_to_csv import main_json_to_csv


def main_correct_missing_one_page(page_key, today_date_str=None):
    if today_date_str is None:
        today_date_str = datetime.now().strftime('%Y-%m-%d')
    s3_session = Session()
    s3_client = s3_session.client('s3')
    count = 0
    resp = s3_client.get_object(
        Bucket=OUTPUT_BUCKET,
        Key=page_key
    )
    csv_content = resp['Body'].read()
    missing_df = pd.read_csv(BytesIO(csv_content))
    missing_html_df = missing_df[~missing_df["has_html"]]

    for ind, row in missing_html_df.iterrows():
        count += 1
        time.sleep(5)
        wine_url = row["wine_url"]
        suffix = row["suffix"]
        wine_key = f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        wine_resp = requests.get(wine_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        print(f"got wine_url={wine_url}")
        s3_client.put_object(
            Body=wine_resp.text,
            Bucket=OUTPUT_BUCKET,
            Key=wine_key
        )

    missing_json_df = missing_df[~missing_df["has_json"]]
    for ind, row in missing_json_df.iterrows():
        suffix = row["suffix"]
        wine_key = f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        print(f"wine_key = {wine_key}")
        main_scrape_one_wine(wine_key)

    missing_json_df = missing_df[~missing_df["has_csv"]]
    if missing_json_df.shape[0] > 0:
        main_json_to_csv(page_key)


def main_correct_missing_parallel(today_date_str=None):
    if today_date_str is None:
        today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/csv/reports",
        glob_pattern='*_missing.csv'
    )
    main_correct_missing_one_page_part = partial(main_correct_missing_one_page, today_date_str=today_date_str)
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(main_correct_missing_one_page, objs)


def enqueue_correct_missing():
    sqs = boto3.client('sqs')
    queue_url = 'wine-sqs-try'
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/csv/reports",
        glob_pattern='*_missing.csv'
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
    logging.basicConfig(level="DEBUG")
    # main_correct_missing_one_page("wine/s01-scrape/2024-03-02/csv/reports/page_01_missing.csv")
    main_correct_missing_parallel()
