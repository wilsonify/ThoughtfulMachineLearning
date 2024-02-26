import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import boto3
import requests
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX


def main_scrape_wine_one_page(page_key):
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
    for wine in wines_list:
        count += 1
        time.sleep(5)
        wine_url = wine["wine_url"]
        suffix = wine["suffix"]
        wine_key = f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        wine_resp = requests.get(wine_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        print(f"got wine_url={wine_url}")
        s3_client.put_object(
            Body=wine_resp.text,
            Bucket=OUTPUT_BUCKET,
            Key=wine_key
        )


def main_scrape_wine_parallel():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages",
        glob_pattern='*.json'
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit each page crawl task to the thread pool executor
        for obj in objs:
            print(obj)
            executor.submit(main_scrape_wine_one_page, obj)


def enqueue_scrape_wine_one_page():
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
            "strategy": "main_scrape_wine_one_page",
            "page_key": obj
        }
        message_str = json.dumps(message_dict)
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_str
        )
