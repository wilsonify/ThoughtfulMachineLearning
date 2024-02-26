import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import boto3
import bs4
import pandas as pd
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from io_library.write_to_s3 import write_csv_to_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX


def main_page_html_to_json(page_key):
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
    html_content = resp['Body'].read()
    soup = bs4.BeautifulSoup(html_content, features="lxml")
    wines_list = []
    for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
        count += 1
        time.sleep(5)
        link_text = x.attrs["href"]
        wine_url = f"https://www.wine.com/product{link_text}"
        suffix = link_text.replace("/", "_")
        suffix = suffix.replace("__", "_")
        suffix = suffix.strip("_")
        print(f"found wine_url={wine_url}")
        wine_key = f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        wines_list.append({
            "page_key": page_key,
            "wine_url": wine_url,
            "link_text": link_text,
            "suffix": suffix,
            "wine_key": wine_key
        })
    page_str = json.dumps(wines_list)
    page_bytes = page_str.encode("utf-8")
    s3_client.put_object(
        Body=page_bytes,
        Bucket=OUTPUT_BUCKET,
        Key=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages/{page_tail}.json"
    )
    df = pd.DataFrame.from_records(wines_list)
    write_csv_to_s3(
        df=df,
        bucket=OUTPUT_BUCKET,
        key=f"{OUTPUT_PREFIX}/{today_date_str}/csv/pages/{page_tail}.csv"
    )


def main_page_html_to_json_parallel():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/html/pages",
        glob_pattern='*.html'
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit each page crawl task to the thread pool executor
        for obj in objs:
            print(obj)
            executor.submit(main_page_html_to_json, obj)


def enqueue_page_html_to_json():
    sqs = boto3.client('sqs')
    queue_url = 'wine-sqs-try'
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/html/pages",
        glob_pattern='*.html'
    )
    for obj in objs:
        message_dict = {
            "strategy": "main_page_html_to_json",
            "page_key": obj
        }
        message_str = json.dumps(message_dict)
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_str
        )


if __name__ == "__main__":
    # main_page_html_to_json("wine/s01-scrape/2024-02-25/html/pages/page_01.html")
    main_page_html_to_json_parallel()
