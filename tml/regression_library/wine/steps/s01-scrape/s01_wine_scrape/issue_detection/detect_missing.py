import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from io_library.read_from_s3 import read_dict_from_json_s3
from io_library.write_to_s3 import write_csv_to_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX


def check_for_json(count, wine_key_json):
    try:
        read_dict_from_json_s3(bucket=OUTPUT_BUCKET, key=wine_key_json)
        count += 1
        print(f"found wine_key_json = {wine_key_json}")
        has_json = True
    except:
        print(f"missing wine_key_json = {wine_key_json}")
        has_json = False
    return has_json


def check_for_html(wine_key_html):
    try:
        s3_session = Session()
        s3_client = s3_session.client('s3')
        s3_client.get_object(Bucket=OUTPUT_BUCKET, Key=wine_key_html)
        has_html = True
    except:
        has_html = False
    return has_html


def main_detect_missing(page_key):
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
        link_text = wine["link_text"]
        wine_url = wine["wine_url"]
        suffix = wine["suffix"]
        wine_key_html = f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        has_html = check_for_html(wine_key_html)
        wine_key_json = f"{OUTPUT_PREFIX}/{today_date_str}/json/wines/{suffix}.json"
        print(f"wine_key_json = {wine_key_json}")
        has_json = check_for_json(count, wine_key_json)
        dfs_to_concat.append(pd.DataFrame(dict(
            page_key=[page_key],
            wine_url=[wine_url],
            link_text=[link_text],
            suffix=[suffix],
            wine_key=[wine_key_json],
            has_html=[has_html],
            has_json=[has_json]
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


def main_detect_missing_parallel():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages",
        glob_pattern='*.json'
    )
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(main_detect_missing, objs)


if __name__ == "__main__":
    # main_detect_missing(f"wine/s01-scrape/2024-02-25/json/pages/page_01.json")
    main_detect_missing_parallel()
