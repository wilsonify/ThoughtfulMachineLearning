import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import bs4
from boto3 import Session

from io_library.list_objects_s3 import list_objects_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX
from s01_wine_scrape.extract import extract_meta, extract_prodAlcoholVolume, extract_prodAlcoholPercent, extract_ratings


def main_scrape_one_wine(wine_key):
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    wine_root, wine_ext = os.path.splitext(wine_key)
    wine_head, wine_tail = os.path.split(wine_root)
    s3_session = Session()
    s3_client = s3_session.client('s3')
    resp = s3_client.get_object(
        Bucket=OUTPUT_BUCKET,
        Key=wine_key
    )
    html_content = resp['Body'].read()
    wine_soup = bs4.BeautifulSoup(html_content, features="lxml")
    data = {}
    data["wine_url"] = wine_key
    extract_meta(data, wine_soup)
    extract_prodAlcoholVolume(data, wine_soup)
    extract_prodAlcoholPercent(data, wine_soup)
    extract_ratings(data, wine_soup)
    data_str = json.dumps(data)
    data_bytes = data_str.encode("utf-8")
    s3_client.put_object(
        Body=data_bytes,
        Bucket=OUTPUT_BUCKET,
        Key=f"{OUTPUT_PREFIX}/{today_date_str}/json/wines/{wine_tail}.json"
    )


def main_scrape_wine_parallel_html_to_json():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/html/wines",
        glob_pattern='*.html'
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        # parse each page from html to json
        for obj in objs:
            print(obj)
            executor.submit(main_scrape_one_wine, obj)
