import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import bs4
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
    html_content = resp['Body'].read()
    soup = bs4.BeautifulSoup(html_content, features="lxml")
    wines_list = []
    for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
        count += 1
        time.sleep(5)
        link_text = x.attrs["href"]
        wine_url = f"https://www.wine.com/product{link_text}"
        wine_resp = requests.get(wine_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        print(f"got wine_url={wine_url}")
        suffix = link_text.replace("/", "_")
        suffix = suffix.replace("__", "_")
        suffix = suffix.strip("_")
        wines_list.append({
            "wine_url": wine_url,
            "link_text": link_text
        })
        s3_client.put_object(
            Body=wine_resp.text,
            Bucket=OUTPUT_BUCKET,
            Key=f"{OUTPUT_PREFIX}/{today_date_str}/html/wines/{suffix}.html"
        )

    page_dict = {
        "page_key": page_key,
        "wines": wines_list
    }
    page_str = json.dumps(page_dict)
    page_bytes = page_str.encode("utf-8")
    s3_client.put_object(
        Body=page_bytes,
        Bucket=OUTPUT_BUCKET,
        Key=f"{OUTPUT_PREFIX}/{today_date_str}/json/pages/{page_key}.json"
    )


def main_scrape_wine_parallel():
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    objs = list_objects_s3(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{today_date_str}/html/pages",
        glob_pattern='*.html'
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit each page crawl task to the thread pool executor
        for obj in objs:
            print(obj)
            executor.submit(main_scrape_wine_one_page, obj)
