import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import bs4
import pandas as pd
from boto3 import Session
from io_library.read_from_s3 import read_dict_from_json_s3

from io_library.list_objects_s3 import list_objects_s3

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
    html_content = resp['Body'].read()
    soup = bs4.BeautifulSoup(html_content, features="lxml")
    dfs_to_concat = []
    for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
        link_text = x.attrs["href"]
        suffix = link_text.replace("/", "_")
        suffix = suffix.replace("__", "_")
        suffix = suffix.strip("_")
        wine_key = f"{OUTPUT_PREFIX}/{today_date_str}/json/wines/{suffix}.json"
        print(f"wine_key = {wine_key}")
        try:
            data_dict = read_dict_from_json_s3(
                bucket=OUTPUT_BUCKET,
                key=wine_key
            )
            count += 1
        except:
            continue
        row_df = pd.DataFrame(data_dict, index=[count])
        dfs_to_concat.append(row_df)
        print(f"length = {len(dfs_to_concat)}")
    df = pd.concat(dfs_to_concat)
    df.index.name = "index"
    print(f"df.shape = {df.shape}")
    write_csv_to_s3(df, bucket=OUTPUT_BUCKET, key=f"{OUTPUT_PREFIX}/{today_date_str}/csv/pages/{page_tail}.csv")


def main_scrape_wine_parallel_pages_json_to_csv():
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
            executor.submit(main_json_to_csv, obj)


if __name__ == "__main__":
    main_scrape_wine_parallel_pages_json_to_csv()