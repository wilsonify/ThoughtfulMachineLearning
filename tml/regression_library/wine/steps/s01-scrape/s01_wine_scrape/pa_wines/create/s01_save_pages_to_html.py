import json
import math
import time
from datetime import datetime

import boto3
import bs4
import requests
from boto3 import Session

from s01_wine_scrape import (
    OUTPUT_BUCKET,
    OUTPUT_PREFIX,
    baseurl
)


def scrape_one_page_pa(i):
    """scrape one html page"""
    url = f"{baseurl}/{i}"
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    s3_session = Session()
    s3_client = s3_session.client('s3')
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
    s3_client.put_object(
        Body=resp.text,
        Bucket=OUTPUT_BUCKET,
        Key=f"{OUTPUT_PREFIX}/{today_date_str}/html/pages/page_{i:02d}.html"
    )


def main_scrape_wine_pa():
    """scrape all html pages in series"""
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    s3_session = Session()
    s3_client = s3_session.client('s3')
    resp = requests.get(baseurl, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
    soup = bs4.BeautifulSoup(resp.text, features="lxml")
    total_items = soup.find("span", class_="countItems").text.strip(" Items").replace(",", "")
    total_items = int(total_items)
    total_pages = math.floor(total_items / 25.0)
    count = 0
    for i in range(1, total_pages + 1):
        time.sleep(0.5)
        url = f"{baseurl}/{i}"
        print(url)
        print(f"{i}/{total_pages} = {i / total_pages:0.2f}")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        s3_client.put_object(
            Body=resp.text,
            Bucket=OUTPUT_BUCKET,
            Key=f"{OUTPUT_PREFIX}/{today_date_str}/html/pages/page_{i:02d}.html"
        )


def enqueue_scrape_one_page_pa():
    sqs = boto3.client('sqs')
    queue_url = 'wine-sqs-try'
    resp = requests.get(baseurl, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
    soup = bs4.BeautifulSoup(resp.text, features="lxml")
    total_items = soup.find("span", class_="countItems").text.strip(" Items").replace(",", "")
    total_items = int(total_items)
    total_pages = math.floor(total_items / 25.0)
    for i in range(1, total_pages + 1):
        print(f"{i}/{total_pages} = {i / total_pages:0.2f}")
        message_dict = {
            "strategy": "scrape_one_page_pa",
            "i": i
        }
        message_str = json.dumps(message_dict)
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=message_str
        )


if __name__ == "__main__":
    main_scrape_wine_pa()
