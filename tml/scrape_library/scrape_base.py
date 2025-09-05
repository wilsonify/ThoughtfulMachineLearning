# scrape_base.py
import logging

import pandas as pd
import random
import requests
import time
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(url, sleep_range=(0.5, 2.0), retries=3, **kwargs):
    """Fetch a URL with retries and polite sleep."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=kwargs.get("timeout", 30))
            resp.raise_for_status()
            if sleep_range:
                time.sleep(random.uniform(*sleep_range))
            return BeautifulSoup(resp.text, features="lxml")
        except Exception as e:
            logging.exception("Error fetching %s: %s", url, e)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def paginate(baseurl, total_items, per_page=25):
    """Yield paginated URLs."""
    total_pages = (total_items + per_page - 1) // per_page
    for i in range(1, total_pages + 1):
        yield f"{baseurl}/{i}", i, total_pages


def init_csv(filename, columns):
    pd.DataFrame(columns=columns).to_csv(filename, index=False)


def append_csv(filename, data, columns):
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(columns=columns)
    df.to_csv(filename, mode="a", header=False, index=False)


def restore_csv(filename):
    return pd.read_csv(filename)
