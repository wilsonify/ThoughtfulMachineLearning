import math
import time
from datetime import datetime

import bs4
import pandas as pd
import requests
from boto3 import Session

from io_library.write_to_s3 import write_csv_to_s3
from s01_wine_scrape import OUTPUT_BUCKET, OUTPUT_PREFIX

result_filename = "wine_spectator.csv"
pd.options.plotting.backend = "plotly"

baseurl = "https://www.wine.com/list/wine/wine-spectator/7155-202"

columns_in_order = [
    "search_url", "name", "wine_url", "productVarietal", "productStock", "productRegion", "productPrice",
    "productOrigin", "productID", "productCompetitiveIntensity", "ProductAvailability", "pProductID", "description",
    "pageName", "shippingRegion", "shipToState", "priceCurrency", "price", "averageRating_bestRating",
    "averageRating_worstRating", "bestRating", "worstRating", "additionalType", "uploadDate", "prodAlcoholVolume_text",
    "prodAlcoholPercent_percent", "JS", "WS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP",
]


def main():
    s3_session = Session()
    s3_client = s3_session.client('s3')
    save_initial({})
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
        soup = bs4.BeautifulSoup(resp.text, features="lxml")
        for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
            data = {}
            count += 1
            wine_url = "https://www.wine.com/product" + x.attrs["href"]
            data[f"w{count:00002d}"] = {}
            data[f"w{count:00002d}"]["search_url"] = url
            data[f"w{count:00002d}"]["wine_url"] = wine_url
            time.sleep(0.1)
            try:
                wine_resp = requests.get(wine_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
                print(f"got wine_url={wine_url}")
            except:
                print(f"failed to get wine_url={wine_url}")
                continue
            wine_soup = bs4.BeautifulSoup(wine_resp.text, features="lxml")
            extract_meta(count, data, wine_soup)
            extract_prodAlcoholVolume(count, data, wine_soup)
            extract_prodAlcoholPercent(count, data, wine_soup)
            extract_ratings(count, data, wine_soup)
            save_progress(data, result_filename)


def save_progress(data, filename):
    result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
    ensure_dtypes(result)
    write_csv_to_s3(
        result[columns_in_order],
        bucket=OUTPUT_BUCKET,
        key=f"{OUTPUT_PREFIX}/{filename}"
    )
    result[columns_in_order].to_csv(filename, mode='a', header=False)


def save_initial(data, filename):
    result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
    ensure_dtypes(result)
    "wines_ship_to_pa.csv"
    write_csv_to_s3(
        result[columns_in_order],
        bucket=OUTPUT_BUCKET,
        key=f"{OUTPUT_PREFIX}/{filename}"
    )


def ensure_dtypes(result):
    result["price"] = pd.to_numeric(result["price"])
    result["WS"] = pd.to_numeric(result["WS"])
    result["JS"] = pd.to_numeric(result["JS"])


def extract_ratings(count, data, wine_soup):
    for n, z in enumerate(wine_soup.find_all(attrs={"class": ["wineRatings_list"]})):
        ratings_list = z.find_all("li", class_="wineRatings_listItem")
        for rating in ratings_list:
            initials = rating.find("span", class_="wineRatings_initials").text
            rating_value = rating.find("span", class_="wineRatings_rating").text
            data[f"w{count:00002d}"][initials] = rating_value


def extract_prodAlcoholPercent(count, data, wine_soup):
    for n, b in enumerate(wine_soup.find_all(attrs={"class": ["prodAlcoholPercent_inner"]})):
        percent_element = b.find("span", class_="prodAlcoholPercent_percent")
        data[f"w{count:00002d}"][
            "prodAlcoholPercent_percent"
        ] = percent_element.text.strip()


def extract_prodAlcoholVolume(count, data, wine_soup):
    for n, b in enumerate(wine_soup.find_all(attrs={"class": ["prodAlcoholVolume"]})):
        for c in b.find_all("span", class_="prodAlcoholVolume_text"):
            data[f"w{count:00002d}"]["prodAlcoholVolume_text"] = c.contents[0]


def extract_meta(count, data, wine_soup):
    for y in wine_soup.find_all(name="meta"):
        if "content" in y.attrs:
            v = y.attrs["content"]
            try:
                k = y.attrs["name"]
            except:
                try:
                    k = y.attrs["itemprop"]
                except:
                    k = y.attrs["class"][0]
            data[f"w{count:00002d}"][k] = v


def parse_page(search_url, wine_url, count, wine_soup):
    data = {}
    data[f"w{count:00002d}"] = {}
    data[f"w{count:00002d}"]["search_url"] = search_url
    data[f"w{count:00002d}"]["wine_url"] = wine_url
    extract_meta(count, data, wine_soup)
    extract_prodAlcoholVolume(count, data, wine_soup)
    extract_prodAlcoholPercent(count, data, wine_soup)
    extract_ratings(count, data, wine_soup)
    return data


def main_scrape_wine_pa():
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
        time.sleep(2)
        url = f"{baseurl}/{i}"
        print(url)
        print(f"{i}/{total_pages} = {i / total_pages:0.2f}")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        soup = bs4.BeautifulSoup(resp.text, features="lxml")
        s3_client.put_object(
            Body=resp.text,
            Bucket=OUTPUT_BUCKET,
            Key=f"{OUTPUT_PREFIX}/{today_date_str}/html/page_{i}.html"
        )
        for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
            count += 1
            time.sleep(1)
            link_text = x.attrs["href"]
            wine_url = f"https://www.wine.com/product{link_text}"
            wine_resp = requests.get(wine_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
            print(f"got wine_url={wine_url}")
            suffix = link_text.replace("/", "_")
            suffix = suffix.replace("__", "_")
            s3_client.put_object(
                Body=wine_resp.text,
                Bucket=OUTPUT_BUCKET,
                Key=f"{OUTPUT_PREFIX}/{today_date_str}/html/{count}_{suffix}.html"
            )


if __name__ == "__main__":
    main_scrape_wine_pa()
