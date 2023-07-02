import math
import time

import bs4
import pandas as pd
import requests

from scrape_wine import (
    extract_meta,
    extract_prodAlcoholPercent,
    extract_ratings,
    extract_prodAlcoholVolume,
    save_initial,
    save_progress
)

result_filename = "wine_spectator_pa.csv"
pd.options.plotting.backend = "plotly"

baseurl = "https://www.wine.com/list/wine/7155"

columns_in_order = [
    "search_url", "name", "wine_url", "productVarietal", "productStock", "productRegion", "productPrice",
    "productOrigin", "productID", "productCompetitiveIntensity", "ProductAvailability", "pProductID", "description",
    "pageName", "shippingRegion", "shipToState", "priceCurrency", "price", "averageRating_bestRating",
    "averageRating_worstRating", "bestRating", "worstRating", "additionalType", "uploadDate", "prodAlcoholVolume_text",
    "prodAlcoholPercent_percent", "JS", "WS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP",
]


def main():
    save_initial({})
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
        for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
            data = {}
            count += 1
            wine_url = "https://www.wine.com/product" + x.attrs["href"]
            data[f"w{count:00002d}"] = {}
            data[f"w{count:00002d}"]["search_url"] = url
            data[f"w{count:00002d}"]["wine_url"] = wine_url
            time.sleep(1)
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
            save_progress(data)


if __name__ == "__main__":
    main()
