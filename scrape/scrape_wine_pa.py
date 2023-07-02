import time

import bs4
import pandas as pd
import requests

from scrape_wine import (
    extract_meta,
    extract_prodAlcoholPercent,
    extract_ratings,
    ensure_dtypes,
    extract_prodAlcoholVolume
)

pd.options.plotting.backend = "plotly"

baseurl = "https://www.wine.com/list/wine/7155"

columns_in_order = [
    "name", "wine_url", "productVarietal", "productStock", "productRegion", "productPrice", "productOrigin",
    "productID", "productCompetitiveIntensity", "ProductAvailability", "pProductID", "description", "pageName",
    "shippingRegion", "shipToState", "priceCurrency", "price", "averageRating_bestRating", "averageRating_worstRating",
    "bestRating", "worstRating", "additionalType", "uploadDate", "prodAlcoholVolume_text", "prodAlcoholPercent_percent",
    "JS", "WS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP",
]


def save_progress(data):
    result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
    ensure_dtypes(result)
    result[columns_in_order].to_csv("wine_spectator_pa.csv")


def main():
    data = {}
    count = 0
    for i in range(1, 86):
        time.sleep(1)
        url = f"{baseurl}/{i}"
        print(url)
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        soup = bs4.BeautifulSoup(resp.text, features="lxml")
        for x in soup.find_all(attrs={"class": ["listGridItemName"]}):
            count += 1
            wine_url = "https://www.wine.com/product" + x.attrs["href"]
            data[f"w{count:00002d}"] = {}
            data[f"w{count:00002d}"]["search_url"] = url
            data[f"w{count:00002d}"]["wine_url"] = wine_url
            time.sleep(0.2)
            try:
                wine_resp = requests.get(wine_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
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
