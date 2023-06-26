import time

import bs4
import pandas as pd
import requests

pd.options.plotting.backend = "plotly"

baseurl = "https://www.wine.com/list/wine/wine-spectator/7155-202?sortBy=priceLowToHigh"

columns_in_order = [
    "name", "wine_url", "productVarietal", "productStock", "productRegion", "productPrice", "productOrigin",
    "productID", "productCompetitiveIntensity", "ProductAvailability", "pProductID", "description", "pageName",
    "shippingRegion", "shipToState", "priceCurrency", "price", "averageRating_bestRating", "averageRating_worstRating",
    "bestRating", "worstRating", "additionalType", "uploadDate", "prodAlcoholVolume_text", "prodAlcoholPercent_percent",
    "JS", "WS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP",
]


def main():
    data = {}
    result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
    count = 0
    for i in range(1, 1000):
        time.sleep(5)
        url = f"{baseurl}/{i}"
        print(url)
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
        soup = bs4.BeautifulSoup(resp.text, features="lxml")
        for j, x in enumerate(soup.find_all(attrs={"class": ["listGridItemName"]})):
            time.sleep(0.2)
            result = pd.DataFrame.from_dict(data, orient="index", columns=columns_in_order)
            ensure_dtypes(result)
            result[columns_in_order].to_csv("wine_spectator.csv")
            count += 1
            wine_url = "https://www.wine.com/product" + x.attrs["href"]
            data[f"w{count:0d}"] = {}
            data[f"w{count:0d}"]["search_url"] = url
            data[f"w{count:0d}"]["wine_url"] = wine_url
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

    # import plotly.graph_objects as go
    # fig = go.Figure(data=[go.Scatter(x=result["price"], y=result["WS"], mode="markers", hovertext=result["name"])])
    # fig.show()


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
            data[f"w{count:0d}"][initials] = rating_value


def extract_prodAlcoholPercent(count, data, wine_soup):
    for n, b in enumerate(wine_soup.find_all(attrs={"class": ["prodAlcoholPercent_inner"]})):
        percent_element = b.find("span", class_="prodAlcoholPercent_percent")
        data[f"w{count:0d}"][
            "prodAlcoholPercent_percent"
        ] = percent_element.text.strip()


def extract_prodAlcoholVolume(count, data, wine_soup):
    for n, b in enumerate(wine_soup.find_all(attrs={"class": ["prodAlcoholVolume"]})):
        for c in b.find_all("span", class_="prodAlcoholVolume_text"):
            data[f"w{count:0d}"]["prodAlcoholVolume_text"] = c.contents[0]


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
            data[f"w{count:0d}"][k] = v


if __name__ == "__main__":
    main()
