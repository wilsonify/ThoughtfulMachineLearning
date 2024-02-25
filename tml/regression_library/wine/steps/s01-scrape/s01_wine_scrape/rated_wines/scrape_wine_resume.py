import math
import time
from math import floor

import bs4
import pandas as pd
import requests

from s01_wine_scrape.extract import extract_ratings, extract_prodAlcoholPercent, extract_prodAlcoholVolume, extract_meta
from s01_wine_scrape.save_progress import save_progress

pd.options.plotting.backend = "plotly"

baseurl = "https://www.wine.com/list/wine/wine-spectator/7155-202"


def restore_progress():
    df = pd.read_csv("wine_spectator.csv", index_col=0)
    data = df.to_dict(orient="index")
    wine_count = df.shape[0]
    page_count = floor(wine_count / 25.0)
    return page_count, wine_count, data


def main_scrape_wine_resume():
    i_0, count, data = restore_progress()
    resp = requests.get(baseurl, headers={"User-Agent": "Mozilla/5.0"}, timeout=100)
    soup = bs4.BeautifulSoup(resp.text, features="lxml")
    total_items = soup.find("span", class_="countItems").text.strip(" Items").replace(",", "")
    total_items = int(total_items)
    total_pages = math.floor(total_items / 25.0)
    for i in range(i_0, total_pages):
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
            if count % 1000 == 0:
                save_progress(data)
    save_progress(data)
    # import plotly.graph_objects as go
    # fig = go.Figure(data=[go.Scatter(x=result["price"], y=result["WS"], mode="markers", hovertext=result["name"])])
    # fig.show()


if __name__ == "__main__":
    main_scrape_wine_resume()
