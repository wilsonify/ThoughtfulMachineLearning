import json
import random
import time

import bs4
import pandas as pd
import requests


def main():
    attrs_of_interest = [
        "list-card-addr",
        "list-card-type",
        "list-card-price",
        "list-card-details",
        "list-card-link",
    ]
    data = {}
    for page in range(1, 21):
        time.sleep(1)
        # url=f"https://www.zillow.com/lewisburg-pa-17837/{page}_p"

        searchQueryState = {
            "pagination": {"currentPage": page},
            "mapBounds": {
                "west": -77.16353057625066,
                "east": -76.40616058113348,
                "south": 40.79172057471558,
                "north": 41.214563845944205,
            },
            "isMapVisible": True,
            "mapZoom": 11,
            "filterState": {"sortSelection": {"value": "globalrelevanceex"}},
            "isListVisible": True,
        }
        url = f"https://www.zillow.com/homes/for_sale/{page}_p/?searchQueryState={json.dumps(searchQueryState)}"

        print(url)
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = bs4.BeautifulSoup(resp.text)

        data_len = len(data)
        for i, x in enumerate(soup.find_all(attrs={"class": ["list-card"]})):
            data[data_len + i] = {}
            for y in x.find_all(attrs={"class": ["list-card-info"]}):
                for z in y.find_all(attrs={"class": attrs_of_interest}):
                    data[data_len + i][x.attrs["class"][0]] = x.text
                    data[data_len + i][y.attrs["class"][0]] = y.text
                    data[data_len + i][z.attrs["class"][0]] = z.text
                    if z.attrs["class"][0] == "list-card-link":
                        homedetails_link = z.attrs["href"]
                        data[data_len + i]["list-card-link"] = homedetails_link
                        time.sleep(random.randint(1, 5))
                        homedetails_resp = requests.get(
                            homedetails_link, headers={"User-Agent": "Mozilla/5.0"}
                        )
                        homedetails_soup = bs4.BeautifulSoup(homedetails_resp.text)
                        for hx in homedetails_soup.find_all(
                            attrs={"class": "ds-home-fact-list"}
                        ):
                            for hy in hx:
                                try:
                                    label, value = hy.text.split(":")
                                    data[data_len + i][label] = value
                                except ValueError:
                                    data[data_len + i]["meta"] = hy.text

    result = pd.DataFrame.from_dict(data, orient="index")

    result.shape

    result.head(100)

    result.to_csv("properties-susquehannavalley2020MAR29.csv")


if __name__ == "__main__":
    main()
