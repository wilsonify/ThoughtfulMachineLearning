def extract_ratings(data, wine_soup):
    for n, z in enumerate(wine_soup.find_all(attrs={"class": ["wineRatings_list"]})):
        ratings_list = z.find_all("li", class_="wineRatings_listItem")
        for rating in ratings_list:
            initials = rating.find("span", class_="wineRatings_initials").text
            rating_value = rating.find("span", class_="wineRatings_rating").text
            data[initials] = rating_value


def extract_prodAlcoholPercent(data, wine_soup):
    for n, b in enumerate(wine_soup.find_all(attrs={"class": ["prodAlcoholPercent_inner"]})):
        percent_element = b.find("span", class_="prodAlcoholPercent_percent")
        data["prodAlcoholPercent_percent"] = percent_element.text.strip()


def extract_prodAlcoholVolume(data, wine_soup):
    for n, b in enumerate(wine_soup.find_all(attrs={"class": ["prodAlcoholVolume"]})):
        for c in b.find_all("span", class_="prodAlcoholVolume_text"):
            data["prodAlcoholVolume_text"] = c.contents[0]


def extract_meta(data, wine_soup):
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
            data[k] = v
