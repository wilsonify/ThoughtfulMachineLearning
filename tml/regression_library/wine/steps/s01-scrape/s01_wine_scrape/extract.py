import logging


def extract_ratings(data, wine_soup):
    for n, z in enumerate(wine_soup.find_all(attrs={"class": ["pipProdWineRatings"]})):
        ratings_list = z.find_all("li", class_="pipProdWineRatings_item")
        for rating in ratings_list:
            rating_str = rating.text.strip()
            rating_str = rating_str.replace("Wine &Spirits", "Wine&Spirits")
            rating_str = rating_str.replace("The SommJournal", "TheSommJournal")
            logging.debug(f"rating_str = {rating_str}")
            rating_value, rating_name = rating_str.split()
            data[rating_name] = rating_value


def extract_details(data, wine_soup):
    """
        'Varietal': 'Other Red Blends',
        'Region': 'Spain',
        'Producer': 'Abadia Retuerta',
        'Vintage': '2018',
        'Size': '750ML',
        'ABV': '14.5%'
    """

    titles = wine_soup.find_all('div', class_='pipProdDetails_title')
    names = wine_soup.find_all('div', class_='pipProdDetails_name')
    for title, name in zip(titles, names):
        data[title.text.strip()] = name.text.strip()


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
