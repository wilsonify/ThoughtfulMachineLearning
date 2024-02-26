import pandas as pd

OUTPUT_BUCKET = "064592191516-kaggle"
OUTPUT_PREFIX = "wine/s01-scrape"

baseurl = "https://www.wine.com/list/wine/7155"
# baseurl = "https://www.wine.com/list/wine/wine-spectator/7155-202"

columns_in_order = [
    "search_url", "name", "wine_url", "productVarietal", "productStock", "productRegion", "productPrice",
    "productOrigin", "productID", "productCompetitiveIntensity", "ProductAvailability", "pProductID", "description",
    "pageName", "shippingRegion", "shipToState", "priceCurrency", "price", "averageRating_bestRating",
    "averageRating_worstRating", "bestRating", "worstRating", "additionalType", "uploadDate", "prodAlcoholVolume_text",
    "prodAlcoholPercent_percent", "JS", "WS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP",
]


