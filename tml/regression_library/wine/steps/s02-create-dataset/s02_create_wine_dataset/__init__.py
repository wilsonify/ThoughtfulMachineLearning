import pandas as pd

INPUT_BUCKET = "064592191516-kaggle"
OUTPUT_BUCKET = "064592191516-kaggle"
INPUT_PREFIX = "wine/s01-scrape/2024-02-25"
OUTPUT_PREFIX = "wine/s02-create-dataset"

columns_in_order = [
    "search_url", "name", "wine_url", "productVarietal", "productStock", "productRegion", "productPrice",
    "productOrigin", "productID", "productCompetitiveIntensity", "ProductAvailability", "pProductID", "description",
    "pageName", "shippingRegion", "shipToState", "priceCurrency", "price", "averageRating_bestRating",
    "averageRating_worstRating", "bestRating", "worstRating", "additionalType", "uploadDate", "prodAlcoholVolume_text",
    "prodAlcoholPercent_percent", "JS", "WS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP",
]

pd.options.plotting.backend = "plotly"
