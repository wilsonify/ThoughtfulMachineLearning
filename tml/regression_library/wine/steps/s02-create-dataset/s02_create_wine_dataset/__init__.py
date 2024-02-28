INPUT_BUCKET = "064592191516-kaggle"
OUTPUT_BUCKET = "064592191516-kaggle"
INPUT_PREFIX = "wine/s01-scrape"
OUTPUT_PREFIX = "wine/s02-create-dataset"

columns_in_order = [
    "productID",  # unique ID e.g. 1576827
    "item",  # origal url e.g. https://www.wine.com/product/cafaggio-chianti-classico-riserva-2018/$productID
    "wine_url",  # s3 prefix e.g. wine/s01-scrape/2024-02-27/html/wines/product_..._$productID.html
    "pageName"  # Wine:Product Detail:Cafaggio Chianti Classico Riserva 2018
    "productVarietal",  # type of grape e.g. Sangiovese
    "productRegion",  # general location e.g. Chianti Classico
    "productOrigin",  # detailed location e.g. Chianti Classico, Chianti, Tuscany, Italy
    "Varietal",  # Sangiovese
    "Region",  # Chianti Classico, Chianti, Tuscany Italy
    "Producer",  # winery e.g. Cafaggio
    "Vintage",  # year bottled e.g. 2018
    "Size",  # Size of bottle e.g. 750ML
    "ABV",  # Alchohol by Volume e.g. 14.5%
    "productStock",  # number of bottles in stock e.g. 1186
    "ratingValue",  # Average Rating e.g. 92
    "reviewCount",  # number of reviews e.g. 5
    "priceCurrency",  # currency of price e.g. USD
    "price",  # price of wine e.g. 28.99
    "description",  # plain text description of winery
    "WineSpectator",  # Rating, this is our target variable
    "JamesSuckling",  # Other Rating
    "RobertParker",  # Other Rating
    "WineEnthusiast",  # Other Rating
    "WilfredWong",  # Other Rating
    "Decanter",  # Other Rating
    "JebDunnuck"  # Other Rating

]
