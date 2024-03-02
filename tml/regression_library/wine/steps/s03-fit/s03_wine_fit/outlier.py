# +
from typing import Union

import numpy as np
import pandas as pd
from pycaret.anomaly import *
from scipy import sparse

from io_library.read_from_s3 import read_csv_from_s3
from s03_wine_fit import INPUT_BUCKET, INPUT_PREFIX

# -

SEQUENCE = (list, tuple, np.ndarray, pd.Series)
SEQUENCE_LIKE = Union[SEQUENCE]
DATAFRAME_LIKE = Union[dict, list, tuple, np.ndarray, sparse.spmatrix, pd.DataFrame]
TARGET_LIKE = Union[int, str, list, tuple, np.ndarray, pd.Series]

input_columns_in_order = [
    "productID",  # unique ID e.g. 1576827
    # "item",  # origal url e.g. https://www.wine.com/product/cafaggio-chianti-classico-riserva-2018/$productID
    # "wine_url",  # s3 prefix e.g. wine/s01-scrape/2024-02-27/html/wines/product_..._$productID.html
    # "pageName",  # Wine:Product Detail:Cafaggio Chianti Classico Riserva 2018
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
    # "description",  # plain text description of winery
    "Connoisseurs'Guide",  # Other Rating
    "Decanter",  # Other Rating
    "JamesSuckling",  # Other Rating
    "JasperMorris",  # Other Rating
    "JebDunnuck",  # Other Rating    
    "RobertParker",  # Other Rating
    "TastingPanel",  # Other Rating
    "Vinous",  # Other Rating    
    "WhiskyAdvocate",  # Other Rating
    "WilfredWong",  # Other Rating
    "WineEnthusiast",  # Other Rating
    "WineSpectator",  # Rating, this is our target variable
]


df = read_csv_from_s3(bucket=INPUT_BUCKET, key=f"{INPUT_PREFIX}/csv/train.csv")
df = df[input_columns_in_order]

df.columns[df.columns.duplicated()]

# for col in df.select_dtypes(np.number):
#     df[col].plot.hist(bins=100,title=col)
#     plt.show()

top_num_feat = [
    "price",
    "JebDunnuck",
    "RobertParker",
    "WilfredWong",
    "Vinous",
    "Decanter",
    "ratingValue"
]

setup(data=df[top_num_feat + ["Varietal", "productOrigin"] + ['WineSpectator']], ignore_features=['productID'],
      imputation_type='simple')

iforest = create_model('iforest')

plot_model(iforest, plot='tsne')
