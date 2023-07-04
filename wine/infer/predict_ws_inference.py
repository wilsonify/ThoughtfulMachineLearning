import random
from scipy.stats.mstats import gmean

import numpy as np
import pandas as pd
from pycaret.regression import get_config
from pycaret.regression import set_config
from pycaret.regression import (
    setup,
    compare_models,
    create_model,
    tune_model,
    ensemble_model,
    models,
    blend_models,
    stack_models,
    plot_model,
    evaluate_model,
    interpret_model,
    automl,
    predict_model,
    save_model,
    load_model,
    convert_model,
    create_api,
    create_docker
)
from pycaret.utils import version
from yellowbrick.regressor import CooksDistance

id_cols = ['productID',"name","pProductID", "wine_url","pageName","uploadDate"]
target_cols = ['WS']
categorical_cols=[
    "description", "productPrice", "productCompetitiveIntensity", "ProductAvailability", "priceCurrency", 	"additionalType" ,"productOrigin", "productVarietal", "productRegion"
]
extra_cols=["shippingRegion", "shipToState"]
other_ratings = [ 'JS', 'WW', 'D', 'BH', 'W&S', 'WE', 'RP', 'JD', 'SJ', 'V', 'CG', 'TP' ]
predictors = [
    'productStock',
    'price',
    'prodAlcoholPercent_percent'
    #'averageRating_bestRating',
    #'averageRating_worstRating',
    #'bestRating',
    #'worstRating',
    #'prodAlcoholVolume_text',
]

print(f"pycaret version = {version()}")
print("1. Loading Dataset")
df = pd.read_csv("wines_ship_to_pa.csv")
df = df.drop_duplicates('wine_url') 

df['prodAlcoholVolume_text'].unique()
[750., 0., 1500., 375., 187., 1000.,  500., 3000.,700., 6000.]

df = df[df['prodAlcoholVolume_text']==750]

virtual_ws = load_model("best-model")
virtual_ws_other = load_model("predict_ws_from_other/best-model-other")

df["ws_pred_1"] = virtual_ws.predict(df[predictors])
df["ws_pred_2"] = virtual_ws_other.predict(df[other_ratings])
df['ws_pred'] = gmean([df["ws_pred_1"],df["ws_pred_2"]])

from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(df['ws_pred'],df['WS'])
ax.set_xlabel('predicted WS rating')
ax.set_ylabel('actual WS rating')


def desire(h, low, target, high):
    if h <= low:
        return 0.0
    if h >= high:
        return 0.0
    if low < h <= target:
        return -low / (-low + target) + h / (-low + target)
    if target <= h < high:
        return 1.0 + target / (high - target) - h / (high - target)



from functools import partial

pdesire = partial(desire,low=0, target=5, high=100)

wdesire = partial(desire,low=89,target=100,high=105)

x=pd.Series(np.arange(0,110))
y=x.apply(pdesire)
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x,y)
ax.set_xlabel('price')
ax.set_ylabel('desire')

x=pd.Series(np.arange(80,110))
y=x.apply(wdesire)
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x,y)
ax.set_xlabel('WS rating')
ax.set_ylabel('desire')


# +

def composite(pd,wd):
    return gmean([pd,wd],axis=0)
    


# -

df['price_desire'] = df['price'].apply(pdesire)

df['ws_desire'] = df['WS'].apply(wdesire)

df['ws_pred_desire'] = df['ws_pred'].apply(wdesire)

df['composite_desire_pred'] = composite(df['price_desire'],df['ws_pred_desire'])

df['composite_desire'] = composite(df['price_desire'],df['ws_desire'])

list(df.sort_values('composite_desire',ascending=False).head()['wine_url'])

df.sort_values('composite_desire',ascending=False).head()[['name','WS','price']]

list(df.sort_values('composite_desire_pred',ascending=False).head()['wine_url'])

df.sort_values('composite_desire_pred',ascending=False).head()[['name','ws_pred','price']]


