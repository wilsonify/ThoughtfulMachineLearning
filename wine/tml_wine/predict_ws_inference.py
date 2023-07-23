import os
from functools import partial

import pandas as pd
from matplotlib import pyplot as plt
from pycaret.regression import load_model
from pycaret.utils import version
from scipy.stats.mstats import gmean

id_cols = ["productID", "name", "pProductID", "wine_url", "pageName", "uploadDate"]
target_cols = ["WS"]
categorical_cols = [
    "description",
    "productPrice",
    "productCompetitiveIntensity",
    "ProductAvailability",
    "priceCurrency",
    "additionalType",
    "productOrigin",
    "productVarietal",
    "productRegion",
]
extra_cols = ["shippingRegion", "shipToState"]
other_ratings = ["JS", "WW", "D", "BH", "W&S", "WE", "RP", "JD", "SJ", "V", "CG", "TP"]
predictors = [
    "productStock",
    "price",
    "prodAlcoholPercent_percent"
    # 'averageRating_bestRating',
    # 'averageRating_worstRating',
    # 'bestRating',
    # 'worstRating',
    # 'prodAlcoholVolume_text',
]
model_columns = ["ws_pred_meta", "ws_pred_other"]


def desire(h, low, target, high):
    if h <= low:
        return 0.0
    if h >= high:
        return 0.0
    if low < h <= target:
        return -low / (-low + target) + h / (-low + target)
    if target <= h < high:
        return 1.0 + target / (high - target) - h / (high - target)


def composite(pdes, wd):
    return gmean([pdes, wd], axis=0)


pdesire = partial(desire, low=0, target=10, high=110)

wdesire = partial(desire, low=85, target=99, high=100)


def main():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    print(f"pycaret version = {version()}")
    print("1. Loading Dataset")
    df = pd.read_csv("../wines_ship_to_pa.csv")
    df = df.drop_duplicates("wine_url")

    df["prodAlcoholVolume_text"].unique()
    # [750., 0., 1500., 375., 187., 1000., 500., 3000., 700., 6000.]

    df = df[df["prodAlcoholVolume_text"] == 750]

    path_to_here = os.path.abspath(os.path.dirname(__file__))
    path_to_models = f"{path_to_here}/models"

    virtual_ws_meta = load_model(f"{path_to_models}/predict_ws_from_meta/best-model")
    virtual_ws_other = load_model(f"{path_to_models}/predict_ws_from_other/best-model-other")
    virtual_ws_ensemble = load_model(f"{path_to_models}/ensemble/best-model-ensemble")

    df["ws_pred_meta"] = virtual_ws_meta.predict(df[predictors])
    df["ws_pred_other"] = virtual_ws_other.predict(df[other_ratings])
    df["ws_pred"] = virtual_ws_ensemble.predict(df[model_columns])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(df["ws_pred"], df["WS"])
    ax.set_xlabel("predicted WS rating")
    ax.set_ylabel("actual WS rating")

    df["price_desire"] = df["price"].apply(pdesire)
    df["ws_desire"] = df["WS"].apply(wdesire)
    df["ws_pred_desire"] = df["ws_pred"].apply(wdesire)
    df["composite_desire_pred"] = composite(df["price_desire"], df["ws_pred_desire"])
    df["composite_desire"] = composite(df["price_desire"], df["ws_desire"])

    df = df.sort_values("composite_desire", ascending=False)
    print(df.head(6)[["name", "productVarietal", "WS", "price"]])
    print(list(df.head(6)["wine_url"]))

    df = df.sort_values("composite_desire_pred", ascending=False)
    print(df.head(6)[["name", "productVarietal", "ws_pred", "price"]])
    print(list(df.head(6)["wine_url"]))

    df.to_csv("wines_ship_to_pa_pred.csv")


if __name__ == "__main__":
    main()
