import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pycaret.regression import load_model
from pycaret.utils import version
from skfuzzy import trimf, sigmf, psigmf
from skfuzzy.control import (
    Antecedent,
    Consequent,
    Rule,
    ControlSystem,
    ControlSystemSimulation
)

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


def dataframe_to_dict_of_arrays(df):
    """
    Convert a DataFrame into a dictionary of arrays.
    where keys are column names and values are arrays.
    """
    result_dict = {}
    for column in df.columns:
        result_dict[column] = df[column].values
    return result_dict


def desire(h, low, target, high):
    if h <= low:
        return 0.0
    if h >= high:
        return 0.0
    if low < h <= target:
        return -low / (-low + target) + h / (-low + target)
    if target <= h < high:
        return 1.0 + target / (high - target) - h / (high - target)


def composite(df, visualize=False):
    """
    IF I tell this controller that:
    the price is 20.0
    the rating is 95
    THEN it would recommend yes, make purchase at 0.8 confidence
    """
    # Antecedent objects hold the set of all possible values (universe) and fuzzy membership checks
    # price can be cheap, normal, or expensive
    price = Antecedent(np.linspace(0, 100, 11), 'price')
    price['cheap'] = trimf(price.universe, [0, 0, 10])
    price['normal'] = trimf(price.universe, [0, 20, 50])
    price['expensive'] = trimf(price.universe, [20, 100, 100])

    # priority fuzzily can be top, bottom on a scale of 0 to 10

    # rating can be low, average, high on a scale of 0 to 100
    steep = 1.5
    shift = 95
    rating = Antecedent(np.linspace(0, 100, 100), 'rating')
    rating['low'] = sigmf(rating.universe, shift, -steep)
    rating['average'] = psigmf(rating.universe, 91, steep, 99, -steep)
    rating['high'] = sigmf(rating.universe, shift, steep)

    # purchase can be no, maybe, yes
    purchase = Consequent(np.arange(0, 1.1, 0.1), 'action')
    purchase['no'] = trimf(purchase.universe, [0.0, 0.0, 0.5])
    purchase['maybe'] = trimf(purchase.universe, [0, 0.5, 1.0])
    purchase['yes'] = trimf(purchase.universe, [0.5, 1.0, 1.0])

    # IF cheap and highly rated, THEN take the purchase action
    take_action = ControlSystemSimulation(ControlSystem([
        Rule(price['cheap'] & rating['high'], purchase['yes']),
        Rule(price['expensive'], purchase['no']),
        Rule(rating['low'], purchase['no']),
        Rule(price['normal'] & rating['high'], purchase['maybe']),

    ]))

    take_action.input['price'] = df["price"].values
    take_action.input['rating'] = df["ws_pred"].values
    take_action.compute()
    result = take_action.output['action']

    if visualize:
        price.view()
        plt.show()

        rating.view()
        plt.show()

        purchase.view()
        plt.show()

        purchase.view(sim=take_action)
        plt.show()

    return result


def main():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    print(f"pycaret version = {version()}")
    print("1. Loading Dataset")
    df = pd.read_csv("wines_ship_to_pa_pred_bak.csv", index_col=1)
    unnamed_cols = df.filter(regex="Unnamed").columns
    df = df.drop(unnamed_cols, axis=1)

    for col in predictors:
        df[col] = df[col].astype(float)
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    df = df.drop_duplicates("wine_url")

    vol_unique = df["prodAlcoholVolume_text"].unique()
    print(f"vol_unique = {list(vol_unique)}")

    # [750., 0., 1500., 375., 187., 1000., 500., 3000., 700., 6000.]
    is_750 = df["prodAlcoholVolume_text"] == "750"
    df_subset = df[is_750]

    path_to_here = os.path.abspath(os.path.dirname(__file__))
    path_to_models = f"{path_to_here}/models"

    virtual_ws_meta = load_model(f"{path_to_models}/predict_ws_from_meta/best-model")
    virtual_ws_other = load_model(f"{path_to_models}/predict_ws_from_other/best-model-other")
    virtual_ws_ensemble = load_model(f"{path_to_models}/ensemble/best-model-ensemble")

    x_new = df[predictors]
    df["ws_pred_meta"] = virtual_ws_meta.predict(x_new)
    df["ws_pred_other"] = virtual_ws_other.predict(df[other_ratings])
    df["ws_pred"] = virtual_ws_ensemble.predict(df[model_columns])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(df["ws_pred"], df["WS"])
    ax.set_xlabel("predicted WS rating")
    ax.set_ylabel("actual WS rating")
    # plt.show()

    df["composite_desire"] = composite(df)

    df = df.sort_values("composite_desire", ascending=False)
    print(df.head(6)[["name", "productVarietal", "WS", "ws_pred", "price", "composite_desire"]])
    print(list(df.head(6)["wine_url"]))

    df.to_csv("wines_ship_to_pa_pred.csv")


if __name__ == "__main__":
    main()
