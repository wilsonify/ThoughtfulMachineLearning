import pandas as pd

from tml_wine.main import app
from tml_wine.models.models import virtual_ws_meta, virtual_ws_other, virtual_ws_ensemble
from tml_wine.predict_ws_api import input_model, output_model
from tml_wine.predict_ws_inference import predictors, other_ratings, model_columns, pdesire, wdesire, composite


@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    df = pd.DataFrame([data.dict()])
    df["ws_pred_meta"] = virtual_ws_meta.predict(df[predictors])
    df["ws_pred_other"] = virtual_ws_other.predict(df[other_ratings])
    df["ws_pred"] = virtual_ws_ensemble.predict(df[model_columns])
    df["price_desire"] = df["price"].apply(pdesire)
    df["ws_pred_desire"] = df["ws_pred"].apply(wdesire)
    df["composite_desire"] = composite(df["price_desire"], df["ws_pred_desire"])
    return {"ws_pred": df["ws_pred"].round(4).iloc[0], "composite_desire": df["composite_desire"].round(4).iloc[0]}
