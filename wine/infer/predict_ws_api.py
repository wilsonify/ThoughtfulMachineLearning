# -*- coding: utf-8 -*-

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pycaret.regression import load_model
from pydantic import create_model

from wine.infer.predict_ws_inference import predictors, other_ratings, model_columns, pdesire, wdesire, composite

# Create the app
app = FastAPI()

# Load trained Pipeline
virtual_ws_meta = load_model("../predict_ws_from_meta/best-model")
virtual_ws_other = load_model("../predict_ws_from_other/best-model-other")
virtual_ws_ensemble = load_model("../ensemble/best-model-ensemble")

# Create input/output pydantic models
input_model = create_model("predict_ws_input", **{
    'productStock': 22.0,
    'price': 79.98999786376953,
    'prodAlcoholPercent_percent': 14.5,
    'JS': 93.0, 'WW': 93.0, 'D': 93.0, 'BH': 93.0,
    'W&S': 93.0, 'WE': 93.0, 'RP': 92.0, 'JD': 93.0, 'SJ': 93.0,
    'V': 93.0, 'CG': 93.0, 'TP': 93.0
})
output_model = create_model("predict_ws_output", **{
    "ws_pred": 90.0,
    "composite_desire": 100.0
})


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    df = pd.DataFrame([data.dict()])
    df["ws_pred_meta"] = virtual_ws_meta.predict(df[predictors])
    df["ws_pred_other"] = virtual_ws_other.predict(df[other_ratings])
    df['ws_pred'] = virtual_ws_ensemble.predict(df[model_columns])
    df['price_desire'] = df['price'].apply(pdesire)
    df['ws_pred_desire'] = df['ws_pred'].apply(wdesire)
    df['composite_desire'] = composite(df['price_desire'], df['ws_pred_desire'])
    return {
        "ws_pred": df["ws_pred"].round(4).iloc[0],
        "composite_desire": df["composite_desire"].round(4).iloc[0]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
