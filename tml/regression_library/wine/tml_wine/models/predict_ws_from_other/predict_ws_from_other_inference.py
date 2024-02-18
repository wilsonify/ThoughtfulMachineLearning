# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("predict_ws_from_other")

# Create input/output pydantic models
input_model = create_model("predict_ws_from_other_input", **{'JS': 93.0, 'WW': np.nan, 'D': np.nan, 'BH': np.nan, 'W&S': np.nan, 'WE': np.nan, 'RP': 92.0, 'JD': np.nan, 'SJ': np.nan, 'V': np.nan, 'CG': np.nan, 'TP': np.nan})
output_model = create_model("predict_ws_from_other_output", prediction=90.0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
