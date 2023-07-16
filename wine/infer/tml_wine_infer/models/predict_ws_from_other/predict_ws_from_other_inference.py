# -*- coding: utf-8 -*-

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
input_model = create_model("predict_ws_from_other_input", **{'JS': 93.0, 'WW': nan, 'D': nan, 'BH': nan, 'W&S': nan, 'WE': nan, 'RP': 92.0, 'JD': nan, 'SJ': nan, 'V': nan, 'CG': nan, 'TP': nan})
output_model = create_model("predict_ws_from_other_output", prediction=90.0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
