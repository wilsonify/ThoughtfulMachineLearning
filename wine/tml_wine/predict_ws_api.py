# -*- coding: utf-8 -*-

from pycaret.regression import load_model
from pydantic import create_model


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
