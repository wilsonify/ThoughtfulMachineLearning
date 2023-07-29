import connexion
import pandas as pd

from openapi_server.models.predict_ws_input import PredictWsInput  # noqa: E501
from openapi_server.models.predict_ws_output import PredictWsOutput  # noqa: E501
from tml_wine.models.models import virtual_ws_meta, virtual_ws_other, virtual_ws_ensemble
from tml_wine.predict_ws_inference import predictors, other_ratings, model_columns, pdesire, wdesire, composite


def predict_predict_post(body):  # noqa: E501
    """
    Predict
    """
    print(f"body={body}")
    request = connexion.request.get_json()
    PredictWsInput.from_dict(request)  # data validation
    df = pd.DataFrame([request])
    df["ws_pred_meta"] = virtual_ws_meta.predict(df[predictors])
    df["ws_pred_other"] = virtual_ws_other.predict(df[other_ratings])
    df["ws_pred"] = virtual_ws_ensemble.predict(df[model_columns])
    df["price_desire"] = df["price"].apply(pdesire)
    df["ws_pred_desire"] = df["ws_pred"].apply(wdesire)
    df["composite_desire"] = composite(df["price_desire"], df["ws_pred_desire"])
    predict_ws_output = PredictWsOutput.from_dict({
        "ws_pred": df["ws_pred"].round(4).iloc[0],
        "composite_desire": df["composite_desire"].round(4).iloc[0]
    })
    return predict_ws_output
