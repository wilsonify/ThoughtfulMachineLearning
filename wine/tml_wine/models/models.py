import os.path

from pycaret.regression import load_model

path_to_here = os.path.abspath(os.path.dirname(__file__))
path_to_models = f"{path_to_here}"

# Load the Models
virtual_ws_meta = load_model(f"{path_to_models}/predict_ws_from_meta/best-model")
virtual_ws_other = load_model(f"{path_to_models}/predict_ws_from_other/best-model-other")
virtual_ws_ensemble = load_model(f"{path_to_models}/ensemble/best-model-ensemble")
