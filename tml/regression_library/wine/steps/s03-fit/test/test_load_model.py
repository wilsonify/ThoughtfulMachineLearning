import mlflow
from pycaret.regression import load_model

from io_library.skl_to_py import convert_pipeline_to_list_of_tuple, get_skl_to_dict


def test_mlflow_save_model():
    loaded_model = load_model('models/ws_rf')
    mlflow.sklearn.save_model(loaded_model, "mlflow/ws_rf")


def test_get_skl_to_dict():
    loaded_model = load_model('models/ws_rf')
    params_dict = get_skl_to_dict(loaded_model)
    assert params_dict == {}


def test_convert_pipeline_to_list_of_tuple():
    loaded_model = load_model('models/ws_rf')
    lot = convert_pipeline_to_list_of_tuple(loaded_model)
    assert lot == {}
