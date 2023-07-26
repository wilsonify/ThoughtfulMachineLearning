import logging

import connexion
import six
from typing import Dict
from typing import Tuple
from typing import Union
from datetime import datetime, timedelta
from openapi_server.models.http_validation_error import HTTPValidationError  # noqa: E501
from openapi_server.models.predict_ws_input import PredictWsInput  # noqa: E501
from openapi_server.models.predict_ws_output import PredictWsOutput  # noqa: E501
from openapi_server.models.token import Token  # noqa: E501
from openapi_server.models.user import User  # noqa: E501
from openapi_server import util




def predict_predict_post(predict_ws_input):  # noqa: E501
    """Predict

     # noqa: E501

    :param predict_ws_input: 
    :type predict_ws_input: dict | bytes

    :rtype: Union[PredictWsOutput, Tuple[PredictWsOutput, int], Tuple[PredictWsOutput, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        predict_ws_input = PredictWsInput.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def read_items_items_get():  # noqa: E501
    """Read Items

     # noqa: E501


    :rtype: Union[object, Tuple[object, int], Tuple[object, int, Dict[str, str]]
    """
    return 'do some magic!'


def read_own_items_users_me_items_get():  # noqa: E501
    """Read Own Items

     # noqa: E501


    :rtype: Union[object, Tuple[object, int], Tuple[object, int, Dict[str, str]]
    """
    return 'do some magic!'


def read_users_me_users_me_get():  # noqa: E501
    """Read Users Me

     # noqa: E501


    :rtype: Union[User, Tuple[User, int], Tuple[User, int, Dict[str, str]]
    """
    return 'do some magic!'
