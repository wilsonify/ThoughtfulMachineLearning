import connexion
import six
from typing import Dict
from typing import Tuple
from typing import Union

from openapi_server.models.http_validation_error import HTTPValidationError  # noqa: E501
from openapi_server.models.predict_ws_input import PredictWsInput  # noqa: E501
from openapi_server.models.predict_ws_output import PredictWsOutput  # noqa: E501
from openapi_server.models.token import Token  # noqa: E501
from openapi_server.models.user import User  # noqa: E501
from openapi_server import util


def login_for_access_token_token_post(username, password, grant_type=None, scope=None, client_id=None, client_secret=None):  # noqa: E501
    """Login For Access Token

     # noqa: E501

    :param username: 
    :type username: dict | bytes
    :param password: 
    :type password: dict | bytes
    :param grant_type: 
    :type grant_type: dict | bytes
    :param scope: 
    :type scope: dict | bytes
    :param client_id: 
    :type client_id: dict | bytes
    :param client_secret: 
    :type client_secret: dict | bytes

    :rtype: Union[Token, Tuple[Token, int], Tuple[Token, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        username = object.from_dict(connexion.request.get_json())  # noqa: E501
    if connexion.request.is_json:
        password = object.from_dict(connexion.request.get_json())  # noqa: E501
    if connexion.request.is_json:
        grant_type = object.from_dict(connexion.request.get_json())  # noqa: E501
    if connexion.request.is_json:
        scope = object.from_dict(connexion.request.get_json())  # noqa: E501
    if connexion.request.is_json:
        client_id = object.from_dict(connexion.request.get_json())  # noqa: E501
    if connexion.request.is_json:
        client_secret = object.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


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
