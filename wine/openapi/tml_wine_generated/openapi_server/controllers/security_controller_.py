from jose import jwt
from typing import List

# Your secret key used to sign the JWT tokens (should match the one in your app)
SECRET_KEY = "your_secret_key"

# Algorithm used to sign the JWT tokens (should match the one in your app)
ALGORITHM = "HS256"

def info_from_OAuth2PasswordBearer(token):
    """
    Validate and decode token.
    Returned value will be passed in 'token_info' parameter of your operation function, if there is one.
    'sub' or 'uid' will be set in 'user' parameter of your operation function, if there is one.
    'scope' or 'scopes' will be passed to scope validation function.

    :param token Token provided by Authorization header
    :type token: str
    :return: Decoded token information or None if token is invalid
    :rtype: dict | None
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"scopes": payload.get("scopes", []), "uid": payload.get("sub")}
    except jwt.JWTError:
        return None

def validate_scope_OAuth2PasswordBearer(required_scopes, token_scopes):
    """
    Validate required scopes are included in token scope

    :param required_scopes Required scope to access called API
    :type required_scopes: List[str]
    :param token_scopes Scope present in token
    :type token_scopes: List[str]
    :return: True if access to called API is allowed
    :rtype: bool
    """
    return set(required_scopes).issubset(set(token_scopes))

