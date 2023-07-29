import os
from datetime import datetime, timedelta
from typing import Union

from jose import JWTError
from jose import jwt
from passlib.context import CryptContext
from werkzeug.exceptions import Unauthorized

# Secret Key and Algorithm for JWT
SECRET_KEY = os.getenv("TML_JWT_SECRET_KEY", "d5aac7c87d5fff9e1a7d116a5aaeca20f5a1b5b82fc144ac69a08dafee270633")
ALGORITHM = "HS256"

# Access token expiry in minutes
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
ACCESS_TOKEN_EXPIRE_MINUTES = MINUTES_PER_HOUR * HOURS_PER_DAY

# Fake user database for demonstration purposes
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$SoA2zz/odSHuWJabAQm1Wunp4griYQXiTIqJ99JAgbj/0uUMhhngm",
        "disabled": False,
        "scopes": ["read", "write"]
    }
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def info_from_oauth2_password_bearer(token: str):
    """
    Validate and decode token.
    Returned value will be passed in 'token_info' parameter of your operation function, if there is one.
    'sub' or 'uid' will be set in 'user' parameter of your operation function, if there is one.
    'scope' or 'scopes' will be passed to scope validation function.

    :param token JSON Web Token provided by Authorization header
    :type token: str

    :return: Decoded token information or None if token is invalid
    :rtype: dict
    """

    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def validate_scope_oauth2_password_bearer(required_scopes, allowed_scopes):
    """ validate scope meets required scopes """
    return set(required_scopes).issubset(set(allowed_scopes))


def authenticate_user(username: str, password: str):
    """
    Authenticate a user with their username and password.

    Parameters:
        username (str): The username of the user to authenticate.
        password (str): The plain password of the user to authenticate.

    Returns:
        UserInDB: The authenticated user object if successful, False otherwise.
    """
    try:
        user = fake_users_db[username]
    except KeyError:
        raise Unauthorized

    if not user:
        raise Unauthorized

    verified_password = pwd_context.verify(
        secret=password,
        hash=user["hashed_password"]
    )

    if not verified_password:
        raise Unauthorized

    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    """
    Create an access token with the given data.

    Parameters:
        data (dict): The data to include in the token.
        expires_delta (timedelta, optional): The expiration time of the token. Defaults to None.

    Returns:
        str: The generated access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token):
    """
    Get the current authenticated user from the access token.

    Parameters:
        token (str): The access token provided in the request header.

    Raises:
        HTTPException: If the token is invalid or expired.

    Returns:
        User: The current authenticated user.
    """

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise Unauthorized

    username: str = payload.get("sub")

    try:
        user = fake_users_db[username]
    except KeyError:
        raise Unauthorized

    return user


def login_for_access_token_token_post(body):  # noqa: E501
    """
    Login For Access Token
    """
    print("login_for_access_token_token_post")
    grant_type = next(iter(body["grant_type"]))
    client_id = body.get("client_id", "")
    client_secret = body.get("client_secret", "")
    username = next(iter(body["username"]))
    password = next(iter(body["password"]))
    user_dict = authenticate_user(username, password)
    username_from_db = user_dict["username"]
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": username_from_db}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}
