import logging
from datetime import datetime, timedelta
from typing import Union
import connexion
from fastapi import HTTPException, status
from jose import JWTError
from jose import jwt
from passlib.context import CryptContext

# CryptContext for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret Key and Algorithm for JWT
SECRET_KEY = "d5aac7c87d5fff9e1a7d116a5aaeca20f5a1b5b82fc144ac69a08dafee270633"
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


def info_from_OAuth2PasswordBearer(token):
    """
    Validate and decode token.
    Returned value will be passed in 'token_info' parameter of your operation function, if there is one.
    'sub' or 'uid' will be set in 'user' parameter of your operation function, if there is one.
    'scope' or 'scopes' will be passed to scope validation function.

    :param token JSON Web Token provided by Authorization header
    :type token: str
    :return: Decoded token information or None if token is invalid
    :rtype: dict | None
    """

    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def validate_scope_OAuth2PasswordBearer(required_scopes, allowed_scopes):
    return set(required_scopes).issubset(set(allowed_scopes))


def authenticate_user(fake_db, username: str, password: str):
    """
    Authenticate a user with their username and password.

    Parameters:
        fake_db (dict): The fake user database.
        username (str): The username of the user to authenticate.
        password (str): The plain password of the user to authenticate.

    Returns:
        UserInDB: The authenticated user object if successful, False otherwise.
    """
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


def verify_password(plain_password, hashed_password):
    """
    Verify a plain password against a hashed password.

    Parameters:
        plain_password (str): The plain password to verify.
        hashed_password (str): The hashed password to compare with.

    Returns:
        bool: True if the passwords match, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """
    Hash a password.

    Parameters:
        password (str): The plain password to hash.

    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)


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


async def get_current_user(token):
    """
    Get the current authenticated user from the access token.

    Parameters:
        token (str): The access token provided in the request header.

    Raises:
        HTTPException: If the token is invalid or expired.

    Returns:
        User: The current authenticated user.
    """
    credentials_exception = dict(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user):
    """
    Get the current active user.

    Parameters:
        current_user (User): The current authenticated user.

    Raises:
        HTTPException: If the user is inactive.

    Returns:
        User: The current active user.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def fake_hash_password(password: str):
    """
    Generate a fake hashed password for demonstration purposes.

    Parameters:
        password (str): The plain password.

    Returns:
        str: The fake hashed password.
    """
    return "fakehashed" + password


def fake_decode_token(token):
    """
    Decode a fake token for demonstration purposes.

    Parameters:
        token: The token to decode.

    Returns:
        UserInDB: The user object decoded from the token.
    """
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db, token)
    return user


def get_user(database, username: str):
    """
    Get a user from the fake user database.

    Parameters:
        database (dict): The fake user database.
        username (str): The username of the user to retrieve.

    Returns:
        UserInDB: The user object if found, None otherwise.
    """
    if username in database:
        user_dict = database[username]
        return user_dict

def login_for_access_token_token_post():  # noqa: E501
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
    print("login_for_access_token_token_post")
    body = connexion.request.get_data()
    logging.warning(f"body = {body}")
    username = body["username"]
    password = body["password"]
    grant_type = body.get("grant_type","")
    scope = body.get("scope","")
    client_id = body.get("client_id","")
    client_secret = body.get("client_secret","")
    user = authenticate_user(fake_users_db, username, password)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}
