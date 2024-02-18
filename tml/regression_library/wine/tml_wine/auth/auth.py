from datetime import datetime, timedelta
from typing import Annotated, Union

from fastapi import Depends
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# OAuth2PasswordBearer token scheme for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
    }
}


class Token(BaseModel):
    """Model representing the access token."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Model representing the data inside the JWT token."""

    username: Union[str, None] = None


class User(BaseModel):
    """Model representing the user data."""

    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    """Model representing the user data stored in the database."""

    hashed_password: str


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
    if not verify_password(password, user.hashed_password):
        return False
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


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Get the current authenticated user from the access token.

    Parameters:
        token (str): The access token provided in the request header.

    Raises:
        HTTPException: If the token is invalid or expired.

    Returns:
        User: The current authenticated user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
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
        return UserInDB(**user_dict)
