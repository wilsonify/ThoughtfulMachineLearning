from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuthError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from starlette import status

from tml_wine.auth.auth import Token
from tml_wine.main import app

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Your Google OpenID Connect configuration
GOOGLE_CLIENT_ID = "your_google_client_id"
GOOGLE_CLIENT_SECRET = "your_google_client_secret"
GOOGLE_REDIRECT_URI = "your_redirect_uri"
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# A list of scopes that your application requests from the user
SCOPES = ["openid", "email", "profile"]


@app.post("/token", response_model=Token)
async def login_for_access_token_google(token: str = Depends(oauth2_scheme)):
    # Step 1: Create an OAuth2Client instance with the Google OpenID configuration
    oauth_client = AsyncOAuth2Client(client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET)

    try:
        # Step 2: Fetch and verify the access token from Google
        token_info = await oauth_client.parse_request_body_response(token)

        # Step 3: Fetch user info from Google using the access token
        userinfo_url = GOOGLE_DISCOVERY_URL.replace("/.well-known/openid-configuration", "/userinfo")
        userinfo = await oauth_client.get(userinfo_url, token=token)

        # Step 4: Verify the ID token to get the user's email and other details
        id_token = token_info.get("id_token")
        if id_token:
            claims = jwt.decode(id_token, GOOGLE_CLIENT_SECRET, algorithms=["HS256"])
            email = claims.get("email")
            # You can retrieve other user details from 'claims' based on your requirements
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid ID token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Step 5: Handle the authenticated user
        # Here, you may want to validate the email against your user database
        # and retrieve the corresponding user object
        # For demonstration, we will just return the email as the username
        return {"access_token": token, "token_type": "bearer", "username": email}

    except OAuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
