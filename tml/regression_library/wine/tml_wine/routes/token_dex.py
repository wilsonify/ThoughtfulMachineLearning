from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuthError
from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from starlette import status

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dex OIDC configuration

DEX_CLIENT_ID = "example-app"
DEX_CLIENT_SECRET = "ZXhhbXBsZS1hcHAtc2VjcmV0"
DEX_ISSUER_URL = "http://localhost:5556/dex"
DEX_REDIRECT_URI = "http://localhost:5555/callback"
DEX_DISCOVERY_URL = f"{DEX_ISSUER_URL}/.well-known/openid-configuration"

# Scopes requested from Dex
SCOPES = ["openid", "profile", "email"]

async def login_for_access_token(token: str = Depends(oauth2_scheme)):
    # Step 1: Create an OAuth2Client instance with the Dex OIDC configuration
    oauth_client = AsyncOAuth2Client(client_id=DEX_CLIENT_ID, client_secret=DEX_CLIENT_SECRET)

    try:
        # Step 2: Fetch and verify the access token from Dex
        token_info = await oauth_client.parse_request_body_response(token)

        # Step 3: Fetch user info from Dex using the access token
        userinfo_url = DEX_DISCOVERY_URL.replace("/.well-known/openid-configuration", "/userinfo")
        userinfo = await oauth_client.get(userinfo_url, token=token)

        # Step 4: Verify the ID token to get the user's email and other details
        id_token = token_info.get("id_token")
        if id_token:
            claims = jwt.decode(id_token, DEX_CLIENT_SECRET, algorithms=["HS256"])
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
