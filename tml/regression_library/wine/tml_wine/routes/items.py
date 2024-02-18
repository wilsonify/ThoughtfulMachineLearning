from typing import Annotated

from fastapi import Depends

from tml_wine.auth.auth import oauth2_scheme
from tml_wine.main import app


@app.get("/items/")
async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"token": token}
