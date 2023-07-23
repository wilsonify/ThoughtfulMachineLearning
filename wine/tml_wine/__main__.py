import uvicorn

from tml_wine import auth
from tml_wine.main import app
from tml_wine.routes import items
from tml_wine.routes import predict
from tml_wine.routes import token
from tml_wine.routes import users

dir(auth)
dir(items)
dir(predict)
dir(token)
dir(users)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
