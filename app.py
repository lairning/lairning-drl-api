import os
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager
)
from flask_restful import Api
from flask_injector import FlaskInjector, singleton

from drl.api import API_HANDLERS

from lairning_core import DRLServer

DATABASE_TYPE = os.environ.get("DATABASE_TYPE", "mysql")
DATABASE_HOST = os.environ.get("DATABASE_HOST", "127.0.0.1")
DATABASE_USER = os.environ.get("DATABASE_USER", "root")
DATABASE_PWD = os.environ.get("DATABASE_PWD", "root")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "Pocosi12!")
DATABASE_FILE = os.environ.get("DATABASE_FILE", "database.sqlite")
JWT_SECRET = os.environ.get("JWT_SECRET", "Pocosi12!")
PORT = int(os.environ.get("PORT", "5002"))

print(f"DATABASE_TYPE: {DATABASE_TYPE}")
print(f"DATABASE_HOST:  {DATABASE_HOST}")
print(f"DATABASE_USER: {DATABASE_USER}")
print(f"DATABASE_PWD: {DATABASE_PWD}")
print(f"DATABASE_NAME: {DATABASE_NAME}")
print(f"DATABASE_FILE: {DATABASE_FILE}")
print(f"JWT_SECRET: {JWT_SECRET}")
print(f"PORT:  {PORT}")

API_PREFIX = "/v1"
app = Flask(__name__)
CORS(app)
api = Api(app, prefix=API_PREFIX)

for handler in API_HANDLERS:
    handler.decorators = handler.DECORATORS
    api.add_resource(handler, handler.ENDPOINT)

def configure(binder):
    binder.bind(DRLServer, scope=singleton)

injector = FlaskInjector(app=app, modules=[configure])

app.config["JWT_SECRET_KEY"] = JWT_SECRET
jwt = JWTManager(app)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=PORT)
