from flask import Flask
import os
from app.config import Config
from app.routes.bugs import bugs as bugs_blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    # Register blueprint
    app.register_blueprint(bugs_blueprint)

    # Pastikan folder model ada
    os.makedirs(os.path.join('app', 'utils', 'model'), exist_ok=True)

    return app
