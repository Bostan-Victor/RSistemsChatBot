from __future__ import annotations

from flask import Flask, render_template

from .config import Config
from .extensions import cors, db, migrate


def create_app(config_object: type[Config] = Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)

    cors.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)

    from .api import register_blueprints

    register_blueprints(app)

    @app.get("/")
    def index():
        return render_template("index.html")

    return app
