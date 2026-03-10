from __future__ import annotations

from flask import Flask, jsonify

from api.db_endpoints import db_blueprint


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
    )

    app.register_blueprint(db_blueprint, url_prefix="/api/db")

    @app.get("/api/health")
    def healthcheck():
        return jsonify({"status": "ok"})

    return app
