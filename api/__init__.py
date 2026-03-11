from __future__ import annotations

import os

from flask import Flask, jsonify, request

from api.db_endpoints import db_blueprint


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)


def get_allowed_origins() -> set[str]:
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS")
    if not configured_origins:
        return set(DEFAULT_CORS_ORIGINS)

    return {
        origin.strip() for origin in configured_origins.split(",") if origin.strip()
    }


def is_allowed_origin(origin: str | None, allowed_origins: set[str]) -> bool:
    if not origin:
        return False
    if "*" in allowed_origins:
        return True
    return origin in allowed_origins


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        CORS_ALLOWED_ORIGINS=get_allowed_origins(),
    )

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        allowed_origins = app.config["CORS_ALLOWED_ORIGINS"]
        if is_allowed_origin(origin, allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = request.headers.get(
                "Access-Control-Request-Headers",
                "Content-Type, Authorization",
            )
            response.headers["Access-Control-Max-Age"] = "86400"
        return response

    app.register_blueprint(db_blueprint, url_prefix="/api/db")

    @app.get("/api/health")
    def healthcheck():
        return jsonify({"status": "ok"})

    return app
