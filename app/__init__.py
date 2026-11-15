"""Flask application factory."""
from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

import config
from app import database


def create_app(testing: bool = False, preload_models: bool = True) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024, TESTING=testing,
                      JSON_SORT_KEYS=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    database.init_db()

    from app.routes import api, pages
    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")

    @app.context_processor
    def inject_globals():
        return {"config_max": config.MAX_UPLOAD_MB}

    def wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(413)
    def too_large(_e):
        msg = f"File too large (limit {config.MAX_UPLOAD_MB} MB)."
        return (jsonify(error=msg), 413) if wants_json() else (render_template("error.html", message=msg), 413)

    @app.errorhandler(404)
    def not_found(_e):
        return (jsonify(error="Not found"), 404) if wants_json() else (
            render_template("error.html", message="Page not found."), 404)

    @app.errorhandler(500)
    def server_error(_e):
        return (jsonify(error="Internal server error"), 500) if wants_json() else (
            render_template("error.html", message="Internal server error."), 500)

    if preload_models:
        from ml.service import get_service
        get_service()
    return app
