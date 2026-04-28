"""Flask app for the keyframe review UI.

S7 §3 -- local 127.0.0.1 only, no auth, single-user. The app factory
takes a story_id (required at boot -- one story per process) plus
data_root and returns a configured Flask instance.

Routes are added in subsequent tasks; this skeleton ships only the
factory + healthcheck.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify


def create_app(*, story_id: str, data_root: Path) -> Flask:
    """Build a Flask app bound to one story + the data_root containing
    its keyframes/ subtree.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["STORY_ID"] = story_id
    app.config["DATA_ROOT"] = Path(data_root)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    return app
