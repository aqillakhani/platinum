"""Flask app for the keyframe review UI.

S7 §3 -- local 127.0.0.1 only, no auth, single-user. The app factory
takes a story_id (required at boot -- one story per process) plus
data_root and returns a configured Flask instance.

Routes are added in subsequent tasks; this skeleton ships only the
factory + healthcheck.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify

from platinum.models.story import ReviewStatus, Story


def _story_path(data_root: Path, story_id: str) -> Path:
    return data_root / story_id / "story.json"


def _load_story_or_404(data_root: Path, story_id: str) -> Story:
    path = _story_path(data_root, story_id)
    if not path.exists():
        abort(404, description=f"story not found: {story_id}")
    return Story.load(path)


def _rollup(story: Story) -> dict[str, int]:
    counts = {"pending": 0, "approved": 0, "rejected": 0, "regen_requested": 0}
    for scene in story.scenes:
        if scene.review_status == ReviewStatus.PENDING:
            counts["pending"] += 1
        elif scene.review_status == ReviewStatus.APPROVED:
            counts["approved"] += 1
        elif scene.review_status == ReviewStatus.REJECTED:
            counts["rejected"] += 1
        elif scene.review_status == ReviewStatus.REGENERATE:
            counts["regen_requested"] += 1
    return counts


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

    @app.get("/api/story/<story_id>")
    def api_story(story_id: str):
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        body = story.to_dict()
        body["rollup"] = _rollup(story)
        return jsonify(body)

    return app
