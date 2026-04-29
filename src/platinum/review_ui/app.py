"""Flask app for the keyframe review UI.

S7 §3 -- local 127.0.0.1 only, no auth, single-user. The app factory
takes a story_id (required at boot -- one story per process) plus
data_root and returns a configured Flask instance.

Routes are added in subsequent tasks; this skeleton ships only the
factory + healthcheck.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, send_file, url_for
from werkzeug.security import safe_join

from platinum.models.story import ReviewStatus, Story
from platinum.review_ui import decisions


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

    @app.get("/image/<story_id>/<path:relpath>")
    def image(story_id: str, relpath: str):
        keyframes_root = app.config["DATA_ROOT"] / story_id / "keyframes"
        full = safe_join(str(keyframes_root), relpath)
        if full is None:
            abort(404)
        full_path = Path(full)
        if not full_path.exists() or not full_path.is_file():
            abort(404)
        return send_file(full_path)

    @app.get("/reference_image/<story_id>/<path:relpath>")
    def reference_image(story_id: str, relpath: str):
        """S7.1.B6.2: serves files under <story>/references/<character>/.

        Mirrors /image/ but rooted at the references subtree so the
        character gallery template can reach candidate PNGs without
        leaking the project layout to the client.
        """
        refs_root = app.config["DATA_ROOT"] / story_id / "references"
        full = safe_join(str(refs_root), relpath)
        if full is None:
            abort(404)
        full_path = Path(full)
        if not full_path.exists() or not full_path.is_file():
            abort(404)
        return send_file(full_path)

    def _save_and_respond(story: Story, *, scene_id: str | None = None):
        """Common tail: finalize -> save -> return JSON of touched scene + rollup."""
        decisions.finalize_review_if_complete(story)
        story.save(_story_path(app.config["DATA_ROOT"], story.id))
        body: dict = {"rollup": _rollup(story)}
        if scene_id is not None:
            for sc in story.scenes:
                if sc.id == scene_id:
                    body["scene"] = sc.to_dict()
                    break
        return jsonify(body)

    def _scene_relpath(scene) -> str:
        """The keyframe_path is stored absolute or relative; return a relpath
        usable in url_for('image', relpath=...). Falls back to filename."""
        if scene.keyframe_path is None:
            return ""
        return f"scene_{scene.index:03d}/{Path(scene.keyframe_path).name}"

    def _candidate_relpath(scene, idx: int) -> str:
        return f"scene_{scene.index:03d}/candidate_{idx}.png"

    def _selected_score(scene) -> float | None:
        if scene.keyframe_path is None:
            return None
        try:
            i = scene.keyframe_candidates.index(scene.keyframe_path)
        except ValueError:
            return None
        if i >= len(scene.keyframe_scores):
            return None
        return scene.keyframe_scores[i]

    @app.get("/")
    def index():
        return redirect(url_for("story", story_id=app.config["STORY_ID"]))

    @app.get("/story/<story_id>")
    def story(story_id: str):
        story_obj = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        return render_template(
            "keyframe_gallery.html",
            story=story_obj,
            rollup=_rollup(story_obj),
            default_threshold=app.config.get("DEFAULT_THRESHOLD", 6.0),
            scene_relpath=_scene_relpath,
            candidate_relpath=_candidate_relpath,
            selected_score_for=_selected_score,
        )

    @app.get("/story/<story_id>/characters")
    def character_gallery(story_id: str):
        """S7.1.B6.2: per-character ref-image gallery.

        Discovers character names from scene.character_refs union and lists
        every candidate PNG under <story>/references/<character>/. Each
        candidate gets a Pick button which (B6.4) POSTs to
        /api/story/<id>/select_character_reference.
        """
        story_obj = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        # Discover character names that appear in any scene.
        discovered = sorted({
            n for s in story_obj.scenes for n in (s.character_refs or [])
        })
        # For each character with a refs/<name>/ dir, list candidate PNGs.
        chars_with_candidates: dict[str, list[str]] = {}
        for name in discovered:
            refs_dir = (
                app.config["DATA_ROOT"] / story_id / "references" / name
            )
            if not refs_dir.exists():
                continue
            paths = sorted(
                str(p) for p in refs_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            )
            if paths:
                chars_with_candidates[name] = paths
        return render_template(
            "character_gallery.html",
            story=story_obj,
            characters_with_candidates=chars_with_candidates,
        )

    @app.post("/api/story/<story_id>/scene/<scene_id>/approve")
    def post_approve(story_id: str, scene_id: str):
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_approve(story, scene_id)
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/scene/<scene_id>/regenerate")
    def post_regenerate(story_id: str, scene_id: str):
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_regenerate(story, scene_id)
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/scene/<scene_id>/reject")
    def post_reject(story_id: str, scene_id: str):
        from flask import request
        body = request.get_json(silent=True) or {}
        feedback = body.get("feedback", "")
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_reject(story, scene_id, feedback=feedback)
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        except ValueError as exc:
            abort(400, description=str(exc))
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/scene/<scene_id>/select_candidate")
    def post_select_candidate(story_id: str, scene_id: str):
        from flask import request
        body = request.get_json(silent=True) or {}
        if "index" not in body:
            abort(400, description="'index' field required")
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_swap_candidate(
                story, scene_id, candidate_index=int(body["index"])
            )
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        except IndexError as exc:
            abort(400, description=str(exc))
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/batch_approve")
    def post_batch_approve(story_id: str):
        from flask import request
        body = request.get_json(silent=True) or {}
        if "threshold" not in body:
            abort(400, description="'threshold' field required")
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        decisions.apply_batch_approve_above(
            story, threshold=float(body["threshold"])
        )
        return _save_and_respond(story)

    return app
