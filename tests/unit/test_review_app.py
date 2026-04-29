"""Flask app + route tests via app.test_client().

S7 §3.3 / §6.2.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from platinum.models.story import (
    Adapted,
    Scene,
    Source,
    Story,
)


@pytest.fixture
def story_factory(tmp_path: Path):
    """Build a story on disk under tmp_path/data/stories/<id>/, return (story, story_dir)."""
    def _make(*, n_scenes: int = 3) -> tuple[Story, Path]:
        src = Source(
            type="gutenberg", url="https://example.com",
            title="Test", author="A", raw_text="hello",
            fetched_at=datetime.now(UTC), license="PD-US",
        )
        adapted = Adapted(
            title="Test", synopsis="x", narration_script="y",
            estimated_duration_seconds=600.0, tone_notes="z",
        )
        scenes = []
        for i in range(n_scenes):
            scene = Scene(
                id=f"scene_{i+1:03d}", index=i + 1,
                narration_text=f"scene {i}",
                visual_prompt=f"prompt {i}",
                negative_prompt="bright daylight",
            )
            scenes.append(scene)
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
        )
        story_dir = tmp_path / "data" / "stories" / story.id
        story_dir.mkdir(parents=True, exist_ok=True)
        story.save(story_dir / "story.json")
        return story, story_dir
    return _make


def test_app_factory_creates_app(story_factory, tmp_path: Path) -> None:
    """create_app(story_id, data_root) returns a Flask app instance."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=3)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    assert app is not None
    assert app.config["STORY_ID"] == story.id


def test_health_check_route_returns_200(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json == {"status": "ok"}


def test_get_api_story_returns_snapshot(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory(n_scenes=3)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/api/story/{story.id}")
    assert resp.status_code == 200
    data = resp.json
    assert data["id"] == story.id
    assert len(data["scenes"]) == 3
    assert data["rollup"]["pending"] == 3
    assert data["rollup"]["approved"] == 0


def test_get_api_story_404_on_missing(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id="story_doesnt_exist",
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get("/api/story/story_doesnt_exist")
    assert resp.status_code == 404


def test_get_image_serves_png(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    keyframes_dir = story_dir / "keyframes" / "scene_001"
    keyframes_dir.mkdir(parents=True)
    png_path = keyframes_dir / "candidate_0.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/image/{story.id}/scene_001/candidate_0.png")
    assert resp.status_code == 200
    assert resp.data == b"\x89PNG\r\n\x1a\nfake"


def test_get_image_404_on_missing(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/image/{story.id}/nonexistent/x.png")
    assert resp.status_code == 404


def test_get_image_blocks_path_traversal(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    secret = tmp_path / "secret.txt"
    secret.write_text("hunter2")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/image/{story.id}/../../../secret.txt")
    # safe_join returns None for traversal -> 404
    assert resp.status_code == 404


def test_post_approve_persists_to_disk(story_factory, tmp_path: Path) -> None:
    from platinum.models.story import ReviewStatus, Story
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=3)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(f"/api/story/{story.id}/scene/scene_001/approve")
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].review_status == ReviewStatus.APPROVED
    assert resp.json["rollup"]["approved"] == 1


def test_post_regenerate_bumps_count_and_clears_keyframe(story_factory, tmp_path: Path) -> None:
    from platinum.models.story import Story
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    # Give scene_001 a keyframe_path to clear
    story.scenes[0].keyframe_path = Path("scene_001/candidate_0.png")
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(f"/api/story/{story.id}/scene/scene_001/regenerate")
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].keyframe_path is None
    assert rt.scenes[0].regen_count == 1


def test_post_reject_requires_feedback(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/scene/scene_001/reject",
            json={},  # no feedback field
        )
    assert resp.status_code == 400


def test_post_reject_persists_feedback(story_factory, tmp_path: Path) -> None:
    from platinum.models.story import ReviewStatus, Story
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/scene/scene_001/reject",
            json={"feedback": "scene 1 face needs amber"},
        )
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].review_feedback == "scene 1 face needs amber"
    assert rt.scenes[0].review_status == ReviewStatus.REJECTED


def test_post_select_candidate_swaps_keyframe_path(story_factory, tmp_path: Path) -> None:
    from platinum.models.story import Story
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    # Give scene 0 candidate paths
    story.scenes[0].keyframe_candidates = [
        Path("scene_001/candidate_0.png"),
        Path("scene_001/candidate_1.png"),
        Path("scene_001/candidate_2.png"),
    ]
    story.scenes[0].keyframe_scores = [5.5, 6.2, 5.9]
    story.scenes[0].keyframe_path = story.scenes[0].keyframe_candidates[1]
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/scene/scene_001/select_candidate",
            json={"index": 0},
        )
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].keyframe_path == Path("scene_001/candidate_0.png")


def test_post_batch_approve_marks_pending_above_threshold(story_factory, tmp_path: Path) -> None:
    from platinum.models.story import ReviewStatus, Story
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=2)
    # scene 0 selected score = 6.2 (above 6.0)
    story.scenes[0].keyframe_candidates = [Path("a.png"), Path("b.png")]
    story.scenes[0].keyframe_scores = [5.0, 6.2]
    story.scenes[0].keyframe_path = story.scenes[0].keyframe_candidates[1]
    # scene 1 selected score = 5.5 (below 6.0)
    story.scenes[1].keyframe_candidates = [Path("c.png")]
    story.scenes[1].keyframe_scores = [5.5]
    story.scenes[1].keyframe_path = story.scenes[1].keyframe_candidates[0]
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/batch_approve",
            json={"threshold": 6.0},
        )
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].review_status == ReviewStatus.APPROVED
    assert rt.scenes[1].review_status == ReviewStatus.PENDING


def test_post_finalizes_when_all_approved(story_factory, tmp_path: Path) -> None:
    """The last approve should append a keyframe_review StageRun."""
    from platinum.models.story import StageStatus, Story
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=2)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        client.post(f"/api/story/{story.id}/scene/scene_001/approve")
        resp = client.post(f"/api/story/{story.id}/scene/scene_002/approve")
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    run = rt.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE


def test_get_story_renders_template(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory(n_scenes=2)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/story/{story.id}")
    assert resp.status_code == 200
    assert b"Keyframe Review" in resp.data
    assert story.id.encode() in resp.data
    # Each scene appears
    assert b"scene_001" in resp.data
    assert b"scene_002" in resp.data


def test_get_root_redirects_to_story(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get("/")
    assert resp.status_code == 302
    assert f"/story/{story.id}" in resp.location


# ---- B6.2: GET /story/<id>/characters --------------------------------------


def _seed_character_refs(
    story: Story, story_dir: Path, *,
    character: str, n_candidates: int = 3,
) -> list[Path]:
    """Drop n_candidates fake PNGs at <story>/references/<character>/.

    Returns the candidate paths in deterministic order (candidate_0..N-1).
    """
    refs_dir = story_dir / "references" / character
    refs_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_candidates):
        p = refs_dir / f"candidate_{i}.png"
        p.write_bytes(b"\x89PNG_test_" + str(i).encode())
        paths.append(p)
    return paths


def test_get_character_gallery_renders_row_per_character(
    story_factory, tmp_path: Path
) -> None:
    """S7.1.B6.2: GET /story/<id>/characters returns 200 with one section
    per character that has candidates on disk."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=2)
    story.scenes[0].character_refs = ["Fortunato", "Montresor"]
    story.scenes[1].character_refs = ["Fortunato"]
    story.save(story_dir / "story.json")

    _seed_character_refs(story, story_dir, character="Fortunato")
    _seed_character_refs(story, story_dir, character="Montresor")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/story/{story.id}/characters")
    assert resp.status_code == 200, resp.data
    # Both characters appear as section headers.
    assert b"Fortunato" in resp.data
    assert b"Montresor" in resp.data
    # Pick buttons rendered (3 per character).
    pick_count = resp.data.count(b"data-character=")
    assert pick_count >= 6  # 3 buttons * 2 characters


def test_get_character_gallery_shows_picked_marker_for_already_picked(
    story_factory, tmp_path: Path
) -> None:
    """S7.1.B6.2: when story.characters[name] is set, the Pick button for
    that ref is disabled / shows 'Picked'."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=1)
    story.scenes[0].character_refs = ["Fortunato"]
    paths = _seed_character_refs(story, story_dir, character="Fortunato")
    story.characters["Fortunato"] = str(paths[1])  # picked candidate_1
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/story/{story.id}/characters")
    assert resp.status_code == 200
    assert b"Picked" in resp.data


def test_reference_image_serves_candidate_png(
    story_factory, tmp_path: Path
) -> None:
    """S7.1.B6.2: /reference_image/<story>/<character>/<file> serves the
    candidate PNG from the references directory."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=1)
    story.scenes[0].character_refs = ["Fortunato"]
    story.save(story_dir / "story.json")
    paths = _seed_character_refs(story, story_dir, character="Fortunato")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(
            f"/reference_image/{story.id}/Fortunato/candidate_0.png"
        )
    assert resp.status_code == 200
    assert resp.data == paths[0].read_bytes()


def test_reference_image_404_on_missing(story_factory, tmp_path: Path) -> None:
    """S7.1.B6.2: nonexistent reference path returns 404."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=1)

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(
            f"/reference_image/{story.id}/Ghost/missing.png"
        )
    assert resp.status_code == 404


# ---- B6.3: POST /api/story/<id>/select_character_reference -----------------


def test_post_select_character_reference_updates_story(
    story_factory, tmp_path: Path
) -> None:
    """S7.1.B6.3: POST happy path -- writes story.characters[name] and persists."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=1)
    story.scenes[0].character_refs = ["Fortunato"]
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/select_character_reference",
            json={
                "character": "Fortunato",
                "path": "references/Fortunato/candidate_2.png",
            },
        )
    assert resp.status_code == 200, resp.data
    body = resp.get_json()
    assert body["characters"]["Fortunato"] == "references/Fortunato/candidate_2.png"

    # Persisted on disk -- reload story.json and verify.
    reloaded = Story.load(story_dir / "story.json")
    assert reloaded.characters["Fortunato"] == (
        "references/Fortunato/candidate_2.png"
    )


def test_post_select_character_reference_unknown_character_returns_400(
    story_factory, tmp_path: Path
) -> None:
    """S7.1.B6.3: ValueError -> HTTP 400 with message; story unchanged."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=1)
    story.scenes[0].character_refs = ["Fortunato"]
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/select_character_reference",
            json={
                "character": "Lord Verres",  # not in any scene
                "path": "references/Lord Verres/candidate_0.png",
            },
        )
    assert resp.status_code == 400

    # No persistence side effects.
    reloaded = Story.load(story_dir / "story.json")
    assert "Lord Verres" not in reloaded.characters


def test_post_select_character_reference_missing_fields_returns_400(
    story_factory, tmp_path: Path
) -> None:
    """S7.1.B6.3: missing 'character' or 'path' fields -> 400."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=1)

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/select_character_reference",
            json={"character": "Fortunato"},  # no 'path'
        )
    assert resp.status_code == 400
