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
