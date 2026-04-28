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
