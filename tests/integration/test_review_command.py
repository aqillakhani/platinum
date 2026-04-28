"""Integration tests for `platinum review keyframes`.

S7 §5.1 / §6.2.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from platinum.cli import app
from platinum.config import Config
from platinum.models.story import Adapted, Scene, Source, Story


@pytest.fixture
def cask_story_factory(tmp_path: Path, monkeypatch):
    """Build a story on disk under tmp_path/data/stories/, return its id."""
    # Patch Config to use tmp_path as root
    monkeypatch.setattr(
        "platinum.cli.Config",
        lambda: Config(root=tmp_path),
    )
    def _make() -> str:
        src = Source(
            type="gutenberg", url="https://example.com",
            title="Cask", author="Poe", raw_text="hello",
            fetched_at=datetime.now(UTC), license="PD-US",
        )
        adapted = Adapted(
            title="Cask", synopsis="x", narration_script="y",
            estimated_duration_seconds=600.0, tone_notes="z",
        )
        scenes = [
            Scene(id=f"scene_{i+1:03d}", index=i+1, narration_text=f"s{i}",
                  visual_prompt=f"p{i}", negative_prompt="bright daylight")
            for i in range(2)
        ]
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
        )
        d = tmp_path / "data" / "stories" / story.id
        d.mkdir(parents=True, exist_ok=True)
        story.save(d / "story.json")
        return story.id
    return _make


def test_review_keyframes_missing_story_exit_1(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "platinum.cli.Config",
        lambda: Config(root=tmp_path),
    )
    (tmp_path / "data" / "stories").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(app, ["review", "keyframes", "story_does_not_exist"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_review_keyframes_no_browser_skips_open(cask_story_factory) -> None:
    story_id = cask_story_factory()
    runner = CliRunner()
    with patch("webbrowser.open") as mock_open, \
         patch("flask.Flask.run") as mock_run:  # do not actually run the server
        result = runner.invoke(
            app, ["review", "keyframes", story_id, "--no-browser"],
        )
    assert result.exit_code == 0, result.output
    mock_open.assert_not_called()
    mock_run.assert_called_once()
