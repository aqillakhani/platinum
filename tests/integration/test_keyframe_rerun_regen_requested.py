"""Integration tests for `platinum keyframes --rerun-regen-requested`.

S7 §5.2.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from platinum.cli import app
from platinum.models.story import (
    Adapted,
    ReviewStatus,
    Scene,
    Source,
    StageRun,
    StageStatus,
    Story,
)


@pytest.fixture
def cli_project(tmp_path: Path) -> Iterator[Path]:
    """Mirror the real project layout under tmp_path for CLI tests."""
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copytree(repo_root / "config" / "prompts", tmp_path / "config" / "prompts")
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data" / "stories").mkdir(parents=True)
    yield tmp_path


@pytest.fixture
def cask_story_factory(cli_project: Path, monkeypatch):
    """Build a story with visual_prompts COMPLETE, varied review statuses."""
    monkeypatch.chdir(cli_project)

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
                  visual_prompt=f"p{i}", negative_prompt="bright daylight",
                  keyframe_path=Path(f"scene_{i+1:03d}/candidate_0.png")
                  if i != 1 else None,
                  review_status=(
                      ReviewStatus.APPROVED if i == 0
                      else ReviewStatus.REGENERATE if i == 1
                      else ReviewStatus.PENDING
                  ),
                  regen_count=1 if i == 1 else 0)
            for i in range(3)
        ]
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
            stages=[
                StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(UTC)),
            ],
        )
        d = cli_project / "data" / "stories" / story.id
        d.mkdir(parents=True, exist_ok=True)
        story.save(d / "story.json")
        return story.id

    return _make


def test_rerun_regen_requested_filters_to_REGENERATE_status(
    cask_story_factory, cli_project: Path, monkeypatch,
) -> None:
    """--rerun-regen-requested builds scene_filter from REGENERATE-status only."""
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    story_id = cask_story_factory()
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["keyframes", story_id, "--rerun-regen-requested", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    # Dry-run prints the planned scene set; should be exactly [2] (the REGENERATE scene)
    assert "Would generate keyframes for scenes [2]" in result.output


def test_rerun_regen_requested_empty_set_exits_zero(
    cli_project: Path, monkeypatch,
) -> None:
    """If no scenes are flagged REGENERATE, exit 0 with helpful message."""
    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    src = Source(type="gutenberg", url="https://example.com", title="t",
                 author="a", raw_text="x",
                 fetched_at=datetime.now(UTC), license="PD-US")
    adapted = Adapted(title="t", synopsis="x", narration_script="y",
                      estimated_duration_seconds=600.0, tone_notes="z")
    story = Story(
        id="story_x", track="atmospheric_horror",
        source=src, adapted=adapted,
        scenes=[
            Scene(id="scene_001", index=1, narration_text="x",
                  visual_prompt="p", negative_prompt="bright daylight",
                  review_status=ReviewStatus.APPROVED),
        ],
        stages=[
            StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                     completed_at=datetime.now(UTC)),
        ],
    )
    d = cli_project / "data" / "stories" / story.id
    d.mkdir(parents=True, exist_ok=True)
    story.save(d / "story.json")

    runner = CliRunner()
    result = runner.invoke(
        app, ["keyframes", story.id, "--rerun-regen-requested"],
    )
    assert result.exit_code == 0, result.output
    assert "no scenes flagged" in result.output.lower()
