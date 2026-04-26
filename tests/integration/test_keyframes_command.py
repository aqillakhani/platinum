"""Integration tests for `platinum keyframes` CLI command."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from platinum.cli import app


@pytest.fixture
def cli_project(tmp_path: Path) -> Iterator[Path]:
    """Mirror the real project layout under tmp_path for keyframes CLI tests."""
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    shutil.copytree(
        repo_root / "config" / "workflows",
        tmp_path / "config" / "workflows",
        dirs_exist_ok=True,
    )
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data" / "stories").mkdir(parents=True)
    yield tmp_path


def _seed_adapted_story(project: Path, story_id: str, n_scenes: int = 3) -> Path:
    """Build a Story with N scenes, all visual_prompts populated, visual_prompts COMPLETE."""
    from platinum.models.story import (
        Scene,
        Source,
        StageRun,
        StageStatus,
        Story,
    )

    s = Story(
        id=story_id,
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url=f"https://example/{story_id}",
            title=f"Title {story_id}",
            author="A",
            raw_text="raw...",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=[
            Scene(
                id=f"scene_{i:03d}",
                index=i,
                narration_text=f"Narration {i}.",
                visual_prompt=f"prompt {i}",
                negative_prompt="bright daylight",
            )
            for i in range(n_scenes)
        ],
        stages=[
            StageRun(
                stage="visual_prompts",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 25),
                completed_at=datetime(2026, 4, 25),
            )
        ],
    )
    story_dir = project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")
    return story_dir / "story.json"


def test_keyframes_dry_run_prints_plan_and_exits_zero(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(cli_project)
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://test:8188")
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://test:8189")

    # Patch Config to use cli_project as root (Config defaults to package's _ROOT)
    from platinum import cli as cli_mod

    original_init = cli_mod.Config.__init__

    def init_with_root(self, root=None):  # type: ignore[no-untyped-def]
        original_init(self, root=cli_project)

    monkeypatch.setattr(cli_mod.Config, "__init__", init_with_root)

    _seed_adapted_story(cli_project, "TEST_STORY", n_scenes=3)

    runner = CliRunner()
    result = runner.invoke(
        app, ["keyframes", "TEST_STORY", "--scenes", "0,2", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "0" in result.output
    assert "2" in result.output
    assert "test:8188" in result.output or "test:8189" in result.output
