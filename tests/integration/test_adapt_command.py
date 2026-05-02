"""Integration tests for `platinum adapt`."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


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


def _seed_curated_story(project: Path, story_id: str) -> Path:
    from platinum.models.story import (
        Source,
        StageRun,
        StageStatus,
        Story,
    )
    s = Story(
        id=story_id, track="atmospheric_horror",
        source=Source(
            type="gutenberg", url=f"https://example/{story_id}",
            title=f"Title {story_id}", author="A", raw_text="raw...",
            fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
        stages=[
            StageRun(stage="source_fetcher", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25)),
            StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25),
                      artifacts={"decision": "approved"}),
        ],
    )
    story_dir = project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")
    return story_dir / "story.json"


def _router_factory() -> Any:
    """Return a recorder that dispatches by tool name."""
    async def router(req: dict[str, Any]) -> dict[str, Any]:
        name = req["tool_choice"]["name"]
        if name == "submit_adapted_story":
            return {
                "id": "ad", "content": [{"type": "tool_use", "name": name, "input": {
                    "title": "T", "synopsis": "S", "narration_script": "word " * 1300,
                    "tone_notes": "n",
                    "arc": {"setup":"a","rising":"b","climax":"c","resolution":"d"},
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if name == "submit_scene_breakdown":
            return {
                "id": "br", "content": [{"type": "tool_use", "name": name, "input": {
                    "scenes": [{"index": i, "narration_text": " ".join(["w"] * 162),
                                 "mood": "ambient_drone", "sfx_cues": []}
                                for i in range(1, 9)],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if name == "submit_story_bible":
            # S8.B: synthetic bible covering the 8 scenes scene_breakdown emits.
            return {
                "id": "bb", "content": [{"type": "tool_use", "name": name, "input": {
                    "world_genre_atmosphere": "carnival to catacombs.",
                    "character_continuity": {
                        "Montresor": {"face": "lean", "costume": "black cloak",
                                       "posture": "patient"},
                    },
                    "environment_continuity": {"palazzo": "ash-grey wall"},
                    "scenes": [
                        {
                            "index": i,
                            "narrative_beat": f"beat {i}",
                            "hero_shot": "medium shot",
                            "visible_characters": ["Montresor"],
                            "gaze_map": {"Montresor": "off-camera"},
                            "props_visible": ["candle"],
                            "blocking": "centered",
                            "light_source": "single beeswax candle",
                            "color_anchors": ["black"],
                            "brightness_floor": "low",
                        }
                        for i in range(1, 9)
                    ],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        return {
            "id": "vp", "content": [{"type": "tool_use", "name": name, "input": {
                # "Montresor" must appear in each visual_prompt to satisfy the
                # S8.B.5 post-condition (the synthesized bible above declares
                # visible_characters=["Montresor"] for every scene).
                "scenes": [{"index": i,
                             "visual_prompt": f"vp{i} Montresor in candlelit study",
                             "negative_prompt": f"np{i}"}
                            for i in range(1, 9)],
            }}], "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }
    return router


def _wire_recorder_into_config(
    monkeypatch: pytest.MonkeyPatch, cli_mod: Any, recorder: Any
) -> None:
    """Subclass Config so its __init__ stashes the recorder in settings."""
    original_init = cli_mod.Config.__init__

    def init_with_recorder(self: Any, root: Any = None) -> None:
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = recorder

    monkeypatch.setattr(
        cli_mod, "Config",
        type("C", (cli_mod.Config,), {"__init__": init_with_recorder}),
    )


def test_cli_adapt_walks_eligible_stories(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`platinum adapt` adapts every curator-approved-but-not-yet-adapted story."""
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    _wire_recorder_into_config(monkeypatch, cli_mod, _router_factory())

    _seed_curated_story(cli_project, "story_a")
    _seed_curated_story(cli_project, "story_b")

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt"])
    assert result.exit_code == 0, result.output

    for sid in ("story_a", "story_b"):
        data_path = (
            cli_project / "data" / "stories" / sid / "story.json"
        )
        data = json.loads(data_path.read_text(encoding="utf-8"))
        assert data["adapted"] is not None
        assert len(data["scenes"]) == 8
        assert all(s.get("visual_prompt") for s in data["scenes"])


def test_cli_adapt_no_eligible_stories_exits_zero(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt"])
    assert result.exit_code == 0
    assert "no eligible stories" in result.output.lower()


def test_cli_adapt_story_filter_targets_one(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from platinum import cli as cli_mod
    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    _wire_recorder_into_config(monkeypatch, cli_mod, _router_factory())

    _seed_curated_story(cli_project, "story_a")
    _seed_curated_story(cli_project, "story_b")

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt", "--story", "story_a"])
    assert result.exit_code == 0

    a = json.loads(
        (cli_project / "data" / "stories" / "story_a" / "story.json").read_text(
            encoding="utf-8"
        )
    )
    b = json.loads(
        (cli_project / "data" / "stories" / "story_b" / "story.json").read_text(
            encoding="utf-8"
        )
    )
    assert a["adapted"] is not None
    assert b["adapted"] is None


def test_cli_adapt_then_status_reflects_complete(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    original_init = cli_mod.Config.__init__

    def init_with_recorder(self: Any, root: Any = None) -> None:
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = _router_factory()

    monkeypatch.setattr(
        cli_mod, "Config",
        type("C", (cli_mod.Config,), {"__init__": init_with_recorder}),
    )

    _seed_curated_story(cli_project, "story_x")

    runner = CliRunner()
    assert runner.invoke(cli_mod.app, ["adapt"]).exit_code == 0

    status = runner.invoke(cli_mod.app, ["status", "--story", "story_x"])
    assert status.exit_code == 0
    assert "story_adapter" in status.output
    assert status.output.count("COMPLETE") >= 3
