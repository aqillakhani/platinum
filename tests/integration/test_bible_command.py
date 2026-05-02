"""Integration tests for `platinum bible <story_id>` CLI command (S8.B.4).

The bible command runs ``StoryBibleStage`` standalone for a story whose
scene_breakdown has completed. The standalone path is used to (a) bootstrap
old stories that pre-date S8.B, and (b) regenerate a bible after a prompt
revision via ``--rerun``.
"""
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
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copytree(repo_root / "config" / "prompts", tmp_path / "config" / "prompts")
    for track_yaml in (
        "atmospheric_horror.yaml",
        "childrens_fables.yaml",
    ):
        shutil.copy(
            repo_root / "config" / "tracks" / track_yaml,
            tmp_path / "config" / "tracks" / track_yaml,
        )
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data" / "stories").mkdir(parents=True)
    yield tmp_path


def _seed_scene_broken_story(
    project: Path, story_id: str, *, track: str = "atmospheric_horror",
    n_scenes: int = 4,
) -> Path:
    """Build a Story whose scene_breakdown has produced N scenes — the
    minimum precondition for running platinum bible."""
    from platinum.models.story import (
        Adapted,
        Scene,
        Source,
        StageRun,
        StageStatus,
        Story,
    )

    s = Story(
        id=story_id, track=track,
        source=Source(
            type="gutenberg", url=f"https://example/{story_id}",
            title=f"Title {story_id}", author="A", raw_text="raw...",
            fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
        adapted=Adapted(
            title="T", synopsis="S",
            narration_script="word " * 200,
            estimated_duration_seconds=600.0,
            tone_notes="n",
            arc={"setup": "a", "rising": "b", "climax": "c", "resolution": "d"},
        ),
        scenes=[
            Scene(id=f"scene_{i:03d}", index=i,
                  narration_text=f"Narration {i}.")
            for i in range(1, n_scenes + 1)
        ],
        stages=[
            StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25),
                      artifacts={"decision": "approved"}),
            StageRun(stage="story_adapter", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25)),
            StageRun(stage="scene_breakdown", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25)),
        ],
    )
    story_dir = project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")
    return story_dir / "story.json"


def _bible_router_factory(*, n_scenes: int = 4) -> Any:
    """Return an async recorder that responds with a synthesized bible."""
    async def router(req: dict[str, Any]) -> dict[str, Any]:
        name = req["tool_choice"]["name"]
        assert name == "submit_story_bible", name  # bible cmd must call only this tool
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
                    for i in range(1, n_scenes + 1)
                ],
            }}], "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }
    return router


def _wire_recorder_into_config(
    monkeypatch: pytest.MonkeyPatch, cli_mod: Any, recorder: Any
) -> None:
    original_init = cli_mod.Config.__init__

    def init_with_recorder(self: Any, root: Any = None) -> None:
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = recorder

    monkeypatch.setattr(
        cli_mod, "Config",
        type("C", (cli_mod.Config,), {"__init__": init_with_recorder}),
    )


def test_bible_command_missing_story_exits_1(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["bible", "DOES_NOT_EXIST"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_bible_command_track_with_bible_disabled_exits_1(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracks that haven't opted into the bible pre-pass should refuse cleanly."""
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    _seed_scene_broken_story(cli_project, "FABLE", track="childrens_fables")

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["bible", "FABLE"])
    assert result.exit_code == 1
    output_lower = result.output.lower()
    assert "story_bible" in output_lower or "bible" in output_lower
    assert "disabled" in output_lower or "not enabled" in output_lower or "false" in output_lower


def test_bible_command_without_scene_breakdown_exits_1(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bible needs scenes; running before scene_breakdown completes is a
    hard error, not silent skip."""
    from platinum import cli as cli_mod
    from platinum.models.story import Source, StageRun, StageStatus, Story

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    story_id = "NO_SCENES"
    s = Story(
        id=story_id, track="atmospheric_horror",
        source=Source(
            type="gutenberg", url=f"https://example/{story_id}",
            title="x", author="A", raw_text="r",
            fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
        scenes=[],
        stages=[
            StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25),
                      artifacts={"decision": "approved"}),
        ],
    )
    story_dir = cli_project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["bible", story_id])
    assert result.exit_code == 1
    assert "scene_breakdown" in result.output.lower()


def test_bible_command_happy_path_persists_bible(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`platinum bible <id>` runs StoryBibleStage and writes story.bible.

    Asserts: exit 0, bible appears on disk with the expected scene count."""
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    _wire_recorder_into_config(monkeypatch, cli_mod, _bible_router_factory(n_scenes=4))

    story_path = _seed_scene_broken_story(cli_project, "STORY_OK", n_scenes=4)

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["bible", "STORY_OK"])
    assert result.exit_code == 0, result.output

    saved = json.loads(story_path.read_text(encoding="utf-8"))
    assert saved["bible"] is not None
    assert len(saved["bible"]["scenes"]) == 4
    assert "Montresor" in saved["bible"]["character_continuity"]


def test_bible_command_skips_when_bible_already_complete(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running platinum bible against a story whose bible already covers
    every scene is a no-op (StoryBibleStage.is_complete returns True). The
    recorder must NOT be invoked."""
    from platinum import cli as cli_mod
    from platinum.models.story import Story
    from platinum.models.story_bible import BibleScene, StoryBible

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    call_count = {"n": 0}

    async def recorder(req):
        call_count["n"] += 1
        # Should not be invoked
        raise AssertionError("recorder invoked despite is_complete=True")

    _wire_recorder_into_config(monkeypatch, cli_mod, recorder)

    story_path = _seed_scene_broken_story(cli_project, "STORY_HAS_BIBLE", n_scenes=4)
    story = Story.load(story_path)
    story.bible = StoryBible(
        world_genre_atmosphere="x",
        character_continuity={},
        environment_continuity={},
        scenes=[
            BibleScene(index=i, narrative_beat="b", hero_shot="m",
                        visible_characters=[], gaze_map={}, props_visible=[],
                        blocking="c", light_source="lamp", color_anchors=[],
                        brightness_floor="low")
            for i in range(1, 5)
        ],
    )
    story.save(story_path)

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["bible", "STORY_HAS_BIBLE"])
    assert result.exit_code == 0, result.output
    assert call_count["n"] == 0


def test_bible_command_rerun_clears_bible_then_regenerates(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--rerun` clears story.bible and forces regeneration even when the
    bible already covers every scene."""
    from platinum import cli as cli_mod
    from platinum.models.story import Story
    from platinum.models.story_bible import BibleScene, StoryBible

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    _wire_recorder_into_config(monkeypatch, cli_mod, _bible_router_factory(n_scenes=4))

    story_path = _seed_scene_broken_story(cli_project, "RERUN_ME", n_scenes=4)
    story = Story.load(story_path)
    # Pre-populate a bible with sentinel data so we can confirm regeneration
    # actually replaced it.
    story.bible = StoryBible(
        world_genre_atmosphere="OLD_SENTINEL",
        character_continuity={},
        environment_continuity={},
        scenes=[
            BibleScene(index=i, narrative_beat="OLD", hero_shot="m",
                        visible_characters=[], gaze_map={}, props_visible=[],
                        blocking="c", light_source="lamp", color_anchors=[],
                        brightness_floor="low")
            for i in range(1, 5)
        ],
    )
    story.save(story_path)

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["bible", "RERUN_ME", "--rerun"])
    assert result.exit_code == 0, result.output

    saved = json.loads(story_path.read_text(encoding="utf-8"))
    # The router's atmosphere replaces the sentinel.
    assert saved["bible"]["world_genre_atmosphere"] != "OLD_SENTINEL"
    assert "carnival" in saved["bible"]["world_genre_atmosphere"].lower()
