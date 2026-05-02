"""Integration tests for `platinum adapt --rerun-rejected`.

S7 §5.3.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

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


def _minimal_bible(scene_indices: list[int]):  # type: ignore[no-untyped-def]
    """Build a StoryBible covering the given scene indices with empty
    visible_characters so the S8.B.5 post-condition is a no-op (these
    rerun-rejected tests verify exact prompt equality, not character
    naming)."""
    from platinum.models.story_bible import BibleScene, StoryBible

    return StoryBible(
        world_genre_atmosphere="test",
        character_continuity={},
        environment_continuity={},
        scenes=[
            BibleScene(
                index=i, narrative_beat=f"b{i}", hero_shot="medium",
                visible_characters=[], gaze_map={},
                props_visible=[], blocking="centered",
                light_source="single candle", color_anchors=[],
                brightness_floor="low",
            )
            for i in scene_indices
        ],
    )


@pytest.fixture
def rejected_story_factory(cli_project: Path, monkeypatch):
    """Build a story with story_curator + visual_prompts COMPLETE; one scene REJECTED."""
    monkeypatch.chdir(cli_project)
    def _make() -> tuple[str, Path]:
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
            Scene(
                id="scene_001", index=1,
                narration_text="In the catacombs",
                visual_prompt="dark catacombs, candlelight",
                negative_prompt="bright daylight",
                review_status=ReviewStatus.REJECTED,
                review_feedback="too dark; need more amber lighting",
            ),
            Scene(
                id="scene_002", index=2,
                narration_text="Walking deeper",
                visual_prompt="deeper catacombs, torchlight",
                negative_prompt="bright daylight",
                review_status=ReviewStatus.APPROVED,
            ),
            Scene(
                id="scene_003", index=3,
                narration_text="Final cellar",
                visual_prompt="cellar, lanterns",
                negative_prompt="bright daylight",
                review_status=ReviewStatus.PENDING,
            ),
        ]
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
            stages=[
                StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(UTC),
                         artifacts={"decision": "approved"}),
                StageRun(stage="story_adapter", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(UTC)),
                StageRun(stage="scene_breakdown", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(UTC)),
                StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(UTC)),
            ],
            bible=_minimal_bible([1, 2, 3]),
        )
        d = cli_project / "data" / "stories" / story.id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "story.json"
        story.save(path)
        return story.id, path
    return _make


def _make_visual_prompts_response(scenes_out: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct an Anthropic API response shape with a tool_use block."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "submit_visual_prompts",
                "input": {"scenes": scenes_out},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def test_rerun_rejected_empty_set_exits_zero(cli_project: Path, monkeypatch) -> None:
    """If no scenes are REJECTED, exit 0."""
    monkeypatch.chdir(cli_project)
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
            StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                     completed_at=datetime.now(UTC),
                     artifacts={"decision": "approved"}),
            StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                     completed_at=datetime.now(UTC)),
        ],
    )
    d = cli_project / "data" / "stories" / story.id
    d.mkdir(parents=True, exist_ok=True)
    story.save(d / "story.json")

    from platinum import cli as cli_mod
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt", "--story", story.id, "--rerun-rejected"])
    assert result.exit_code == 0
    assert "no rejected" in result.output.lower()


def test_rerun_all_prompts_rewrites_every_scene_and_clears_keyframes(
    cli_project: Path, rejected_story_factory, monkeypatch,
) -> None:
    """`--rerun-all-prompts` forces a full Phase B re-author.

    Every scene's visual_prompt + composition_notes + character_refs
    must be rewritten, every keyframe_path cleared, and every
    review_status flipped to REGENERATE so the orchestrator's
    keyframe stage will re-render with new conditioning.

    S7.1 Phase B verify run gating -- before this flag, only
    REJECTED scenes picked up the new structural fields, and APPROVED
    keyframes were silently skipped by the orchestrator's per-scene
    `is_complete` check.
    """
    story_id, story_path = rejected_story_factory()
    pre = Story.load(story_path)
    pre.scenes[1].keyframe_path = Path("/workspace/old/scene_002/candidate_0.png")
    pre.scenes[2].keyframe_path = Path("/workspace/old/scene_003/candidate_0.png")
    pre.save(story_path)

    fixture_path = cli_project / "fixture.json"
    response = _make_visual_prompts_response([
        {"index": 1, "visual_prompt": "ALL: scene 1 amber-lit nobleman",
         "negative_prompt": "bright daylight",
         "composition_notes": "single oil lamp camera-right",
         "character_refs": ["Fortunato"]},
        {"index": 2, "visual_prompt": "ALL: scene 2 catacombs torchlit",
         "negative_prompt": "bright daylight",
         "composition_notes": "two figures descending stairs",
         "character_refs": ["Fortunato", "Montresor"]},
        {"index": 3, "visual_prompt": "ALL: scene 3 cellar lantern",
         "negative_prompt": "bright daylight",
         "composition_notes": "lone figure at brick wall",
         "character_refs": ["Montresor"]},
    ])
    fixture_path.write_text(
        json.dumps({"request": {}, "response": response}),
        encoding="utf-8",
    )

    from tests._fixtures import FixtureRecorder
    recorder = FixtureRecorder(path=fixture_path, mode="replay")

    from platinum import cli as cli_mod
    original_init = cli_mod.Config.__init__

    def init_with_recorder(self, root=None) -> None:
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = recorder

    monkeypatch.setattr(
        cli_mod, "Config",
        type("C", (cli_mod.Config,), {"__init__": init_with_recorder}),
    )
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app, ["adapt", "--story", story_id, "--rerun-all-prompts"],
    )
    assert result.exit_code == 0, result.output

    rt = Story.load(story_path)
    # Every scene's visual_prompt + structural fields must be rewritten.
    assert rt.scenes[0].visual_prompt == "ALL: scene 1 amber-lit nobleman"
    assert rt.scenes[1].visual_prompt == "ALL: scene 2 catacombs torchlit"
    assert rt.scenes[2].visual_prompt == "ALL: scene 3 cellar lantern"
    assert rt.scenes[0].composition_notes == "single oil lamp camera-right"
    assert rt.scenes[1].composition_notes == "two figures descending stairs"
    assert rt.scenes[2].composition_notes == "lone figure at brick wall"
    assert rt.scenes[0].character_refs == ["Fortunato"]
    assert rt.scenes[1].character_refs == ["Fortunato", "Montresor"]
    assert rt.scenes[2].character_refs == ["Montresor"]
    # Every scene's keyframe_path cleared so orchestrator re-renders.
    assert rt.scenes[0].keyframe_path is None
    assert rt.scenes[1].keyframe_path is None
    assert rt.scenes[2].keyframe_path is None
    # Every scene flipped to REGENERATE -- including the previously APPROVED one.
    assert rt.scenes[0].review_status == ReviewStatus.REGENERATE
    assert rt.scenes[1].review_status == ReviewStatus.REGENERATE
    assert rt.scenes[2].review_status == ReviewStatus.REGENERATE
    # review_feedback cleared on the previously REJECTED scene.
    assert rt.scenes[0].review_feedback is None


def test_rerun_all_prompts_and_rerun_rejected_are_mutually_exclusive(
    cli_project: Path, monkeypatch,
) -> None:
    """Passing both flags fails fast with a clear error."""
    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    from platinum import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        ["adapt", "--story", "x", "--rerun-rejected", "--rerun-all-prompts"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or \
           "cannot use both" in result.output.lower()


def test_rerun_rejected_only_applies_new_prompts_to_REJECTED(
    cli_project: Path, rejected_story_factory, monkeypatch,
) -> None:
    """Re-run rewrites only the REJECTED scene; APPROVED+PENDING untouched."""
    story_id, story_path = rejected_story_factory()

    # Stage a fixture file at the FixtureRecorder's expected path. The recorder is
    # constructed with path=fixture_path, mode="replay". When called by visual_prompts,
    # it reads the JSON and returns the "response" field.
    fixture_path = cli_project / "fixture.json"
    response = _make_visual_prompts_response([
        # Scene 1 is the REJECTED scene the test rewrites; it must include
        # a lit anchor to satisfy the S8.B.6 exposure guardrail. Scenes 2/3
        # are scene_filter-skipped so the guardrail never inspects them.
        {"index": 1, "visual_prompt": "REWRITTEN scene 1 with amber candlelight",
         "negative_prompt": "bright daylight"},
        {"index": 2, "visual_prompt": "WOULD-be-rewritten scene 2",
         "negative_prompt": "bright daylight"},
        {"index": 3, "visual_prompt": "WOULD-be-rewritten scene 3",
         "negative_prompt": "bright daylight"},
    ])
    fixture_path.write_text(
        json.dumps({"request": {}, "response": response}),
        encoding="utf-8",
    )

    from tests._fixtures import FixtureRecorder
    recorder = FixtureRecorder(path=fixture_path, mode="replay")

    from platinum import cli as cli_mod
    original_init = cli_mod.Config.__init__

    def init_with_recorder(self, root=None) -> None:
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = recorder

    monkeypatch.setattr(
        cli_mod, "Config",
        type("C", (cli_mod.Config,), {"__init__": init_with_recorder}),
    )

    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app, ["adapt", "--story", story_id, "--rerun-rejected"],
    )
    assert result.exit_code == 0, result.output

    rt = Story.load(story_path)
    # Scene 1 (was REJECTED): visual_prompt rewritten, status now REGENERATE
    assert rt.scenes[0].visual_prompt == "REWRITTEN scene 1 with amber candlelight"
    assert rt.scenes[0].review_status == ReviewStatus.REGENERATE
    assert rt.scenes[0].review_feedback is None
    assert rt.scenes[0].keyframe_path is None
    # Scene 2 (was APPROVED): visual_prompt UNCHANGED
    assert rt.scenes[1].visual_prompt == "deeper catacombs, torchlight"
    assert rt.scenes[1].review_status == ReviewStatus.APPROVED
    # Scene 3 (was PENDING): visual_prompt UNCHANGED
    assert rt.scenes[2].visual_prompt == "cellar, lanterns"
    assert rt.scenes[2].review_status == ReviewStatus.PENDING
