"""Integration tests: each Session-4 Stage subclass runs end-to-end with a
synthetic recorder injected via PipelineContext."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from platinum.config import Config
from platinum.models.db import create_all
from platinum.models.story import Source, Story
from platinum.pipeline.context import PipelineContext


def _seeded_story() -> Story:
    return Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="https://example/poe", title="The Cask",
            author="Edgar Allan Poe",
            raw_text="The thousand injuries of Fortunato I had borne...",
            fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
    )


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_context(tmp_path: Path, repo_root: Path, *, recorder) -> PipelineContext:
    """Build a PipelineContext that mirrors the real layout for stages.

    The Stage code reads track YAML and prompts from the project root, so
    we copy them under tmp_path's config/. The track YAML's story_bible
    block is force-disabled so these pre-S8.B tests exercise the
    narration-only rewriter path; bible-aware coverage lives in
    tests/unit/test_visual_prompts_with_bible.py.
    """
    import yaml as _yaml

    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copytree(repo_root / "config" / "prompts", tmp_path / "config" / "prompts")
    track_path = tmp_path / "config" / "tracks" / "atmospheric_horror.yaml"
    src = repo_root / "config" / "tracks" / "atmospheric_horror.yaml"
    cfg_yaml = _yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg_yaml["track"].setdefault("story_bible", {})["enabled"] = False
    track_path.write_text(_yaml.safe_dump(cfg_yaml), encoding="utf-8")
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data").mkdir()

    cfg = Config(root=tmp_path)
    create_all(cfg.data_dir / "platinum.db")
    ctx = PipelineContext(config=cfg, logger=logging.getLogger("test"))
    # Stash the recorder where the Stage can find it.
    ctx.config.settings.setdefault("test", {})["claude_recorder"] = recorder
    return ctx


@pytest.mark.asyncio
async def test_story_adapter_stage_run_populates_adapted(tmp_path, repo_root) -> None:
    from platinum.pipeline.story_adapter import StoryAdapterStage

    async def synth(req):
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "submit_adapted_story", "input": {
                "title": "The Cask",
                "synopsis": "...",
                "narration_script": "word " * 1300,
                "tone_notes": "...",
                "arc": {"setup":"a","rising":"b","climax":"c","resolution":"d"},
            }}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 50,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    stage = StoryAdapterStage()

    artifacts = await stage.run(story, ctx)

    assert story.adapted is not None
    assert story.adapted.title == "The Cask"
    assert story.adapted.arc["climax"] == "c"
    assert artifacts["model"] == "claude-opus-4-7"
    assert artifacts["cost_usd"] > 0


@pytest.mark.asyncio
async def test_scene_breakdown_stage_run_populates_scenes(tmp_path, repo_root) -> None:
    from platinum.models.story import Adapted
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage

    async def synth(req):
        scenes = [{
            "index": i, "narration_text": " ".join(["w"] * 162),
            "mood": "ambient_drone", "sfx_cues": [],
        } for i in range(1, 9)]
        return {
            "id": "x", "content": [{"type": "tool_use",
                                     "name": "submit_scene_breakdown",
                                     "input": {"scenes": scenes}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    story.adapted = Adapted(
        title="t", synopsis="s", narration_script="word " * 1300,
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup":"a","rising":"b","climax":"c","resolution":"d"},
    )
    stage = SceneBreakdownStage()
    artifacts = await stage.run(story, ctx)

    assert len(story.scenes) == 8
    assert artifacts["attempts"] == 1
    assert artifacts["in_tolerance"] is True
    assert artifacts["final_seconds"] > 0


@pytest.mark.asyncio
async def test_visual_prompts_stage_run_populates_prompts(tmp_path, repo_root) -> None:
    from platinum.models.story import Adapted, Scene
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    async def synth(req):
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
                "scenes": [
                    {"index": 1, "visual_prompt": "vault, candle", "negative_prompt": "bright"},
                    {"index": 2, "visual_prompt": "fog, stones", "negative_prompt": "neon"},
                ],
            }}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    story.adapted = Adapted(
        title="t", synopsis="s", narration_script="word " * 1300,
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup":"a","rising":"b","climax":"c","resolution":"d"},
    )
    story.scenes = [
        Scene(id="scene_001", index=1, narration_text="It was night."),
        Scene(id="scene_002", index=2, narration_text="The vault opened."),
    ]
    stage = VisualPromptsStage()
    artifacts = await stage.run(story, ctx)

    assert story.scenes[0].visual_prompt == "vault, candle"
    assert story.scenes[1].negative_prompt == "neon"
    assert artifacts["model"] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_three_stages_in_sequence_produces_complete_story(tmp_path, repo_root) -> None:
    """Run StoryAdapterStage -> SceneBreakdownStage -> VisualPromptsStage
    via the orchestrator and verify the final Story has all three populated."""
    from platinum.models.story import StageStatus
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage
    from platinum.pipeline.story_adapter import StoryAdapterStage
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    async def router(req):
        # Distinguish by tool_choice name to dispatch the right canned response.
        tool_name = req["tool_choice"]["name"]
        if tool_name == "submit_adapted_story":
            return {
                "id": "ad", "content": [{"type": "tool_use", "name": tool_name, "input": {
                    "title": "T", "synopsis": "S", "narration_script": "word " * 1300,
                    "tone_notes": "N",
                    "arc": {"setup":"a","rising":"b","climax":"c","resolution":"d"},
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if tool_name == "submit_scene_breakdown":
            return {
                "id": "br", "content": [{"type": "tool_use", "name": tool_name, "input": {
                    "scenes": [
                        {"index": i, "narration_text": " ".join(["w"] * 162),
                         "mood": "ambient_drone", "sfx_cues": []}
                        for i in range(1, 9)
                    ],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if tool_name == "submit_visual_prompts":
            return {
                "id": "vp", "content": [{"type": "tool_use", "name": tool_name, "input": {
                    "scenes": [{"index": i, "visual_prompt": f"vp{i}",
                                 "negative_prompt": f"np{i}"} for i in range(1, 9)],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        raise AssertionError(f"unexpected tool_choice: {tool_name}")

    ctx = _make_context(tmp_path, repo_root, recorder=router)
    story = _seeded_story()

    orch = Orchestrator(stages=[
        StoryAdapterStage(), SceneBreakdownStage(), VisualPromptsStage(),
    ])
    final = await orch.run(story, ctx)

    assert final.adapted is not None
    assert len(final.scenes) == 8
    assert all(s.visual_prompt for s in final.scenes)
    completed = [r for r in final.stages if r.status == StageStatus.COMPLETE]
    assert {r.stage for r in completed} >= {"story_adapter", "scene_breakdown", "visual_prompts"}


@pytest.mark.asyncio
async def test_resume_skips_completed_stage(tmp_path, repo_root) -> None:
    """A pre-existing COMPLETE story_adapter run should make that stage skip."""
    from platinum.models.story import StageRun, StageStatus
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.story_adapter import StoryAdapterStage

    calls = {"n": 0}

    async def synth(req):
        calls["n"] += 1
        return {
            "id": "x", "content": [{"type": "tool_use", "name": "submit_adapted_story",
                                     "input": {"title": "T", "synopsis": "S",
                                                "narration_script": "x", "tone_notes": "n",
                                                "arc": {"setup":"a","rising":"b",
                                                         "climax":"c","resolution":"d"}}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    # Mark adapter already complete.
    story.stages.append(StageRun(
        stage="story_adapter", status=StageStatus.COMPLETE,
        started_at=datetime(2026, 4, 25), completed_at=datetime(2026, 4, 25),
    ))

    orch = Orchestrator(stages=[StoryAdapterStage()])
    await orch.run(story, ctx)
    assert calls["n"] == 0  # adapter was skipped
