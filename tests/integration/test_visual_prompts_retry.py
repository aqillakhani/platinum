"""Stage-level retry-on-feedback for visual_prompts (S8.B.10).

When the post-condition guardrail in `_zip_into_scenes` raises a per-scene
``VisualPromptsRewriteViolation`` (banned light token in negative_prompt,
missing visible_character, missing positive light word), the Stage catches
it once and re-prompts Opus with the violation as ``deviation_feedback``.
Single-retry only; a second consecutive violation propagates so the
operator can intervene.

Discovered S8.B verify (S8.B.9 wasn't enough): even after the j2 directive
was tightened to list all 7 banned stems, Opus emitted "flame" in scene 6
of the cask story. The rewriter docstring (visual_prompts.py:185) already
promised a "single-retry path"; this module ships it.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from platinum.config import Config
from platinum.models.db import create_all
from platinum.models.story import Adapted, Scene, Source, Story
from platinum.models.story_bible import BibleScene, StoryBible
from platinum.pipeline.context import PipelineContext


def _seeded_story_with_bible() -> Story:
    s = Story(
        id="story_test_retry",
        track="atmospheric_horror",
        source=Source(
            type="g", url="x", title="t", author="a",
            raw_text="r", fetched_at=datetime(2026, 5, 2),
            license="PD-US",
        ),
    )
    s.adapted = Adapted(
        title="t", synopsis="s", narration_script="x",
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup": "a", "rising": "b", "climax": "c", "resolution": "d"},
    )
    s.scenes = [
        Scene(id="scene_001", index=1, narration_text="text 1"),
        Scene(id="scene_002", index=2, narration_text="text 2"),
    ]
    s.bible = StoryBible(
        world_genre_atmosphere="x",
        character_continuity={"Montresor": {
            "face": "f", "costume": "c", "posture": "p",
        }},
        environment_continuity={"vault": "v"},
        scenes=[
            BibleScene(
                index=i, narrative_beat=f"b{i}", hero_shot=f"h{i}",
                visible_characters=["Montresor"],
                gaze_map={"Montresor": "off-camera"},
                props_visible=["candle"], blocking="b",
                light_source="single beeswax candle",
                color_anchors=["amber"], brightness_floor="low",
            )
            for i in (1, 2)
        ],
    )
    return s


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_bible_context(
    tmp_path: Path, repo_root: Path, *, recorder,
) -> PipelineContext:
    """Build PipelineContext with bible-enabled atmospheric_horror config."""
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copytree(repo_root / "config" / "prompts", tmp_path / "config" / "prompts")
    src = repo_root / "config" / "tracks" / "atmospheric_horror.yaml"
    cfg_yaml = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg_yaml["track"].setdefault("story_bible", {})["enabled"] = True
    track_path = tmp_path / "config" / "tracks" / "atmospheric_horror.yaml"
    track_path.write_text(yaml.safe_dump(cfg_yaml), encoding="utf-8")
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8",
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data").mkdir()
    cfg = Config(root=tmp_path)
    create_all(cfg.data_dir / "platinum.db")
    ctx = PipelineContext(config=cfg, logger=logging.getLogger("test"))
    ctx.config.settings.setdefault("test", {})["claude_recorder"] = recorder
    return ctx


def _vp_response(prompts: list[tuple[str, str]]) -> dict:
    return {
        "id": "x",
        "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
            "scenes": [
                {"index": i + 1, "visual_prompt": vp, "negative_prompt": np_}
                for i, (vp, np_) in enumerate(prompts)
            ],
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


# ---------- Exception type carries enough state for retry --------------------


def test_violation_exception_carries_metadata() -> None:
    """VisualPromptsRewriteViolation exposes scene_index, emitted_prompt,
    and feedback so the Stage retry layer can build a deviation_feedback
    entry without parsing the exception message."""
    from platinum.pipeline.visual_prompts import VisualPromptsRewriteViolation
    from platinum.utils.claude import ClaudeProtocolError

    exc = VisualPromptsRewriteViolation(
        "scene 5: neg banned ['lamp']",
        scene_index=5,
        emitted_prompt="Montresor in vault, beeswax candle",
        feedback="Remove 'lamp' from negative_prompt.",
    )
    assert exc.scene_index == 5
    assert exc.emitted_prompt == "Montresor in vault, beeswax candle"
    assert "Remove 'lamp'" in exc.feedback
    # Subclass relationship preserves backward compat for existing
    # ``pytest.raises(ClaudeProtocolError)`` assertions.
    assert isinstance(exc, ClaudeProtocolError)


# ---------- Stage retries once on violation ----------------------------------


@pytest.mark.asyncio
async def test_stage_retries_once_on_rewrite_violation(
    tmp_path: Path, repo_root: Path,
) -> None:
    """First call emits 'flame' in scene 2 neg → guardrail raises →
    Stage catches, builds deviation_feedback for scene 2, re-calls →
    second response is clean → Stage returns success."""
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    calls: list[dict] = []

    async def synth(req: dict) -> dict:
        calls.append(req)
        if len(calls) == 1:
            # Scene 2 negative_prompt bans "flame" (banned stem)
            return _vp_response([
                ("Montresor in candlelight, beeswax candle on desk",
                 "neon, plastic"),
                ("Montresor in vault, candle on stone",
                 "bright daylight, flame"),
            ])
        # Second call: scene 2 negative is now clean
        return _vp_response([
            ("Montresor in candlelight, beeswax candle on desk",
             "neon, plastic"),
            ("Montresor in vault, candle on stone",
             "bright daylight, anime"),
        ])

    ctx = _make_bible_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story_with_bible()
    stage = VisualPromptsStage()

    artifacts = await stage.run(story, ctx)

    assert len(calls) == 2, "Stage must retry exactly once after violation"
    s1_neg = story.scenes[1].negative_prompt
    s0_vp = story.scenes[0].visual_prompt
    assert s1_neg is not None and s0_vp is not None
    assert "flame" not in s1_neg
    assert "candle" in s0_vp.lower()
    second_user_msg = calls[1]["messages"][0]["content"]
    assert "DEVIATION FEEDBACK" in second_user_msg
    assert "Scene 2" in second_user_msg
    assert "flame" in second_user_msg.lower()
    assert artifacts["model"] == "claude-opus-4-7"


# ---------- No retry on protocol errors --------------------------------------


@pytest.mark.asyncio
async def test_stage_does_not_retry_on_count_mismatch(
    tmp_path: Path, repo_root: Path,
) -> None:
    """Count mismatch is a protocol error, not a retryable per-scene
    violation. The Stage propagates without re-calling Opus."""
    from platinum.pipeline.visual_prompts import VisualPromptsStage
    from platinum.utils.claude import ClaudeProtocolError

    calls: list[dict] = []

    async def synth(req: dict) -> dict:
        calls.append(req)
        # Story has 2 scenes; we return only 1 → ClaudeProtocolError
        return _vp_response([
            ("Montresor, beeswax candle", "neon"),
        ])

    ctx = _make_bible_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story_with_bible()
    stage = VisualPromptsStage()

    with pytest.raises(ClaudeProtocolError, match="count"):
        await stage.run(story, ctx)
    assert len(calls) == 1, "count mismatch must NOT trigger retry"


# ---------- Single-retry policy: second violation propagates -----------------


@pytest.mark.asyncio
async def test_stage_propagates_when_second_call_also_violates(
    tmp_path: Path, repo_root: Path,
) -> None:
    """Single-retry policy: if the retry also violates, the Stage propagates
    the second exception. Bounds blast radius at 2 Opus calls per Stage run."""
    from platinum.pipeline.visual_prompts import (
        VisualPromptsRewriteViolation,
        VisualPromptsStage,
    )

    calls: list[dict] = []

    async def synth(req: dict) -> dict:
        calls.append(req)
        # Always emit "torch" in scene 2 neg → always violates
        return _vp_response([
            ("Montresor in candlelight, beeswax candle on desk",
             "neon, plastic"),
            ("Montresor in vault, candle on stone",
             "bright daylight, torch"),
        ])

    ctx = _make_bible_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story_with_bible()
    stage = VisualPromptsStage()

    with pytest.raises(VisualPromptsRewriteViolation):
        await stage.run(story, ctx)
    assert len(calls) == 2, "exactly one retry, then propagate"


# ---------- Different violation types all trigger retry ----------------------


@pytest.mark.asyncio
async def test_stage_retries_on_missing_character_violation(
    tmp_path: Path, repo_root: Path,
) -> None:
    """The retry path covers H1 (missing visible_character), not just H2
    (banned in neg). First call drops 'Montresor' from scene 2 prompt →
    retry → second call includes the name → success."""
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    calls: list[dict] = []

    async def synth(req: dict) -> dict:
        calls.append(req)
        if len(calls) == 1:
            return _vp_response([
                ("Montresor in candlelight, beeswax candle on desk", "neon"),
                ("a man in vault, candle on stone", "neon"),  # no Montresor
            ])
        return _vp_response([
            ("Montresor in candlelight, beeswax candle on desk", "neon"),
            ("Montresor in vault, candle on stone", "neon"),
        ])

    ctx = _make_bible_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story_with_bible()
    stage = VisualPromptsStage()

    await stage.run(story, ctx)

    assert len(calls) == 2
    s1_vp = story.scenes[1].visual_prompt
    assert s1_vp is not None
    assert "montresor" in s1_vp.lower()
    second_user_msg = calls[1]["messages"][0]["content"]
    assert "DEVIATION FEEDBACK" in second_user_msg
    assert "Montresor" in second_user_msg
