"""Integration tests for MotionPromptsStage (S8.A.5).

End-to-end Stage.run() with a real Config (tmp_project layout copied from
the repo's config/) and a mock claude_recorder injected via settings.yaml.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from platinum.config import Config
from platinum.models.story import Adapted, Scene, Source, Story

REPO_ROOT = Path(__file__).resolve().parents[2]


def _setup_tmp_project(tmp_project: Path) -> Config:
    """Copy real config/{tracks,prompts,workflows} into a tmp project root."""
    cfg_dir = tmp_project / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        REPO_ROOT / "config" / "tracks", cfg_dir / "tracks",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        REPO_ROOT / "config" / "prompts", cfg_dir / "prompts",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        REPO_ROOT / "config" / "workflows", cfg_dir / "workflows",
        dirs_exist_ok=True,
    )
    (cfg_dir / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_project / "data").mkdir(parents=True, exist_ok=True)
    (tmp_project / "secrets").mkdir(parents=True, exist_ok=True)
    return Config(root=tmp_project)


def _story_with_keyframes(tmp_project: Path, n: int = 2) -> Story:
    """Story with n scenes, each with a real PNG keyframe on disk."""
    from tests._fixtures import make_synthetic_png

    story_dir = tmp_project / "data" / "stories" / "story_mp_int"
    (story_dir / "keyframes").mkdir(parents=True, exist_ok=True)

    story = Story(
        id="story_mp_int",
        track="atmospheric_horror",
        source=Source(
            type="g", url="x", title="t", author="a", raw_text="r",
            fetched_at=datetime(2026, 5, 1), license="PD-US",
        ),
    )
    story.adapted = Adapted(
        title="t", synopsis="s", narration_script="x",
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup": "", "rising": "", "climax": "", "resolution": ""},
    )
    story.scenes = []
    for i in range(1, n + 1):
        kf_dir = story_dir / "keyframes" / f"scene_{i:03d}"
        kf_dir.mkdir(parents=True, exist_ok=True)
        kf = kf_dir / "candidate_0.png"
        make_synthetic_png(kf, kind="grey", value=64 + i * 16)
        story.scenes.append(
            Scene(
                id=f"scene_{i:03d}",
                index=i,
                narration_text=f"narration line {i}",
                visual_prompt=(
                    f"medium shot, subject {i} performing action {i}, "
                    "candlelit corridor, painterly cinematography, dim mood"
                ),
                keyframe_path=kf,
            )
        )
    story.save(story_dir / "story.json")
    return story


@pytest.mark.asyncio
async def test_motion_prompts_stage_runs_with_injected_recorder(
    tmp_path: Path,
) -> None:
    """Stage reads recorder from settings, invokes Claude per scene, mutates
    motion_prompt on each scene, persists story.json, returns artifacts."""
    from platinum.models.db import create_all
    from platinum.pipeline.context import PipelineContext
    from platinum.pipeline.motion_prompts import MotionPromptsStage

    cfg = _setup_tmp_project(tmp_path)
    create_all(cfg.data_dir / "platinum.db")

    story = _story_with_keyframes(tmp_path, n=3)

    async def synth(_: dict) -> dict:
        return {
            "id": "msg_synth",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_motion_prompt",
                    "input": {
                        "motion_prompt": "slow camera drift; flames flicker",
                        "rationale": "subject already in pose",
                    },
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 50, "output_tokens": 25,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            },
        }

    # Recorder injected via settings.test.claude_recorder (mirrors how
    # other stages pick up test recorders -- see VisualPromptsStage.run).
    cfg.settings.setdefault("test", {})["claude_recorder"] = synth

    ctx = PipelineContext(config=cfg, logger=logging.getLogger("test_mp_stage"))
    stage = MotionPromptsStage()
    artifacts = await stage.run(story, ctx)

    for scene in story.scenes:
        assert scene.motion_prompt == "slow camera drift; flames flicker"
    assert artifacts["scenes_processed"] == 3
    assert artifacts.get("cost_usd", 0) >= 0


@pytest.mark.asyncio
async def test_motion_prompts_stage_skips_when_no_keyframes(
    tmp_path: Path,
) -> None:
    """A story whose scenes have no keyframe_path is a no-op (no Claude calls)."""
    from platinum.models.db import create_all
    from platinum.pipeline.context import PipelineContext
    from platinum.pipeline.motion_prompts import MotionPromptsStage

    cfg = _setup_tmp_project(tmp_path)
    create_all(cfg.data_dir / "platinum.db")

    story = _story_with_keyframes(tmp_path, n=2)
    for s in story.scenes:
        s.keyframe_path = None
    story.save(cfg.data_dir / "stories" / story.id / "story.json")

    call_count = 0

    async def synth(_: dict) -> dict:
        nonlocal call_count
        call_count += 1
        raise AssertionError("Stage should not call Claude when no keyframes")

    cfg.settings.setdefault("test", {})["claude_recorder"] = synth
    ctx = PipelineContext(config=cfg, logger=logging.getLogger("test_mp_skip"))
    stage = MotionPromptsStage()
    artifacts = await stage.run(story, ctx)

    assert call_count == 0
    assert artifacts["scenes_processed"] == 0
