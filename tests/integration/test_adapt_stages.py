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
    we copy them under tmp_path's config/.
    """
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
