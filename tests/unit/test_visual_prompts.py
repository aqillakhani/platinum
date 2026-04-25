"""Unit tests for pipeline/visual_prompts.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from platinum.models.story import Adapted, Scene, Source, Story


def _track() -> dict:
    track_path = (
        Path(__file__).resolve().parents[2]
        / "config" / "tracks" / "atmospheric_horror.yaml"
    )
    return yaml.safe_load(track_path.read_text(encoding="utf-8"))["track"]


def _story_with_scenes(n: int = 4) -> Story:
    s = Story(
        id="story_vp", track="atmospheric_horror",
        source=Source(type="g", url="x", title="t", author="a",
                      raw_text="r", fetched_at=datetime(2026, 4, 25), license="PD-US"),
    )
    s.adapted = Adapted(title="t", synopsis="s", narration_script="x",
                         estimated_duration_seconds=600.0, tone_notes="n",
                         arc={"setup":"","rising":"","climax":"","resolution":""})
    s.scenes = [Scene(id=f"scene_{i:03d}", index=i, narration_text=f"text {i}")
                for i in range(1, n + 1)]
    return s


def _synth_response(n: int) -> dict:
    return {
        "id": "x",
        "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
            "scenes": [
                {"index": i, "visual_prompt": f"vp{i}", "negative_prompt": f"np{i}"}
                for i in range(1, n + 1)
            ],
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


@pytest.mark.asyncio
async def test_visual_prompts_zips_into_scenes_by_index(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(4)

    async def synth(req):
        # Return out-of-order to exercise zip-by-index
        r = _synth_response(4)
        r["content"][0]["input"]["scenes"].reverse()
        return r

    scenes, _ = await visual_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=synth,
    )
    assert [(s.index, s.visual_prompt, s.negative_prompt) for s in scenes] == [
        (1, "vp1", "np1"), (2, "vp2", "np2"), (3, "vp3", "np3"), (4, "vp4", "np4"),
    ]


@pytest.mark.asyncio
async def test_visual_prompts_count_must_match_scene_count(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(4)

    async def synth(req):
        return _synth_response(3)  # only 3 prompts for 4 scenes

    with pytest.raises(ClaudeProtocolError, match="count"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
            db_path=db_path, recorder=synth,
        )
