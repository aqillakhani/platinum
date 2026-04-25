"""Unit tests for pipeline/scene_breakdown.py."""

from __future__ import annotations

import pytest

from platinum.models.story import Scene


def test_estimate_total_seconds_pure_helper() -> None:
    from platinum.pipeline.scene_breakdown import estimate_total_seconds

    scenes = [
        Scene(id="scene_001", index=1, narration_text=" ".join(["w"] * 130)),
        Scene(id="scene_002", index=2, narration_text=" ".join(["w"] * 130)),
    ]
    # 260 words / 130 wpm = 2 minutes = 120s
    assert estimate_total_seconds(scenes, pace_wpm=130) == 120.0


def test_breakdown_report_in_tolerance_range() -> None:
    from platinum.pipeline.scene_breakdown import BreakdownReport

    r = BreakdownReport(attempts=1, final_seconds=605.0, in_tolerance=True)
    assert r.attempts == 1
    assert r.in_tolerance is True


def test_scenes_from_tool_input_assigns_ids() -> None:
    from platinum.pipeline.scene_breakdown import scenes_from_tool_input

    tool_input = {
        "scenes": [
            {
                "index": 1,
                "narration_text": "It begins.",
                "mood": "ambient_drone",
                "sfx_cues": ["clock_ticking_distant"],
            },
            {
                "index": 2,
                "narration_text": "It builds.",
                "mood": "slow_strings_dread",
                "sfx_cues": [],
            },
            {
                "index": 3,
                "narration_text": "It grows.",
                "mood": "increasing_tension",
                "sfx_cues": ["heartbeat"],
            },
            {
                "index": 4,
                "narration_text": "It ends.",
                "mood": "resolution",
                "sfx_cues": [],
            },
        ]
    }
    scenes = scenes_from_tool_input(tool_input)
    assert [s.id for s in scenes] == [
        "scene_001",
        "scene_002",
        "scene_003",
        "scene_004",
    ]
    assert [s.index for s in scenes] == [1, 2, 3, 4]
    assert scenes[0].music_cue == "ambient_drone"
    assert scenes[0].sfx_cues == ["clock_ticking_distant"]


def test_scenes_from_tool_input_rejects_too_few() -> None:
    from platinum.pipeline.scene_breakdown import scenes_from_tool_input
    from platinum.utils.claude import ClaudeProtocolError

    with pytest.raises(ClaudeProtocolError, match="minItems"):
        scenes_from_tool_input(
            {
                "scenes": [
                    {
                        "index": 1,
                        "narration_text": "x",
                        "mood": "ambient_drone",
                        "sfx_cues": [],
                    },
                ]
            }
        )


@pytest.mark.asyncio
async def test_breakdown_first_pass_in_tolerance(tmp_path) -> None:
    from datetime import datetime
    from pathlib import Path

    import yaml

    from platinum.models.db import create_all
    from platinum.models.story import Adapted, Source, Story
    from platinum.pipeline.scene_breakdown import breakdown

    # Build fixture story
    story = Story(
        id="story_brk",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="x",
            title="t",
            author="a",
            raw_text="raw",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
    )
    # 8 scenes of 162 words each = 1296 words / 130 wpm * 60 = 597.69s
    # atmospheric_horror has min=480, max=720, so this lands cleanly in tolerance.
    story.adapted = Adapted(
        title="t",
        synopsis="s",
        narration_script=" ".join(["word"] * 1300),
        estimated_duration_seconds=600.0,
        tone_notes="n",
        arc={"setup": "a", "rising": "b", "climax": "c", "resolution": "d"},
    )

    def _scene_dict(idx: int, words: int) -> dict:
        return {
            "index": idx,
            "narration_text": " ".join(["w"] * words),
            "mood": "ambient_drone",
            "sfx_cues": ["wind_through_window"] if idx % 2 == 0 else [],
        }

    async def synth(req):
        return {
            "id": "ok",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_scene_breakdown",
                    "input": {
                        "scenes": [_scene_dict(i, 162) for i in range(1, 9)],
                    },
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    db_path = tmp_path / "p.db"
    create_all(db_path)
    scenes, report, _result = await breakdown(
        story=story,
        track_cfg=yaml.safe_load(
            (
                Path(__file__).resolve().parents[2]
                / "config"
                / "tracks"
                / "atmospheric_horror.yaml"
            ).read_text(encoding="utf-8")
        )["track"],
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path,
        recorder=synth,
    )
    assert len(scenes) == 8
    assert report.attempts == 1
    assert report.in_tolerance is True
    # Bounded by atmospheric_horror's min=480 and max=720 (the YAML values)
    assert 480 <= report.final_seconds <= 720
