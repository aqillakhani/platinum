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
