"""scene_breakdown pipeline: adapter narration -> Scene list with mood/sfx.

Pure validator + regen-once flow. The pure parts -- estimate, parse,
report -- are unit-tested without any Claude call. The async breakdown()
function comes in Task 16; the Stage subclass in Task 18.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from platinum.models.story import Scene
from platinum.utils.claude import ClaudeProtocolError

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
MIN_SCENES = 4
MAX_SCENES = 20


BREAKDOWN_TOOL: dict[str, Any] = {
    "name": "submit_scene_breakdown",
    "description": "Submit the scene-by-scene breakdown of the narration.",
    "input_schema": {
        "type": "object",
        "required": ["scenes"],
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": MIN_SCENES,
                "maxItems": MAX_SCENES,
                "items": {
                    "type": "object",
                    "required": ["index", "narration_text", "mood", "sfx_cues"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 1},
                        "narration_text": {"type": "string"},
                        "mood": {"type": "string"},
                        "sfx_cues": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class BreakdownReport:
    """Result summary returned alongside the scene list."""

    attempts: int
    final_seconds: float
    in_tolerance: bool


def estimate_total_seconds(scenes: list[Scene], *, pace_wpm: int) -> float:
    """Estimate total duration in seconds from scene narration text.

    Args:
        scenes: List of Scene objects.
        pace_wpm: Words per minute for narration pacing.

    Returns:
        Total duration in seconds, rounded to 2 decimal places.
    """
    total_words = sum(len(s.narration_text.split()) for s in scenes)
    return round(total_words / pace_wpm * 60, 2)


def scenes_from_tool_input(tool_input: dict) -> list[Scene]:
    """Parse Claude tool response into Scene objects with assigned IDs.

    Args:
        tool_input: Dict with "scenes" key containing list of scene dicts.

    Returns:
        List of Scene objects with assigned IDs (scene_001, scene_002, etc).

    Raises:
        ClaudeProtocolError: If scenes list is missing or too small.
    """
    raw = tool_input.get("scenes", [])
    if not isinstance(raw, list) or len(raw) < MIN_SCENES:
        count = len(raw) if isinstance(raw, list) else type(raw)
        raise ClaudeProtocolError(
            f"breakdown response failed minItems={MIN_SCENES}: got {count}"
        )
    out: list[Scene] = []
    for i, item in enumerate(raw, start=1):
        out.append(
            Scene(
                id=f"scene_{i:03d}",
                index=item.get("index", i),
                narration_text=item["narration_text"],
                music_cue=item.get("mood"),
                sfx_cues=list(item.get("sfx_cues", [])),
            )
        )
    return out
