"""scene_breakdown pipeline: adapter narration -> Scene list with mood/sfx.

Pure validator + regen-once flow. The pure parts -- estimate, parse,
report -- are unit-tested without any Claude call. The async breakdown()
function comes in Task 16; the Stage subclass in Task 18.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage
from platinum.utils.claude import ClaudeProtocolError, ClaudeResult, Recorder
from platinum.utils.claude import call as claude_call
from platinum.utils.prompts import render_template

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


def _build_request(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    deviation_feedback: str,
) -> tuple[list[dict], list[dict]]:
    """Build system and user messages for breakdown call.

    Args:
        story: The Story object (must have adapted set).
        track_cfg: Track configuration dict (from track YAML).
        prompts_dir: Path to prompts directory.
        deviation_feedback: Feedback from previous attempt; "" on first try.

    Returns:
        Tuple of (system messages list, user messages list).
    """
    assert story.adapted is not None, "scene_breakdown requires story.adapted set"
    target_seconds = int(track_cfg["length"]["target_seconds"])
    min_s = int(track_cfg["length"]["min_seconds"])
    max_s = int(track_cfg["length"]["max_seconds"])
    # Tolerance derived from min/max so it tracks the track config exactly.
    tolerance = max(target_seconds - min_s, max_s - target_seconds)

    system = [
        {
            "type": "text",
            "text": render_template(
                prompts_dir=prompts_dir,
                track=story.track,
                name="system.j2",
                context={"track": track_cfg},
            ),
        }
    ]
    messages = [
        {
            "role": "user",
            "content": render_template(
                prompts_dir=prompts_dir,
                track=story.track,
                name="breakdown.j2",
                context={
                    "narration_script": story.adapted.narration_script,
                    "target_seconds": target_seconds,
                    "pace_wpm": int(track_cfg["voice"]["pace_wpm"]),
                    "tolerance_seconds": tolerance,
                    "deviation_feedback": deviation_feedback,
                    "music_moods": list(track_cfg["music"]["moods"]),
                },
            ),
        }
    ]
    return system, messages


async def breakdown(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    db_path: Path,
    recorder: Recorder | None = None,
) -> tuple[list[Scene], BreakdownReport, ClaudeResult]:
    """Run the breakdown call. Regen once on tolerance miss; accept second.

    Args:
        story: The Story object (must have adapted set).
        track_cfg: Track configuration dict (from track YAML).
        prompts_dir: Path to prompts directory.
        db_path: Path to SQLite database for cost tracking.
        recorder: Optional mock Recorder for testing; None = real API call.

    Returns:
        Tuple of (scenes list, BreakdownReport, ClaudeResult).
        Scenes are always returned; in_tolerance flag indicates if second attempt
        should be regenerated (Task 17). First success short-circuits the loop.
    """
    pace_wpm = int(track_cfg["voice"]["pace_wpm"])
    target = int(track_cfg["length"]["target_seconds"])
    min_s = int(track_cfg["length"]["min_seconds"])
    max_s = int(track_cfg["length"]["max_seconds"])

    deviation_feedback = ""
    last_scenes: list[Scene] = []
    last_total: float = 0.0
    last_result: ClaudeResult | None = None

    for attempt in (1, 2):
        system, messages = _build_request(
            story=story,
            track_cfg=track_cfg,
            prompts_dir=prompts_dir,
            deviation_feedback=deviation_feedback,
        )
        result = await claude_call(
            model=MODEL,
            system=system,
            messages=messages,
            tool=BREAKDOWN_TOOL,
            story_id=story.id,
            stage="scene_breakdown",
            db_path=db_path,
            recorder=recorder,
        )
        scenes = scenes_from_tool_input(result.tool_input)
        total = estimate_total_seconds(scenes, pace_wpm=pace_wpm)
        last_scenes, last_total, last_result = scenes, total, result
        if min_s <= total <= max_s:
            return (
                scenes,
                BreakdownReport(attempts=attempt, final_seconds=total, in_tolerance=True),
                result,
            )
        direction = "Lengthen" if total < min_s else "Shorten"
        deviation_feedback = (
            f"Previous breakdown totalled {total:.0f}s; target is {target}s with "
            f"min={min_s}s and max={max_s}s. {direction} scenes to land in range."
        )
        logger.info(
            "scene_breakdown attempt %d off-tolerance: %.1fs vs target %ds. Regenerating.",
            attempt,
            total,
            target,
        )

    # Second attempt also off-tolerance; accept and flag.
    assert last_result is not None  # the loop ran at least once
    return (
        last_scenes,
        BreakdownReport(attempts=2, final_seconds=last_total, in_tolerance=False),
        last_result,
    )


class SceneBreakdownStage(Stage):
    """Orchestrator wrapper for the scene breakdown Claude call."""

    name: ClassVar[str] = "scene_breakdown"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        if story.adapted is None:
            raise RuntimeError(
                f"scene_breakdown requires story_adapter to have populated story.adapted "
                f"first (story={story.id})."
            )
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        scenes, report, claude_result = await breakdown(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
        )
        story.scenes = scenes
        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
            "attempts": report.attempts,
            "final_seconds": report.final_seconds,
            "in_tolerance": report.in_tolerance,
        }
