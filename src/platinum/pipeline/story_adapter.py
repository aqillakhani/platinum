"""story_adapter pipeline -- one Claude call: source text -> polished narration.

Pure function: takes (story, track_cfg, prompts_dir, db_path, recorder)
and returns (Adapted, ClaudeResult). No I/O outside the injected
claude.call (which itself goes through the recorder seam in tests).

The Stage subclass that wires this into the orchestrator lives in Task 13.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from platinum.models.story import Adapted, Story
from platinum.utils.claude import (
    ClaudeProtocolError,
    ClaudeResult,
    Recorder,
)
from platinum.utils.claude import (
    call as claude_call,
)
from platinum.utils.prompts import (
    render_template,
)

logger = logging.getLogger(__name__)


MODEL = "claude-opus-4-7"
MAX_SOURCE_CHARS = 80_000


ADAPT_TOOL: dict[str, Any] = {
    "name": "submit_adapted_story",
    "description": "Submit the polished cinematic adaptation.",
    "input_schema": {
        "type": "object",
        "required": ["title", "synopsis", "narration_script", "tone_notes", "arc"],
        "properties": {
            "title": {"type": "string"},
            "synopsis": {"type": "string", "maxLength": 400},
            "narration_script": {"type": "string"},
            "tone_notes": {"type": "string"},
            "arc": {
                "type": "object",
                "required": ["setup", "rising", "climax", "resolution"],
                "properties": {
                    "setup": {"type": "string"},
                    "rising": {"type": "string"},
                    "climax": {"type": "string"},
                    "resolution": {"type": "string"},
                },
            },
        },
    },
}


def _truncate_source(text: str, *, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 8]
    return head + "\n[...]\n"


def _build_request(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
) -> tuple[list[dict], list[dict]]:
    system = [
        {
            "type": "text",
            "text": render_template(
                prompts_dir=prompts_dir, track=story.track,
                name="system.j2", context={"track": track_cfg},
            ),
        }
    ]
    user = [
        {
            "role": "user",
            "content": render_template(
                prompts_dir=prompts_dir, track=story.track, name="adapt.j2",
                context={
                    "title": story.source.title,
                    "author": story.source.author or "",
                    "raw_text": _truncate_source(story.source.raw_text),
                    "target_seconds": int(track_cfg["length"]["target_seconds"]),
                    "pace_wpm": int(track_cfg["voice"]["pace_wpm"]),
                },
            ),
        }
    ]
    return system, user


def _adapted_from_tool_input(tool_input: dict, *, pace_wpm: int) -> Adapted:
    arc = tool_input.get("arc", {})
    required_arc = {"setup", "rising", "climax", "resolution"}
    if not isinstance(arc, dict) or not required_arc.issubset(arc.keys()):
        arc_repr = sorted(arc.keys()) if isinstance(arc, dict) else arc
        raise ClaudeProtocolError(
            f"adapter response missing arc keys: got {arc_repr!r}"
        )
    script = tool_input["narration_script"]
    word_count = len(script.split())
    return Adapted(
        title=tool_input["title"],
        synopsis=tool_input["synopsis"],
        narration_script=script,
        estimated_duration_seconds=round(word_count / pace_wpm * 60, 2),
        tone_notes=tool_input["tone_notes"],
        arc={k: arc[k] for k in ("setup", "rising", "climax", "resolution")},
    )


async def adapt(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    db_path: Path,
    recorder: Recorder | None = None,
) -> tuple[Adapted, ClaudeResult]:
    """Run the adapter Claude call. Mutates nothing; returns (Adapted, ClaudeResult)."""
    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
    )
    pace_wpm = int(track_cfg["voice"]["pace_wpm"])

    if len(story.source.raw_text) > MAX_SOURCE_CHARS:
        logger.warning(
            "Source raw_text length %d exceeds %d; truncating in adapter prompt.",
            len(story.source.raw_text), MAX_SOURCE_CHARS,
        )

    result: ClaudeResult = await claude_call(
        model=MODEL,
        system=system, messages=messages, tool=ADAPT_TOOL,
        story_id=story.id, stage="story_adapter",
        db_path=db_path, recorder=recorder,
    )
    return _adapted_from_tool_input(result.tool_input, pace_wpm=pace_wpm), result
