"""story_bible pipeline: produce a whole-story narrative directive (S8.B).

The visual_prompts stage previously ran per-scene with no story-wide
context. Result: character drift, missing props, generic "moody horror"
instead of the specific beat. The bible stage fixes this — Opus 4.7
reads the entire story once and emits a structured directive
(world_genre_atmosphere, character_continuity, environment_continuity,
per-scene hero_shot/visible_characters/gaze_map/props_visible/blocking/
light_source/color_anchors/brightness_floor). The visual_prompts stage
(rewritten in S8.B.5) leads with this directive when grounded.

Architecture mirrors visual_prompts.py: tool-use schema +
``_build_request`` + ``_zip_into_story`` + top-level ``story_bible()``
coroutine + ``StoryBibleStage`` orchestrator-friendly Stage subclass.
The single LLM call returns the complete bible JSON via the
``submit_story_bible`` tool.

Design ref: docs/plans/2026-05-02-session-8.B-story-bible-design.md.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Story
from platinum.models.story_bible import StoryBible
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage
from platinum.utils.claude import (
    ClaudeProtocolError,
    ClaudeResult,
    Recorder,
)
from platinum.utils.claude import (
    call as claude_call,
)
from platinum.utils.prompts import render_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema -- one call per story, one tool, one structured response.
# ---------------------------------------------------------------------------


_BIBLE_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "index", "narrative_beat", "hero_shot", "visible_characters",
        "gaze_map", "props_visible", "blocking", "light_source",
        "color_anchors", "brightness_floor",
    ],
    "properties": {
        "index": {"type": "integer", "minimum": 1},
        "narrative_beat": {"type": "string"},
        "hero_shot": {"type": "string"},
        "visible_characters": {"type": "array", "items": {"type": "string"}},
        "gaze_map": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "props_visible": {"type": "array", "items": {"type": "string"}},
        "blocking": {"type": "string"},
        "light_source": {"type": "string"},
        "color_anchors": {"type": "array", "items": {"type": "string"}},
        "brightness_floor": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
}


STORY_BIBLE_TOOL: dict[str, Any] = {
    "name": "submit_story_bible",
    "description": (
        "Submit the whole-story narrative directive consumed by "
        "visual_prompts. One call per story; every field required."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "world_genre_atmosphere",
            "character_continuity",
            "environment_continuity",
            "scenes",
        ],
        "properties": {
            "world_genre_atmosphere": {"type": "string"},
            "character_continuity": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["face", "costume", "posture"],
                    "properties": {
                        "face": {"type": "string"},
                        "costume": {"type": "string"},
                        "posture": {"type": "string"},
                    },
                },
            },
            "environment_continuity": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "scenes": {
                "type": "array",
                "items": _BIBLE_SCENE_SCHEMA,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


def _build_request(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Render the bible system + user prompts.

    Mirrors ``visual_prompts._build_request`` shape: returns
    ``(system, messages)`` ready to feed ``claude.call``.
    """
    system = [
        {"type": "text", "text": render_template(
            prompts_dir=prompts_dir, track=story.track,
            name="system.j2", context={"track": track_cfg},
        )},
        {"type": "text", "text": render_template(
            prompts_dir=prompts_dir, track=story.track,
            name="system_bible.j2", context={"track": track_cfg},
        )},
    ]
    adapted_synopsis = story.adapted.synopsis if story.adapted else ""
    adapted_narration = story.adapted.narration_script if story.adapted else ""
    messages = [
        {"role": "user", "content": render_template(
            prompts_dir=prompts_dir, track=story.track, name="story_bible.j2",
            context={
                "aesthetic": track_cfg["visual"]["aesthetic"],
                "palette": track_cfg["visual"]["palette"],
                "adapted_synopsis": adapted_synopsis,
                "adapted_narration_script": adapted_narration,
                "scenes": [
                    {"index": s.index, "narration_text": s.narration_text}
                    for s in story.scenes
                ],
            },
        )}
    ]
    return system, messages


# ---------------------------------------------------------------------------
# Response handler -- validate + assign to story.bible
# ---------------------------------------------------------------------------


def _zip_into_story(story: Story, tool_input: dict) -> StoryBible:
    """Validate the bible tool_input and assign to ``story.bible``.

    Validations:
      * Top-level required keys present.
      * One scene entry per Story scene; counts match.
      * Every Story scene index appears in the response.
      * Every BibleScene field is populated (BibleScene.from_dict
        raises KeyError if any required field is missing — we wrap
        that in ClaudeProtocolError to match the visual_prompts contract).

    Returns the constructed StoryBible (also assigned to ``story.bible``).
    """
    required = {
        "world_genre_atmosphere",
        "character_continuity",
        "environment_continuity",
        "scenes",
    }
    missing = required - set(tool_input.keys())
    if missing:
        raise ClaudeProtocolError(
            f"story_bible response missing top-level keys: {sorted(missing)}"
        )

    raw_scenes = tool_input["scenes"]
    if not isinstance(raw_scenes, list):
        raise ClaudeProtocolError(
            f"story_bible scenes must be a list, got {type(raw_scenes).__name__}"
        )
    if len(raw_scenes) != len(story.scenes):
        raise ClaudeProtocolError(
            f"story_bible scenes count {len(raw_scenes)} != "
            f"story.scenes count {len(story.scenes)}"
        )
    raw_by_index = {item.get("index"): item for item in raw_scenes}
    for scene in story.scenes:
        if scene.index not in raw_by_index:
            raise ClaudeProtocolError(
                f"story_bible response missing scene index {scene.index}"
            )

    try:
        bible = StoryBible.from_dict(tool_input)
    except KeyError as exc:
        raise ClaudeProtocolError(
            f"story_bible response missing required field: {exc}"
        ) from exc
    story.bible = bible
    return bible


# ---------------------------------------------------------------------------
# Top-level coroutine
# ---------------------------------------------------------------------------


async def story_bible(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    db_path: Path, recorder: Recorder | None = None,
) -> tuple[StoryBible, ClaudeResult]:
    """Generate the bible. Mutates story.bible. Returns (bible, ClaudeResult).

    The model + max_tokens come from ``track_cfg["story_bible"]``; tracks
    that opt out should not call this stage at all (the orchestrator skips
    it via composition).
    """
    if not story.scenes:
        raise RuntimeError(
            f"story_bible requires scene_breakdown to have populated story.scenes "
            f"first (story={story.id})."
        )
    bible_cfg = track_cfg.get("story_bible", {})
    if not bible_cfg.get("enabled", False):
        raise RuntimeError(
            f"story_bible called but track {story.track!r} has "
            "story_bible.enabled=false; orchestrator should skip the stage."
        )
    model = bible_cfg.get("model", "claude-opus-4-7")
    max_tokens = int(bible_cfg.get("max_tokens", 16000))

    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
    )
    result = await claude_call(
        model=model, system=system, messages=messages, tool=STORY_BIBLE_TOOL,
        story_id=story.id, stage="story_bible",
        db_path=db_path, recorder=recorder,
        max_tokens=max_tokens,
    )
    bible = _zip_into_story(story, result.tool_input)
    return bible, result


# ---------------------------------------------------------------------------
# Stage subclass
# ---------------------------------------------------------------------------


class StoryBibleStage(Stage):
    """Pipeline stage producing Story.bible. Single Opus 4.7 call per story."""

    name: ClassVar[str] = "story_bible"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        _, result = await story_bible(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
        )
        return {
            "model": result.usage.model,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_read_input_tokens": result.usage.cache_read_input_tokens,
            "cost_usd": result.usage.cost_usd,
        }

    def is_complete(self, story: Story) -> bool:
        """Skip when bible already covers every scene by index."""
        if story.bible is None:
            return False
        bible_indices = {s.index for s in story.bible.scenes}
        story_indices = {s.index for s in story.scenes}
        return bible_indices == story_indices
