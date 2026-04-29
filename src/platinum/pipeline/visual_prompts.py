"""visual_prompts pipeline: per-scene Flux visual + negative prompt strings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import ReviewStatus, Scene, Story
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
from platinum.utils.prompts import (
    render_template,
)

logger = logging.getLogger(__name__)
MODEL = "claude-opus-4-7"


VISUAL_PROMPTS_TOOL: dict[str, Any] = {
    "name": "submit_visual_prompts",
    "description": "Submit per-scene Flux visual + negative prompts.",
    "input_schema": {
        "type": "object",
        "required": ["scenes"],
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "visual_prompt", "negative_prompt"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 1},
                        "visual_prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                    },
                },
            },
        },
    },
}


def _load_characters(story_id: str, stories_dir: Path) -> dict[str, str]:
    """Load per-story characters dict from stories_dir/<story_id>/characters.json.

    Returns {} when the file does not exist. Phase A ships this file as
    optional and gitignored; Phase B B3.3 auto-extracts characters from
    the breakdown stage. The returned dict is fed to visual_prompts.j2's
    TRACK CHARACTERS section.
    """
    path = stories_dir / story_id / "characters.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_request(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    deviation_feedback: list | None = None,
    characters: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    system = [
        {"type": "text", "text": render_template(
            prompts_dir=prompts_dir, track=story.track,
            name="system.j2", context={"track": track_cfg},
        )}
    ]
    messages = [
        {"role": "user", "content": render_template(
            prompts_dir=prompts_dir, track=story.track, name="visual_prompts.j2",
            context={
                "aesthetic": track_cfg["visual"]["aesthetic"],
                "palette": track_cfg["visual"]["palette"],
                "default_negative": track_cfg["visual"]["negative_prompt"],
                "scenes": [
                    {"index": s.index, "narration_text": s.narration_text}
                    for s in story.scenes
                ],
                "characters": characters or {},
                "deviation_feedback": deviation_feedback,
            },
        )}
    ]
    return system, messages


def _zip_into_scenes(
    story_scenes: list[Scene], tool_input: dict,
    *, scene_filter: set[int] | None = None,
) -> list[Scene]:
    """Mutate scenes with new visual_prompt + negative_prompt by index match.

    When scene_filter is set, only apply to scenes whose index is in the
    filter. For REJECTED scenes that get applied, also clear review_feedback
    and keyframe_path and flip status to REGENERATE.

    Raises ClaudeProtocolError on count mismatch or missing scene index.
    """
    raw = tool_input.get("scenes", [])
    if len(raw) != len(story_scenes):
        raise ClaudeProtocolError(
            f"visual_prompts count {len(raw)} != scene count {len(story_scenes)}"
        )
    by_index = {item["index"]: item for item in raw}
    out: list[Scene] = []
    for scene in story_scenes:
        item = by_index.get(scene.index)
        if item is None:
            raise ClaudeProtocolError(
                f"visual_prompts response missing scene index {scene.index}"
            )
        if scene_filter is not None and scene.index not in scene_filter:
            continue
        scene.visual_prompt = item["visual_prompt"]
        scene.negative_prompt = item["negative_prompt"]
        # S7.1.B2.3 -- optional composition_notes + character_refs from the new
        # template. Default to leaving Scene fields untouched when the keys are
        # absent so old recorded fixtures (and tracks that don't ask for these)
        # keep round-tripping.
        if "composition_notes" in item:
            scene.composition_notes = item["composition_notes"]
        if "character_refs" in item:
            scene.character_refs = list(item["character_refs"])
        if scene.review_status == ReviewStatus.REJECTED:
            scene.review_status = ReviewStatus.REGENERATE
            scene.review_feedback = None
            scene.keyframe_path = None
        out.append(scene)
    return out


async def visual_prompts(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    db_path: Path, recorder: Recorder | None = None,
    scene_filter: set[int] | None = None,
    deviation_feedback: list | None = None,
    stories_dir: Path | None = None,
) -> tuple[list[Scene], ClaudeResult]:
    """Run the visual_prompts call. Mutates story.scenes in place
    (sets visual_prompt and negative_prompt per scene) and returns it
    alongside the ClaudeResult. Raises if story.scenes is empty or
    if the response count/indexes don't match story.scenes.

    When stories_dir is provided, the per-story characters dict is loaded
    from stories_dir/<story.id>/characters.json (empty if missing) and
    threaded into the visual_prompts.j2 TRACK CHARACTERS section.
    """
    if not story.scenes:
        raise RuntimeError(
            f"visual_prompts requires scene_breakdown to have populated story.scenes "
            f"first (story={story.id})."
        )
    characters = _load_characters(story.id, stories_dir) if stories_dir else {}
    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
        deviation_feedback=deviation_feedback,
        characters=characters,
    )
    result = await claude_call(
        model=MODEL, system=system, messages=messages, tool=VISUAL_PROMPTS_TOOL,
        story_id=story.id, stage="visual_prompts",
        db_path=db_path, recorder=recorder,
    )
    scenes = _zip_into_scenes(story.scenes, result.tool_input, scene_filter=scene_filter)
    return scenes, result


class VisualPromptsStage(Stage):
    name: ClassVar[str] = "visual_prompts"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        if not story.scenes:
            raise RuntimeError(
                f"visual_prompts requires scene_breakdown completed first (story={story.id})."
            )
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        runtime = ctx.config.settings.get("runtime", {})
        scene_filter_raw = runtime.get("scene_filter")
        scene_filter = set(scene_filter_raw) if scene_filter_raw is not None else None
        deviation_feedback = runtime.get("deviation_feedback")
        _, claude_result = await visual_prompts(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
            scene_filter=scene_filter,
            deviation_feedback=deviation_feedback,
            stories_dir=ctx.config.stories_dir,
        )
        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
        }
