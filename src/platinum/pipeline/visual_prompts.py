"""visual_prompts pipeline: per-scene Flux visual + negative prompt strings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
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


def _build_request(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
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
            },
        )}
    ]
    return system, messages


def _zip_into_scenes(story_scenes: list[Scene], tool_input: dict) -> list[Scene]:
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
        scene.visual_prompt = item["visual_prompt"]
        scene.negative_prompt = item["negative_prompt"]
        out.append(scene)
    return out


async def visual_prompts(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    db_path: Path, recorder: Recorder | None = None,
) -> tuple[list[Scene], ClaudeResult]:
    if not story.scenes:
        raise RuntimeError(
            f"visual_prompts requires scene_breakdown to have populated story.scenes "
            f"first (story={story.id})."
        )
    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
    )
    result = await claude_call(
        model=MODEL, system=system, messages=messages, tool=VISUAL_PROMPTS_TOOL,
        story_id=story.id, stage="visual_prompts",
        db_path=db_path, recorder=recorder,
    )
    scenes = _zip_into_scenes(story.scenes, result.tool_input)
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
        _, claude_result = await visual_prompts(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
        )
        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
        }
