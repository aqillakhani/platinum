"""motion_prompts pipeline: per-scene keyframe-grounded motion prompt for Wan I2V.

Inserts between keyframe_review (where one of N candidates is chosen) and
video_generator (where Wan 2.2 14B I2V animates the chosen keyframe). For
each approved scene, a Claude vision call looks at the actual chosen
keyframe and writes a 5-second motion prompt that fits *that depicted
state* — preventing the prompt-keyframe state mismatch that produced
the action duplication / reverse motion / object multiplication failures
in the S8.18 verify run.

Architecture: one Claude call per scene (per-scene cost ~$0.005 on Haiku
4.5). Idempotent — skips scenes whose motion_prompt is already set so a
resumed run is free.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage
from platinum.utils.claude import ClaudeProtocolError, ClaudeResult, Recorder
from platinum.utils.claude import call as claude_call
from platinum.utils.prompts import render_template

logger = logging.getLogger(__name__)
MODEL = "claude-haiku-4-5"  # vision-capable, ~$0.005 per call


MOTION_PROMPT_TOOL: dict[str, Any] = {
    "name": "submit_motion_prompt",
    "description": "Submit a 5-second motion prompt grounded in the keyframe.",
    "input_schema": {
        "type": "object",
        "required": ["motion_prompt"],
        "properties": {
            "motion_prompt": {
                "type": "string",
                "description": (
                    "5-second motion prompt for Wan 2.2 14B I2V, grounded in "
                    "what the keyframe depicts. Under 80 words. Camera language "
                    "preferred when subject motion would contradict the keyframe."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "Optional 1-2 sentences: why this motion fits the keyframe "
                    "state."
                ),
            },
        },
    },
}


MOTION_PROMPT_SYSTEM = (
    "You write 5-second motion prompts for Wan 2.2 14B I2V (image-to-video). "
    "The keyframe you see is the START STATE; the video runs forward 5 seconds. "
    "Rules:\n"
    "1. Never re-perform an action the keyframe has already completed "
    "(don't 'tie a mask' if the mask is already tied).\n"
    "2. Use motion verbs that fit a 5-second window. 'Walks across a room' "
    "fits; 'descends three flights of stairs' does not.\n"
    "3. Prefer camera language ('slow dolly forward', 'subtle pan right', "
    "'tilt down') when subject motion would contradict the keyframe state.\n"
    "4. Don't restate composition the keyframe already shows.\n"
    "5. Atmospheric continuity verbs welcome: torchlight flickers, smoke "
    "drifts, breath fogs, candle wax pools.\n"
    "6. Avoid: stacked actions ('tying mask while turning while stepping'), "
    "state changes the keyframe contradicts, narrative summary, "
    "object multiplication.\n"
    "7. Under 80 words, one paragraph, no bullet points.\n"
    "Submit via the submit_motion_prompt tool."
)


def _read_keyframe_b64(path: Path) -> tuple[str, str]:
    """Return (base64-encoded image bytes, IANA media type)."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    media_type = (
        "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    )
    return data, media_type


def _build_request(
    *,
    scene: Scene,
    track_cfg: dict,
    prompts_dir: Path,
    story_track: str,
) -> tuple[list[dict], list[dict]]:
    """Build (system, messages) for one scene's motion-prompt Claude call.

    The user message is a multi-part content list with:
      - one image content block (the keyframe, base64),
      - one text block with the rendered template (track aesthetic +
        scene narration + original visual_prompt as context).
    """
    if scene.keyframe_path is None:
        raise ValueError(f"scene {scene.index}: keyframe_path is None")
    img_b64, media_type = _read_keyframe_b64(Path(scene.keyframe_path))

    visual = track_cfg.get("visual", {})
    body_text = render_template(
        prompts_dir=prompts_dir,
        track=story_track,
        name="motion_prompt.j2",
        context={
            "aesthetic": visual.get("aesthetic", ""),
            "narration_text": scene.narration_text,
            "visual_prompt": scene.visual_prompt or "",
            "scene_index": scene.index,
        },
    )

    system = [{"type": "text", "text": MOTION_PROMPT_SYSTEM}]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": body_text},
            ],
        }
    ]
    return system, messages


def _parse_response(tool_input: dict, *, scene_index: int) -> str:
    """Extract motion_prompt from tool_input. Raises ClaudeProtocolError if absent."""
    mp = tool_input.get("motion_prompt")
    if not mp or not isinstance(mp, str):
        raise ClaudeProtocolError(
            f"motion_prompts: scene {scene_index} response missing motion_prompt"
        )
    return mp


async def motion_prompts(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    db_path: Path,
    recorder: Recorder | None = None,
) -> list[tuple[Scene, ClaudeResult]]:
    """Per-scene motion-prompt Claude calls.

    Skip rules (idempotent + cost-bounded):
      - scene.keyframe_path is None       -> skip (nothing to ground against)
      - scene.motion_prompt already set   -> skip (resumed runs are free)

    Mutates each processed scene's motion_prompt in place. Returns one
    (scene, ClaudeResult) tuple per Claude call so callers can roll up
    aggregate cost into the stage's artifacts dict.
    """
    results: list[tuple[Scene, ClaudeResult]] = []
    for scene in story.scenes:
        if scene.keyframe_path is None:
            logger.debug(
                "motion_prompts: skipping scene %s -- no keyframe_path",
                scene.index,
            )
            continue
        if scene.motion_prompt:
            logger.debug(
                "motion_prompts: skipping scene %s -- motion_prompt already set",
                scene.index,
            )
            continue
        system, messages = _build_request(
            scene=scene,
            track_cfg=track_cfg,
            prompts_dir=prompts_dir,
            story_track=story.track,
        )
        result = await claude_call(
            model=MODEL,
            system=system,
            messages=messages,
            tool=MOTION_PROMPT_TOOL,
            story_id=story.id,
            stage="motion_prompts",
            db_path=db_path,
            recorder=recorder,
        )
        scene.motion_prompt = _parse_response(
            result.tool_input, scene_index=scene.index
        )
        results.append((scene, result))
    return results


class MotionPromptsStage(Stage):
    name: ClassVar[str] = "motion_prompts"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        results = await motion_prompts(
            story=story,
            track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path,
            recorder=recorder,
        )
        total_cost = sum(r.usage.cost_usd for _, r in results)
        total_in = sum(r.usage.input_tokens for _, r in results)
        total_out = sum(r.usage.output_tokens for _, r in results)
        return {
            "scenes_processed": len(results),
            "cost_usd": total_cost,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "model": MODEL,
        }
