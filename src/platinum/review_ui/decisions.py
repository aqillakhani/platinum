"""Pure-core review decisions.

Every function takes a Story (and possibly a scene id + action params) and
mutates it in place, returning the mutated Story for chaining. NO I/O,
NO Flask, NO SQLite. The impure shell (app.py routes) wraps these with
load + save + sync_from_story.

Mirrors the pure-core / impure-shell pattern from S3 (story_curator.apply_decision)
and S6 (keyframe_generator.generate_for_scene).
"""
from __future__ import annotations

from platinum.models.story import (
    ReviewStatus,
    Scene,
    Story,
)


def _find_scene(story: Story, scene_id: str) -> Scene:
    """Return the scene with matching id, or raise KeyError."""
    for s in story.scenes:
        if s.id == scene_id:
            return s
    raise KeyError(f"scene id not found: {scene_id}")


def apply_approve(story: Story, scene_id: str) -> Story:
    """Mark scene APPROVED. Idempotent.

    Does not auto-finalize the stage run; caller is responsible for
    invoking finalize_review_if_complete afterwards.
    """
    scene = _find_scene(story, scene_id)
    scene.review_status = ReviewStatus.APPROVED
    return story


def apply_regenerate(story: Story, scene_id: str) -> Story:
    """Mark scene for re-render with same prompt + new seed.

    Bumps regen_count, clears keyframe_path (so re-render runs), preserves
    visual_prompt and review_feedback. CLI then re-runs keyframe_generator
    via `platinum keyframes <id> --rerun-regen-requested`.
    """
    scene = _find_scene(story, scene_id)
    scene.regen_count += 1
    scene.keyframe_path = None
    scene.review_status = ReviewStatus.REGENERATE
    return story


def apply_reject(story: Story, scene_id: str, *, feedback: str) -> Story:
    """Reject scene with textual feedback for `--rerun-rejected` Claude regen.

    Clears visual_prompt and keyframe_path (visual_prompts will rewrite the
    prompt; keyframe_generator re-renders from the new prompt).
    """
    if not feedback or not feedback.strip():
        raise ValueError("feedback is required and must not be blank")
    scene = _find_scene(story, scene_id)
    scene.review_feedback = feedback.strip()
    scene.visual_prompt = None
    scene.keyframe_path = None
    scene.review_status = ReviewStatus.REJECTED
    return story


def apply_swap_candidate(story: Story, scene_id: str, *, candidate_index: int) -> Story:
    """Override the auto-selected best with a different candidate.

    Used by the 'view alternatives' UX to let a reviewer pick a candidate
    the gates ranked lower but that reads better on eye-check. Does not
    change review_status -- the user must explicitly approve afterward.
    """
    scene = _find_scene(story, scene_id)
    if candidate_index < 0 or candidate_index >= len(scene.keyframe_candidates):
        raise IndexError(
            f"candidate_index out of range: {candidate_index} "
            f"(scene has {len(scene.keyframe_candidates)} candidates)"
        )
    scene.keyframe_path = scene.keyframe_candidates[candidate_index]
    return story
