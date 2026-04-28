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
