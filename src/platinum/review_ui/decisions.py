"""Pure-core review decisions.

Every function takes a Story (and possibly a scene id + action params) and
mutates it in place, returning the mutated Story for chaining. NO I/O,
NO Flask, NO SQLite. The impure shell (app.py routes) wraps these with
load + save + sync_from_story.

Mirrors the pure-core / impure-shell pattern from S3 (story_curator.apply_decision)
and S6 (keyframe_generator.generate_for_scene).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from platinum.models.story import (
    ReviewStatus,
    Scene,
    StageRun,
    StageStatus,
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


def apply_batch_approve_above(story: Story, *, threshold: float) -> Story:
    """Approve all PENDING scenes whose selected candidate's score >= threshold.

    Already-decided scenes (REJECTED / REGENERATE / APPROVED) are left
    untouched -- batch action is additive, never overrides prior intent.
    Scenes without a selected keyframe (keyframe_path is None) are skipped.
    """
    for scene in story.scenes:
        if scene.review_status != ReviewStatus.PENDING:
            continue
        if scene.keyframe_path is None:
            continue
        # Find selected candidate's score
        try:
            selected_idx = scene.keyframe_candidates.index(scene.keyframe_path)
        except ValueError:
            continue  # keyframe_path not in candidates (shouldn't happen, defensive)
        if selected_idx >= len(scene.keyframe_scores):
            continue
        if scene.keyframe_scores[selected_idx] >= threshold:
            scene.review_status = ReviewStatus.APPROVED
    return story


def finalize_review_if_complete(story: Story) -> Story:
    """If every scene is APPROVED, append a COMPLETE StageRun for
    'keyframe_review' (idempotent) and write the review_gates summary.
    """
    if not story.scenes:
        return story
    if any(s.review_status != ReviewStatus.APPROVED for s in story.scenes):
        return story
    # All approved. Check we haven't already finalized.
    existing = story.latest_stage_run("keyframe_review")
    if existing is not None and existing.status == StageStatus.COMPLETE:
        return story  # idempotent

    now = datetime.now()
    regen_total = sum(s.regen_count for s in story.scenes)
    artifacts: dict[str, Any] = {
        "approved_count": len(story.scenes),
        "regen_total": regen_total,
    }
    story.stages.append(
        StageRun(
            stage="keyframe_review",
            status=StageStatus.COMPLETE,
            started_at=now,
            completed_at=now,
            artifacts=dict(artifacts),
        )
    )
    story.review_gates["keyframe_review"] = {
        "completed_at": now.isoformat(),
        "reviewer": "user",
        **artifacts,
    }
    return story
