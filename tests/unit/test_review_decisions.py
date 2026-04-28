"""Unit tests for review_ui.decisions pure-core functions.

S7 §3.2 / §6.2.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from platinum.models.story import (
    Adapted,
    ReviewStatus,
    Scene,
    Source,
    StageStatus,
    Story,
)
from platinum.review_ui import decisions


def _make_story(*, n_scenes: int = 3, all_have_keyframes: bool = True) -> Story:
    src = Source(
        type="gutenberg",
        url="https://example.com",
        title="Test",
        author="Test",
        raw_text="hello",
        fetched_at=datetime.now(UTC),
        license="PD-US",
    )
    adapted = Adapted(
        title="Test",
        synopsis="x",
        narration_script="y",
        estimated_duration_seconds=600.0,
        tone_notes="z",
    )
    scenes = []
    for i in range(n_scenes):
        scene = Scene(
            id=f"scene_{i+1:03d}",
            index=i + 1,
            narration_text=f"scene {i}",
            visual_prompt=f"prompt {i}",
            negative_prompt="bright daylight",
        )
        if all_have_keyframes:
            scene.keyframe_candidates = [
                Path(f"scene_{i+1:03d}/candidate_{c}.png") for c in range(3)
            ]
            scene.keyframe_scores = [5.5, 6.2, 5.9]
            scene.keyframe_path = scene.keyframe_candidates[1]  # auto-selected highest
        scenes.append(scene)
    return Story(
        id="story_test", track="atmospheric_horror",
        source=src, adapted=adapted, scenes=scenes,
    )


def test_apply_approve_marks_scene_approved() -> None:
    story = _make_story()
    decisions.apply_approve(story, "scene_001")
    assert story.scenes[0].review_status == ReviewStatus.APPROVED


def test_apply_approve_idempotent() -> None:
    story = _make_story()
    decisions.apply_approve(story, "scene_001")
    decisions.apply_approve(story, "scene_001")
    assert story.scenes[0].review_status == ReviewStatus.APPROVED
    # No exceptions, status unchanged on second call


def test_apply_approve_unknown_scene_id_raises() -> None:
    story = _make_story()
    with pytest.raises(KeyError, match="scene_xyz"):
        decisions.apply_approve(story, "scene_xyz")


def test_apply_regenerate_clears_keyframe_path() -> None:
    story = _make_story()
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].keyframe_path is None


def test_apply_regenerate_bumps_regen_count() -> None:
    story = _make_story()
    assert story.scenes[0].regen_count == 0
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].regen_count == 1
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].regen_count == 2


def test_apply_regenerate_sets_status_REGENERATE() -> None:
    story = _make_story()
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].review_status == ReviewStatus.REGENERATE


def test_apply_regenerate_preserves_visual_prompt() -> None:
    """Same prompt, new seed -- visual_prompt MUST stay intact."""
    story = _make_story()
    original_prompt = story.scenes[0].visual_prompt
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].visual_prompt == original_prompt


def test_apply_reject_writes_feedback() -> None:
    story = _make_story()
    decisions.apply_reject(story, "scene_001", feedback="too dark; need amber lighting")
    assert story.scenes[0].review_feedback == "too dark; need amber lighting"


def test_apply_reject_clears_keyframe_path_and_visual_prompt() -> None:
    story = _make_story()
    decisions.apply_reject(story, "scene_001", feedback="bad")
    assert story.scenes[0].keyframe_path is None
    assert story.scenes[0].visual_prompt is None


def test_apply_reject_sets_status_REJECTED() -> None:
    story = _make_story()
    decisions.apply_reject(story, "scene_001", feedback="bad")
    assert story.scenes[0].review_status == ReviewStatus.REJECTED


def test_apply_reject_empty_feedback_raises() -> None:
    """A reject with no feedback is meaningless -- guard against accidental empty submissions."""
    story = _make_story()
    with pytest.raises(ValueError, match="feedback"):
        decisions.apply_reject(story, "scene_001", feedback="")
    with pytest.raises(ValueError, match="feedback"):
        decisions.apply_reject(story, "scene_001", feedback="   ")


def test_apply_swap_candidate_updates_keyframe_path() -> None:
    story = _make_story()
    original = story.scenes[0].keyframe_path
    decisions.apply_swap_candidate(story, "scene_001", candidate_index=0)
    assert story.scenes[0].keyframe_path == story.scenes[0].keyframe_candidates[0]
    assert story.scenes[0].keyframe_path != original


def test_apply_swap_candidate_invalid_index_raises() -> None:
    story = _make_story()  # 3 candidates per scene
    with pytest.raises(IndexError, match="candidate_index"):
        decisions.apply_swap_candidate(story, "scene_001", candidate_index=99)


def test_apply_swap_candidate_preserves_review_status() -> None:
    """Swapping should not silently approve / unapprove."""
    story = _make_story()
    story.scenes[0].review_status = ReviewStatus.PENDING
    decisions.apply_swap_candidate(story, "scene_001", candidate_index=0)
    assert story.scenes[0].review_status == ReviewStatus.PENDING


def test_apply_batch_approve_above_threshold_marks_pending_only() -> None:
    """Already-decided scenes (REJECTED, REGENERATE, APPROVED) must NOT be touched."""
    story = _make_story(n_scenes=4)
    # Scene 0: PENDING, score 6.2 (above)
    # Scene 1: PENDING, score 5.0 (below)
    # Scene 2: REJECTED already, score 6.5 (above but already decided)
    # Scene 3: APPROVED already, score 6.5
    story.scenes[1].keyframe_scores = [5.0, 5.0, 5.0]
    story.scenes[1].keyframe_path = story.scenes[1].keyframe_candidates[0]
    story.scenes[2].review_status = ReviewStatus.REJECTED
    story.scenes[3].review_status = ReviewStatus.APPROVED

    decisions.apply_batch_approve_above(story, threshold=6.0)

    assert story.scenes[0].review_status == ReviewStatus.APPROVED  # promoted
    assert story.scenes[1].review_status == ReviewStatus.PENDING   # below threshold
    assert story.scenes[2].review_status == ReviewStatus.REJECTED  # left alone
    assert story.scenes[3].review_status == ReviewStatus.APPROVED  # already approved


def test_apply_batch_approve_above_uses_selected_candidate_score() -> None:
    """Threshold compares against the score of the SELECTED candidate, not max(scores)."""
    story = _make_story(n_scenes=1)
    story.scenes[0].keyframe_scores = [5.5, 6.5, 5.9]
    # Manually point keyframe_path at candidate 0 (score 5.5)
    story.scenes[0].keyframe_path = story.scenes[0].keyframe_candidates[0]
    decisions.apply_batch_approve_above(story, threshold=6.0)
    # Selected has score 5.5, below threshold -> not approved
    assert story.scenes[0].review_status == ReviewStatus.PENDING


def test_apply_batch_approve_above_skips_no_keyframe_scenes() -> None:
    """A scene with keyframe_path=None has no selected score; must not approve."""
    story = _make_story(n_scenes=1, all_have_keyframes=False)
    decisions.apply_batch_approve_above(story, threshold=0.0)
    assert story.scenes[0].review_status == ReviewStatus.PENDING


def test_finalize_no_op_when_pending_remains() -> None:
    story = _make_story(n_scenes=3)
    decisions.apply_approve(story, "scene_001")
    decisions.apply_approve(story, "scene_002")
    # scene_003 still PENDING
    decisions.finalize_review_if_complete(story)
    assert story.latest_stage_run("keyframe_review") is None
    assert "keyframe_review" not in story.review_gates


def test_finalize_appends_stagerun_when_all_approved() -> None:
    story = _make_story(n_scenes=3)
    for s in story.scenes:
        decisions.apply_approve(story, s.id)
    decisions.finalize_review_if_complete(story)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE
    assert run.completed_at is not None


def test_finalize_writes_review_gate() -> None:
    story = _make_story(n_scenes=3)
    for s in story.scenes:
        decisions.apply_approve(story, s.id)
    decisions.finalize_review_if_complete(story)
    gate = story.review_gates.get("keyframe_review")
    assert gate is not None
    assert gate["approved_count"] == 3


def test_finalize_idempotent() -> None:
    """Running finalize twice on an already-final story does NOT append a second StageRun."""
    story = _make_story(n_scenes=2)
    for s in story.scenes:
        decisions.apply_approve(story, s.id)
    decisions.finalize_review_if_complete(story)
    decisions.finalize_review_if_complete(story)
    runs = [r for r in story.stages if r.stage == "keyframe_review"]
    assert len(runs) == 1


def test_finalize_records_regen_total_in_artifacts() -> None:
    story = _make_story(n_scenes=2)
    decisions.apply_regenerate(story, "scene_001")  # bump 0->1
    decisions.apply_regenerate(story, "scene_001")  # bump 1->2
    decisions.apply_regenerate(story, "scene_002")  # bump 0->1
    # Now approve them -- pretend GPU re-render happened in between
    for s in story.scenes:
        s.review_status = ReviewStatus.APPROVED
    decisions.finalize_review_if_complete(story)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.artifacts["regen_total"] == 3


def test_apply_reject_bumps_reject_count() -> None:
    """reject_count tracks rejection clicks across the review session."""
    story = _make_story()
    assert story.scenes[0].reject_count == 0
    decisions.apply_reject(story, "scene_001", feedback="too dark")
    assert story.scenes[0].reject_count == 1
    # Subsequent rejects (after CLI flips to REGENERATE then user re-rejects)
    # keep climbing.
    decisions.apply_reject(story, "scene_001", feedback="still too dark")
    assert story.scenes[0].reject_count == 2


def test_finalize_records_rejected_total_in_artifacts() -> None:
    """rejected_total = sum of reject_count across all scenes (S7 §4.4)."""
    story = _make_story(n_scenes=3)
    decisions.apply_reject(story, "scene_001", feedback="too dark")
    decisions.apply_reject(story, "scene_001", feedback="still too dark")
    decisions.apply_reject(story, "scene_002", feedback="bad framing")
    # Pretend the CLI re-prompted + re-rendered + user approved.
    for s in story.scenes:
        s.review_status = ReviewStatus.APPROVED
    decisions.finalize_review_if_complete(story)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.artifacts["rejected_total"] == 3
    # Also visible in review_gates summary.
    assert story.review_gates["keyframe_review"]["rejected_total"] == 3
