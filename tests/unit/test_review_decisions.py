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
