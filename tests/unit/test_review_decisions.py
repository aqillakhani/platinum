"""Unit tests for review_ui.decisions pure-core functions.

S7 §3.2 / §6.2.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
        fetched_at=datetime.now(timezone.utc),
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
