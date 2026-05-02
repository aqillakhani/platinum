"""Per-scene-aware KeyframeGeneratorStage.is_complete (S8.B.7).

The base ``Stage.is_complete`` checks ``latest_stage_run.status == COMPLETE``.
That worked before S7's review loop, but it's silently wrong for keyframe
re-runs: once the stage has produced any successful run, the orchestrator
skips it even when the user passes ``--scenes 1,2`` and those specific
scenes have ``keyframe_path = None``.

This bug cost two GPU rentals during the S8.B prototype. The fix:
``KeyframeGeneratorStage.is_complete`` is per-scene-aware. With no
scene_filter, it checks every scene; with a filter, only the filtered
scenes are inspected. A scene with ``keyframe_path is None`` makes
the stage incomplete.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from platinum.models.story import Scene, Source, Story
from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage


def _story(scenes_with_keyframe: dict[int, bool]) -> Story:
    """scenes_with_keyframe maps scene index → has_keyframe_path."""
    return Story(
        id="story_isc", track="atmospheric_horror",
        source=Source(
            type="g", url="x", title="t", author="a",
            raw_text="r", fetched_at=datetime(2026, 5, 2),
            license="PD-US",
        ),
        scenes=[
            Scene(
                id=f"scene_{i:03d}", index=i,
                narration_text=f"text {i}",
                keyframe_path=Path(f"/k/{i}.png") if has_kf else None,
            )
            for i, has_kf in scenes_with_keyframe.items()
        ],
    )


def test_is_complete_true_when_all_scenes_have_keyframe_no_filter() -> None:
    """No scene_filter, every scene has a keyframe → stage is complete."""
    story = _story({1: True, 2: True, 3: True})
    stage = KeyframeGeneratorStage()
    assert stage.is_complete(story) is True


def test_is_complete_false_when_one_scene_missing_keyframe_no_filter() -> None:
    """No scene_filter; scene 2 missing keyframe → stage is incomplete."""
    story = _story({1: True, 2: False, 3: True})
    stage = KeyframeGeneratorStage()
    assert stage.is_complete(story) is False


def test_is_complete_false_when_filtered_scene_missing_keyframe() -> None:
    """scene_filter=[2,3]; scene 2 has no keyframe → incomplete (must run)."""
    story = _story({1: True, 2: False, 3: True})
    stage = KeyframeGeneratorStage(scene_filter={2, 3})
    assert stage.is_complete(story) is False


def test_is_complete_true_when_filter_excludes_missing_scene() -> None:
    """scene_filter=[1,3]; only scene 2 missing but it's NOT in filter →
    complete from this run's perspective. The pre-S8.B.7 behavior would
    silently skip the stage when stage_run was COMPLETE; this test pins
    the new per-scene-aware semantics so the orchestrator runs the stage
    only when it actually has work in the active filter."""
    story = _story({1: True, 2: False, 3: True})
    stage = KeyframeGeneratorStage(scene_filter={1, 3})
    assert stage.is_complete(story) is True


def test_is_complete_true_for_story_with_no_scenes() -> None:
    """Degenerate: a story with zero scenes vacuously has all keyframes."""
    story = _story({})
    stage = KeyframeGeneratorStage()
    assert stage.is_complete(story) is True


def test_is_complete_ignores_stage_run_status() -> None:
    """Even when a previous keyframe_generator StageRun was COMPLETE, a
    scene whose keyframe_path was later cleared (e.g. user marked it for
    regen) makes the stage incomplete. This is the prototype's bug:
    once the stage had run once, the base is_complete returned True
    forever, so partial reruns silently skipped."""
    from platinum.models.story import StageRun, StageStatus

    story = _story({1: True, 2: False, 3: True})
    story.stages.append(
        StageRun(
            stage="keyframe_generator",
            status=StageStatus.COMPLETE,
            started_at=datetime(2026, 5, 1),
            completed_at=datetime(2026, 5, 1),
        )
    )
    stage = KeyframeGeneratorStage()
    assert stage.is_complete(story) is False  # not True per base impl


def test_is_complete_scene_filter_persists_across_calls() -> None:
    """Constructor-provided scene_filter is sticky on the instance."""
    story = _story({1: True, 2: False, 3: False, 4: True})
    stage = KeyframeGeneratorStage(scene_filter={1, 4})
    assert stage.is_complete(story) is True
    assert stage.is_complete(story) is True  # idempotent — same answer
