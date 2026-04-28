"""End-to-end happy-path test for the full review loop.

Simulates the user workflow: render → review → reject some + regenerate
some → re-render → re-review → finalize.

Uses the real keyframe_generator + decisions + finalize_review_if_complete
without any GPU calls (fixture data only).

S7 §6.2.
"""
from __future__ import annotations


def test_full_loop_approve_all_via_batch_then_finalize():
    """All scenes pass; batch-approve clears them; stage finalizes."""
    from datetime import UTC, datetime
    from pathlib import Path

    from platinum.models.story import (
        Adapted,
        ReviewStatus,
        Scene,
        Source,
        StageStatus,
        Story,
    )
    from platinum.review_ui import decisions

    src = Source(
        type="gutenberg", url="https://example.com", title="t", author="a",
        raw_text="x", fetched_at=datetime.now(UTC), license="PD-US",
    )
    adapted = Adapted(
        title="t", synopsis="x", narration_script="y",
        estimated_duration_seconds=600.0, tone_notes="z",
    )
    scenes = []
    for i in range(3):
        s = Scene(
            id=f"scene_{i+1:03d}", index=i + 1, narration_text=f"s{i}",
            visual_prompt=f"p{i}", negative_prompt="bright daylight",
        )
        s.keyframe_candidates = [
            Path(f"scene_{i+1:03d}/candidate_{c}.png") for c in range(3)
        ]
        s.keyframe_scores = [5.5, 6.3, 5.9]
        s.keyframe_path = s.keyframe_candidates[1]
        scenes.append(s)
    story = Story(
        id="story_test", track="atmospheric_horror",
        source=src, adapted=adapted, scenes=scenes,
    )

    # All scenes' selected (index 1) score = 6.3 (above 6.0)
    decisions.apply_batch_approve_above(story, threshold=6.0)
    decisions.finalize_review_if_complete(story)
    assert all(s.review_status == ReviewStatus.APPROVED for s in story.scenes)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE


def test_full_loop_reject_then_regen_then_approve_then_finalize():
    """Rejected → regen → approve loop. Stage finalizes only at the end."""
    from datetime import UTC, datetime
    from pathlib import Path

    from platinum.models.story import (
        Adapted,
        ReviewStatus,
        Scene,
        Source,
        StageStatus,
        Story,
    )
    from platinum.review_ui import decisions

    src = Source(
        type="gutenberg", url="https://example.com", title="t", author="a",
        raw_text="x", fetched_at=datetime.now(UTC), license="PD-US",
    )
    adapted = Adapted(
        title="t", synopsis="x", narration_script="y",
        estimated_duration_seconds=600.0, tone_notes="z",
    )
    scenes = []
    for i in range(2):
        s = Scene(
            id=f"scene_{i+1:03d}", index=i + 1, narration_text=f"s{i}",
            visual_prompt=f"p{i}", negative_prompt="bright daylight",
        )
        s.keyframe_candidates = [Path(f"scene_{i+1:03d}/candidate_0.png")]
        s.keyframe_scores = [6.5]
        s.keyframe_path = s.keyframe_candidates[0]
        scenes.append(s)
    story = Story(
        id="t", track="atmospheric_horror",
        source=src, adapted=adapted, scenes=scenes,
    )

    # Approve scene 1, reject scene 2
    decisions.apply_approve(story, "scene_001")
    decisions.apply_reject(story, "scene_002", feedback="too dark")
    decisions.finalize_review_if_complete(story)
    # Not all approved → no StageRun
    assert story.latest_stage_run("keyframe_review") is None

    # Simulate `platinum adapt --rerun-rejected`: rewrite prompt, set REGENERATE
    story.scenes[1].visual_prompt = "rewritten with amber"
    story.scenes[1].review_status = ReviewStatus.REGENERATE
    story.scenes[1].review_feedback = None
    story.scenes[1].keyframe_path = None

    # Simulate `platinum keyframes --rerun-regen-requested`: regenerate keyframe
    story.scenes[1].keyframe_candidates = [Path("scene_002/candidate_0.png")]
    story.scenes[1].keyframe_scores = [6.7]
    story.scenes[1].keyframe_path = story.scenes[1].keyframe_candidates[0]
    story.scenes[1].review_status = ReviewStatus.PENDING

    # User reviews + approves
    decisions.apply_approve(story, "scene_002")
    decisions.finalize_review_if_complete(story)

    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE
    assert run.artifacts["approved_count"] == 2
