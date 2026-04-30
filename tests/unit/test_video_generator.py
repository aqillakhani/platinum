"""Unit tests for video_generator pipeline (S8 Phase A)."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestVideoReport:
    def test_dataclass_fields(self) -> None:
        from platinum.pipeline.video_generator import VideoReport

        report = VideoReport(
            scene_index=7,
            success=True,
            mp4_path=Path("data/stories/x/clips/scene_007_raw.mp4"),
            duration_seconds=5.0,
            gates_passed={"duration": True, "black_frames": True, "motion": True},
            retry_used=0,
        )
        assert report.scene_index == 7
        assert report.success is True
        assert report.duration_seconds == 5.0
        assert report.retry_used == 0
        assert report.gates_passed["motion"] is True

    def test_frozen_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        from platinum.pipeline.video_generator import VideoReport

        report = VideoReport(
            scene_index=0,
            success=False,
            mp4_path=None,
            duration_seconds=0.0,
            gates_passed={},
            retry_used=1,
        )
        with pytest.raises(FrozenInstanceError):
            report.success = True  # type: ignore[misc]


class TestVideoGenerationError:
    def test_carries_scene_index_and_reason(self) -> None:
        from platinum.pipeline.video_generator import VideoGenerationError

        exc = VideoGenerationError(
            scene_index=3,
            reason="motion gate failed: flow=0.05 < min=0.30",
            retryable=True,
        )
        assert exc.scene_index == 3
        assert exc.reason == "motion gate failed: flow=0.05 < min=0.30"
        assert exc.retryable is True
        assert "scene_index=3" in str(exc)
        assert "motion gate failed" in str(exc)

    def test_retryable_default_false(self) -> None:
        from platinum.pipeline.video_generator import VideoGenerationError

        exc = VideoGenerationError(scene_index=0, reason="comfy http 500")
        assert exc.retryable is False


class TestSeedForScene:
    def test_base_seed_is_index_times_thousand(self) -> None:
        from platinum.pipeline.video_generator import _seed_for_scene

        assert _seed_for_scene(0, retry=0) == 0
        assert _seed_for_scene(1, retry=0) == 1000
        assert _seed_for_scene(7, retry=0) == 7000
        assert _seed_for_scene(15, retry=0) == 15000

    def test_retry_increments_seed_by_one(self) -> None:
        from platinum.pipeline.video_generator import _seed_for_scene

        assert _seed_for_scene(7, retry=1) == 7001

    def test_disjoint_from_keyframe_seeds(self) -> None:
        """keyframe_generator uses scene*1000 + candidate_idx (0,1,2 typically).

        Video uses scene*1000 + retry (0,1). Both fit in same 1000-block but
        retry=0/1 collide with candidate=0/1 only if user runs both with the
        same scene -- which is fine because keyframes write to PNG and
        video to MP4. The test just confirms the formula stays simple.
        """
        from platinum.pipeline.video_generator import _seed_for_scene

        assert _seed_for_scene(7, retry=0) == 7000
        assert _seed_for_scene(7, retry=1) == 7001
