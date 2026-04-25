"""Tests for pipeline/keyframe_generator.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def test_keyframe_report_is_frozen_and_carries_fields() -> None:
    from platinum.pipeline.keyframe_generator import KeyframeReport

    r = KeyframeReport(
        scene_index=0,
        candidates=[Path("a"), Path("b"), Path("c")],
        scores=[7.0, 5.0, 3.0],
        anatomy_passed=[True, True, False],
        selected_index=0,
        selected_via_fallback=False,
    )
    assert r.scene_index == 0
    assert r.selected_index == 0
    assert not r.selected_via_fallback
    with pytest.raises(FrozenInstanceError):
        r.scene_index = 1  # type: ignore[misc]


def test_keyframe_generation_error_carries_per_candidate_exceptions() -> None:
    from platinum.pipeline.keyframe_generator import KeyframeGenerationError

    excs: list[BaseException] = [RuntimeError("a"), ValueError("b"), TimeoutError("c")]
    err = KeyframeGenerationError(scene_index=4, exceptions=excs)
    assert err.scene_index == 4
    assert len(err.exceptions) == 3
    assert "scene 4" in str(err) or "scene_index=4" in str(err)
