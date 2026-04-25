"""Tests for utils/validate.py quality-gate primitives."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from platinum.utils.validate import CheckResult
from tests._fixtures import make_test_audio


def test_check_result_is_frozen_and_carries_fields() -> None:
    r = CheckResult(passed=True, metric=0.5, threshold=0.4, reason="ok")
    assert r.passed and r.metric == 0.5 and r.threshold == 0.4 and r.reason == "ok"
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.passed = False  # type: ignore[misc]


def test_check_duration_match_exact(tmp_path: Path) -> None:
    from platinum.utils.validate import check_duration_match

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=2.0)
    r = check_duration_match(audio, target_seconds=2.0, tolerance_seconds=0.1)
    assert r.passed
    assert abs(r.metric - 2.0) < 0.05


def test_check_duration_match_within_tolerance(tmp_path: Path) -> None:
    from platinum.utils.validate import check_duration_match

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=2.0)
    r = check_duration_match(audio, target_seconds=2.4, tolerance_seconds=0.5)
    assert r.passed


def test_check_duration_match_outside_tolerance(tmp_path: Path) -> None:
    from platinum.utils.validate import check_duration_match

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=2.0)
    r = check_duration_match(
        audio, target_seconds=10.0, tolerance_seconds=0.5
    )
    assert not r.passed
    assert (
        "tolerance" in r.reason.lower()
        or "exceeds" in r.reason.lower()
        or "deviates" in r.reason.lower()
    )


def test_check_duration_match_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_duration_match

    with pytest.raises(FileNotFoundError):
        check_duration_match(
            tmp_path / "nope.wav", target_seconds=1.0, tolerance_seconds=0.1
        )
