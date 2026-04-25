"""Tests for utils/validate.py quality-gate primitives."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from platinum.utils.validate import CheckResult
from tests._fixtures import (
    make_silent_audio,
    make_test_audio,
    make_test_video,
    make_test_video_with_motion,
)


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
    r = check_duration_match(audio, target_seconds=10.0, tolerance_seconds=0.5)
    assert not r.passed
    assert (
        "tolerance" in r.reason.lower()
        or "exceeds" in r.reason.lower()
        or "deviates" in r.reason.lower()
    )


def test_check_duration_match_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_duration_match

    with pytest.raises(FileNotFoundError):
        check_duration_match(tmp_path / "nope.wav", target_seconds=1.0, tolerance_seconds=0.1)


def test_check_audio_levels_passes_for_tone_in_range(tmp_path: Path) -> None:
    """Generate a tone, measure its LUFS, then verify check passes when target=measured."""
    from platinum.utils.validate import _measure_lufs, check_audio_levels

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=3.0, freq_hz=1000.0, amplitude=0.25)
    measured = _measure_lufs(audio)
    r = check_audio_levels(audio, target_lufs=measured, tolerance_db=0.5)
    assert r.passed


def test_check_audio_levels_fails_when_far_from_target(tmp_path: Path) -> None:
    from platinum.utils.validate import check_audio_levels

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=3.0, freq_hz=1000.0, amplitude=0.25)
    r = check_audio_levels(audio, target_lufs=-40.0, tolerance_db=1.0)
    assert not r.passed


def test_check_audio_levels_silent_audio_marks_failed(tmp_path: Path) -> None:
    from platinum.utils.validate import check_audio_levels

    audio = tmp_path / "silent.wav"
    make_silent_audio(audio, seconds=2.0)
    r = check_audio_levels(audio, target_lufs=-16.0, tolerance_db=1.5)
    assert not r.passed
    assert "silent" in r.reason.lower() or "inf" in r.reason.lower()


def test_check_audio_levels_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_audio_levels

    with pytest.raises(FileNotFoundError):
        check_audio_levels(tmp_path / "nope.wav", target_lufs=-16.0, tolerance_db=1.0)


def test_check_black_frames_clean_motion_video_passes(tmp_path: Path) -> None:
    from platinum.utils.validate import check_black_frames

    video = tmp_path / "moving.mp4"
    make_test_video_with_motion(video, n_frames=20, fps=24, size=(64, 64))
    r = check_black_frames(video, max_black_ratio=0.05, luminance_threshold=8.0)
    assert r.passed
    assert r.metric < 0.05


def test_check_black_frames_all_black_video_fails(tmp_path: Path) -> None:
    from platinum.utils.validate import check_black_frames

    video = tmp_path / "black.mp4"
    make_test_video(video, n_frames=20, fps=24, color=(0, 0, 0), size=(64, 64))
    r = check_black_frames(video, max_black_ratio=0.05, luminance_threshold=8.0)
    assert not r.passed
    assert r.metric > 0.9


def test_check_black_frames_corrupt_video_fails_gracefully(tmp_path: Path) -> None:
    from platinum.utils.validate import check_black_frames

    video = tmp_path / "corrupt.mp4"
    video.write_bytes(b"not a real mp4")
    r = check_black_frames(video, max_black_ratio=0.05, luminance_threshold=8.0)
    assert not r.passed
    assert "cannot read" in r.reason.lower() or "no frames" in r.reason.lower()


def test_check_black_frames_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_black_frames

    with pytest.raises(FileNotFoundError):
        check_black_frames(tmp_path / "nope.mp4", max_black_ratio=0.05, luminance_threshold=8.0)
