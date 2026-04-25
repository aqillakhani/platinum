"""Smoke tests for synthetic media helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tests._fixtures import (
    make_silent_audio,
    make_test_audio,
    make_test_video,
    make_test_video_with_motion,
)


def test_make_test_video_writes_readable_mp4(tmp_path: Path) -> None:
    out = tmp_path / "v.mp4"
    make_test_video(out, n_frames=10, fps=24, color=(0, 0, 0), size=(64, 64))
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    try:
        ok, frame = cap.read()
        assert ok and frame is not None
        assert frame.shape == (64, 64, 3)
        assert frame.mean() < 5.0  # all-black frame
    finally:
        cap.release()


def test_make_test_video_with_motion_has_frame_to_frame_change(tmp_path: Path) -> None:
    out = tmp_path / "moving.mp4"
    make_test_video_with_motion(out, n_frames=10, fps=24, size=(64, 64))
    cap = cv2.VideoCapture(str(out))
    try:
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        assert ok1 and ok2
        diff = np.abs(f1.astype(np.int16) - f2.astype(np.int16)).mean()
        assert diff > 1.0
    finally:
        cap.release()


def test_make_test_audio_writes_wav_with_expected_duration(tmp_path: Path) -> None:
    import wave

    out = tmp_path / "tone.wav"
    make_test_audio(out, seconds=2.0, freq_hz=440.0)
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / rate
        assert 1.95 <= duration <= 2.05


def test_make_silent_audio_writes_wav_with_zero_amplitude(tmp_path: Path) -> None:
    import wave

    out = tmp_path / "silent.wav"
    make_silent_audio(out, seconds=1.0)
    with wave.open(str(out), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        assert int(np.abs(samples).max()) == 0
