"""Quality-gate primitives.

Five sync `check_*` functions returning a uniform `CheckResult`. Used by
Sessions 6/8/13 to reject AI-generated assets before human review.

`passed=True` always means the asset cleared the gate.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a quality check.

    `metric` is the measured value; `threshold` is what it was compared
    against; `reason` is a human-readable summary suitable for log lines
    and review-UI display.
    """

    passed: bool
    metric: float
    threshold: float
    reason: str


def _ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe.

    Raises FileNotFoundError if file or ffprobe missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe not on PATH (install ffmpeg).")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def check_duration_match(
    audio_path: Path,
    *,
    target_seconds: float,
    tolerance_seconds: float,
) -> CheckResult:
    """Pass if abs(measured_duration - target_seconds) <= tolerance_seconds."""
    measured = _ffprobe_duration(Path(audio_path))
    delta = abs(measured - target_seconds)
    passed = delta <= tolerance_seconds
    if passed:
        reason = (
            f"passed: {measured:.2f}s within {tolerance_seconds:.2f}s of "
            f"target {target_seconds:.2f}s"
        )
    else:
        reason = (
            f"failed: {measured:.2f}s deviates {delta:.2f}s from target "
            f"{target_seconds:.2f}s (tolerance {tolerance_seconds:.2f}s)"
        )
    return CheckResult(passed=passed, metric=measured, threshold=target_seconds, reason=reason)


def _measure_lufs(path: Path) -> float:
    """Return integrated LUFS via ffmpeg loudnorm. -inf for silent audio."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not on PATH.")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    # loudnorm prints JSON to stderr after the analysis pass.
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.DOTALL)
    if not match:
        raise RuntimeError(f"loudnorm output not parseable: {result.stderr[-400:]}")
    data = json.loads(match.group(0))
    raw = data.get("input_i", "-inf")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def check_audio_levels(
    audio_path: Path,
    *,
    target_lufs: float,
    tolerance_db: float,
) -> CheckResult:
    """Pass if measured integrated LUFS is within tolerance_db of target_lufs."""
    measured = _measure_lufs(Path(audio_path))
    if math.isinf(measured) and measured < 0:
        return CheckResult(
            passed=False,
            metric=measured,
            threshold=target_lufs,
            reason="failed: silent audio (-inf LUFS)",
        )
    delta = abs(measured - target_lufs)
    passed = delta <= tolerance_db
    if passed:
        reason = (
            f"passed: {measured:.1f} LUFS within {tolerance_db:.1f} dB of target {target_lufs:.1f}"
        )
    else:
        reason = (
            f"failed: {measured:.1f} LUFS deviates {delta:.1f} dB from target "
            f"{target_lufs:.1f} (tolerance {tolerance_db:.1f} dB)"
        )
    return CheckResult(passed=passed, metric=measured, threshold=target_lufs, reason=reason)


def check_black_frames(
    video_path: Path,
    *,
    max_black_ratio: float,
    luminance_threshold: float = 8.0,
) -> CheckResult:
    """Pass if share of near-black frames is at most max_black_ratio."""
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return CheckResult(
            passed=False,
            metric=1.0,
            threshold=max_black_ratio,
            reason="failed: cannot read video",
        )
    total = 0
    black = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            total += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(gray.mean()) < luminance_threshold:
                black += 1
    finally:
        cap.release()
    if total == 0:
        return CheckResult(
            passed=False,
            metric=1.0,
            threshold=max_black_ratio,
            reason="failed: no frames decoded",
        )
    ratio = black / total
    passed = ratio <= max_black_ratio
    if passed:
        reason = f"passed: {ratio:.3f} black-frame ratio (max {max_black_ratio:.3f})"
    else:
        reason = (
            f"failed: {ratio:.3f} black-frame ratio exceeds max {max_black_ratio:.3f} "
            f"({black}/{total} frames below luminance {luminance_threshold:.1f})"
        )
    return CheckResult(passed=passed, metric=ratio, threshold=max_black_ratio, reason=reason)
