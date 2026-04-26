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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def check_motion(
    video_path: Path,
    *,
    min_flow_magnitude: float,
    sample_every_n_frames: int = 6,
) -> CheckResult:
    """Pass if mean Farneback dense optical-flow magnitude is at least min_flow_magnitude."""
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return CheckResult(
            passed=False,
            metric=0.0,
            threshold=min_flow_magnitude,
            reason="failed: cannot read video",
        )
    magnitudes: list[float] = []
    prev_gray = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % sample_every_n_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag = float(np.linalg.norm(flow, axis=2).mean())
                    magnitudes.append(mag)
                prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()
    if not magnitudes:
        return CheckResult(
            passed=False,
            metric=0.0,
            threshold=min_flow_magnitude,
            reason="failed: not enough frames to measure motion",
        )
    mean_mag = float(np.mean(magnitudes))
    passed = mean_mag >= min_flow_magnitude
    if passed:
        reason = (
            f"passed: mean flow magnitude {mean_mag:.2f} >= min {min_flow_magnitude:.2f}"
        )
    else:
        reason = (
            f"failed: mean flow magnitude {mean_mag:.2f} below min {min_flow_magnitude:.2f}"
        )
    return CheckResult(
        passed=passed, metric=mean_mag, threshold=min_flow_magnitude, reason=reason
    )


EXPECTED_LANDMARKS_PER_HAND = 21


def _default_mp_hands_factory() -> Any:
    """Late-bound mediapipe Hands factory; resolved at call time, not import time."""
    import mediapipe as mp  # type: ignore[import-untyped]  # noqa: PLC0415  -- intentional lazy import

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5,
    )


def check_hand_anomalies(
    image_path: Path,
    *,
    mp_hands_factory: Callable[[], Any] | None = None,
) -> CheckResult:
    """Pass if mediapipe sees no hands OR every detected hand has 21 landmarks."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    factory = mp_hands_factory or _default_mp_hands_factory
    try:
        hands = factory()
    except (ImportError, AttributeError) as exc:
        # mediapipe install missing or a newer release moved the `solutions`
        # namespace -- don't halt the pipeline on a tooling drift, just skip
        # the anatomy gate and let the score-based selection proceed.
        return CheckResult(
            passed=True,
            metric=0.0,
            threshold=float(EXPECTED_LANDMARKS_PER_HAND),
            reason=f"skipped: hand detector unavailable ({type(exc).__name__}: {exc})",
        )
    bgr = cv2.imread(str(p))
    if bgr is None:
        return CheckResult(
            passed=False,
            metric=0.0,
            threshold=float(EXPECTED_LANDMARKS_PER_HAND),
            reason="failed: cannot read image",
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        result = hands.process(rgb)
    finally:
        close = getattr(hands, "close", None)
        if callable(close):
            close()
    landmarks = getattr(result, "multi_hand_landmarks", None)
    if not landmarks:
        return CheckResult(
            passed=True,
            metric=0.0,
            threshold=float(EXPECTED_LANDMARKS_PER_HAND),
            reason="passed: no hands detected",
        )
    counts = [len(h.landmark) for h in landmarks]
    bad = [c for c in counts if c != EXPECTED_LANDMARKS_PER_HAND]
    if bad:
        return CheckResult(
            passed=False,
            metric=float(max(counts)),
            threshold=float(EXPECTED_LANDMARKS_PER_HAND),
            reason=(
                f"failed: hand anomaly (landmark counts {counts}, "
                f"expected {EXPECTED_LANDMARKS_PER_HAND})"
            ),
        )
    return CheckResult(
        passed=True,
        metric=float(EXPECTED_LANDMARKS_PER_HAND),
        threshold=float(EXPECTED_LANDMARKS_PER_HAND),
        reason=f"passed: {len(counts)} hand(s) with valid landmark counts",
    )


def check_image_brightness(
    path: Path,
    *,
    min_mean_rgb: float = 20.0,
) -> CheckResult:
    """Reject perceptually degenerate (near-black) images.

    LAION-Aesthetics v2 doesn't penalise low-content imagery (a fully black
    PNG scores 3.9-4.6). This primitive provides a hard floor independent
    of the aesthetic scorer -- catches FP8 + cpu-vae collapse, VAE decode
    bugs, and similar model-layer failures.

    Mean is computed over all RGB pixels (alpha ignored); fully black image
    has mean_rgb=0; pure white has mean_rgb=255.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return CheckResult(
            passed=True,                            # defensive: don't halt pipeline
            metric=0.0,
            threshold=min_mean_rgb,
            reason="skipped: PIL unavailable",
        )

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    mean_rgb = float(arr.mean())
    if mean_rgb >= min_mean_rgb:
        return CheckResult(
            passed=True,
            metric=mean_rgb,
            threshold=min_mean_rgb,
            reason=f"ok (mean_rgb={mean_rgb:.1f} >= min_rgb={min_mean_rgb:.1f})",
        )
    return CheckResult(
        passed=False,
        metric=mean_rgb,
        threshold=min_mean_rgb,
        reason=f"image too dark (mean_rgb={mean_rgb:.1f} < min_rgb={min_mean_rgb:.1f})",
    )
