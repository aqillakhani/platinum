"""Quality-gate primitives.

Five sync `check_*` functions returning a uniform `CheckResult`. Used by
Sessions 6/8/13 to reject AI-generated assets before human review.

`passed=True` always means the asset cleared the gate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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
            "-v", "quiet",
            "-print_format", "json",
            "-show_entries", "format=duration",
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
    return CheckResult(
        passed=passed, metric=measured, threshold=target_seconds, reason=reason
    )
