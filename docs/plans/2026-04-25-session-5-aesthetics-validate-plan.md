# Session 5 -- Aesthetic scoring & validation utilities (implementation plan)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Ship `src/platinum/utils/validate.py` (uniform `CheckResult` dataclass + five sync `check_*` functions: black frames, motion, hand anomalies, audio LUFS, duration match) and `src/platinum/utils/aesthetics.py` (`AestheticScorer` Protocol + `FakeAestheticScorer`). Add a `quality_gates:` block to every `config/tracks/*.yaml`. All primitives unit-tested offline; one integration test round-trips a real ffmpeg LUFS measurement.

**Architecture:** Leaf-level utilities. No `Stage` subclass wiring this session -- Sessions 6/8/13 will compose them. Real `AestheticScorer` deferred to Session 6 (GPU on vast.ai). Late-binding for testability throughout (`mp_hands_factory=None`, resolved at call time). Test fixtures generated synthetically at test time; mediapipe mocked at the boundary.

**Tech stack:** Python 3.14, `opencv-python>=4.10`, `mediapipe>=0.10`, `ffmpeg-python>=0.2`, `numpy>=1.26`, stdlib `wave`, `pytest`+`pytest-asyncio` (auto mode already on). Zero new dependencies.

**Design doc:** `docs/plans/2026-04-25-session-5-aesthetics-validate-design.md` -- read first if any task feels under-specified.

---

## Pre-flight context (read before Task 1)

### Existing files you will integrate with

- `src/platinum/config.py` -- `Config.track(track_id)` returns a track dict; track YAML is parsed from `data["track"]` so `quality_gates` lands as `config.track("atmospheric_horror")["quality_gates"]`.
- `tests/conftest.py` -- exposes `tmp_project`, `config`, `context`, `source`, `story` fixtures. `tmp_project` does NOT seed `config/tracks/*.yaml` -- integration tests that need a real track config must write a track YAML themselves.
- `tests/_fixtures.py` -- already contains `FixtureRecorder` and `FixtureMissingError` from Session 4. **Extend this file**, do not create a new one. Append synthetic-media helpers below the existing class.
- `config/tracks/atmospheric_horror.yaml` -- existing track config under a top-level `track:` key. The new `quality_gates:` block goes inside `track:`.

### Conventions established in earlier sessions

- ASCII only in any string that flows to a Windows console. No smart quotes, em dashes, or fancy arrows.
- Pure-core / impure-shell: utilities export pure-ish functions; injectable seams use late binding (`runner=None`, `factory=None`, resolved at call time).
- `pyproject.toml` already has `[tool.pytest.ini_options] asyncio_mode = "auto"`. Plain `def test_x():` for sync tests; `async def test_x():` for async.
- Dataclasses default to `frozen=True, slots=True` for value types.
- ruff config in `pyproject.toml`: line-length 100, target py311, rules E/F/W/I/B/UP. Run `ruff check src tests` before each commit.

### What you are NOT building this session

- Real LAION-Aesthetics model integration -- Session 6.
- Any `Stage` subclass wrapping these primitives -- Session 6.
- A validation orchestrator that aggregates results across primitives -- defer until a caller needs it.
- LUT calibration, ASS subtitle validation, audio-clipping detection -- not in scope.

### Library notes that will save debugging time

- **OpenCV reads BGR, mediapipe wants RGB.** `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before `.process()`.
- **cv2.VideoWriter on Windows** -- use `cv2.VideoWriter_fourcc(*'mp4v')` and `.mp4` extension. Frame size and color depth must match the writer.
- **ffmpeg `loudnorm` prints JSON to stderr**, not stdout. Use `subprocess.run([...], capture_output=True, text=True)` and parse the JSON block out of `result.stderr`. The block is delimited by `{...}` -- find the last balanced `{...}` in stderr.
- **ffprobe duration** -- `ffprobe -v quiet -print_format json -show_entries format=duration <file>` returns a clean JSON object on stdout.
- **mediapipe Hands API** -- `mp.solutions.hands.Hands()` instance, then `.process(rgb_image)` returns a result whose `.multi_hand_landmarks` is either `None` (no hands) or a list of objects each with `.landmark` (a list of 21 NormalizedLandmark in valid mediapipe output).
- **LUFS calibration** -- for `make_test_audio(lufs=...)` to actually produce the requested LUFS, we calibrate amplitude empirically: generate at a target dB FS, measure with loudnorm once, store the calibration constant. For pragmatic tests, generate at amplitude 0.25 (~-12 dB FS, ~-10 LUFS K-weighted for a 1 kHz tone) and assert the measured LUFS falls in a wide tolerance.

---

## Task 1: Synthetic media fixture helpers

**Files:**
- Modify: `tests/_fixtures.py` -- append helpers below `FixtureRecorder`.
- Test: `tests/unit/test_fixture_helpers.py` -- new file.

### Step 1: Write the failing tests

Create `tests/unit/test_fixture_helpers.py`:

```python
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
```

### Step 2: Run, see fail

```
pytest tests/unit/test_fixture_helpers.py -v
```
Expected: 4 ImportErrors -- helpers not defined yet.

### Step 3: Implement the helpers

Append to `tests/_fixtures.py`:

```python
import math
import wave

import cv2
import numpy as np


def make_test_video(
    path: Path,
    *,
    n_frames: int = 30,
    fps: int = 24,
    color: tuple[int, int, int] = (0, 0, 0),
    size: tuple[int, int] = (64, 64),
) -> None:
    """Write a single-color MP4 (BGR) of `n_frames` at `fps`."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {path}")
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    try:
        for _ in range(n_frames):
            writer.write(frame)
    finally:
        writer.release()


def make_test_video_with_motion(
    path: Path,
    *,
    n_frames: int = 30,
    fps: int = 24,
    size: tuple[int, int] = (64, 64),
) -> None:
    """Write an MP4 with frame-to-frame motion (translating checkerboard)."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {path}")
    base = np.zeros((h, w, 3), dtype=np.uint8)
    block = 8
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                base[y : y + block, x : x + block] = 255
    try:
        for i in range(n_frames):
            shift = (i * 2) % w
            frame = np.roll(base, shift, axis=1)
            writer.write(frame)
    finally:
        writer.release()


def make_test_audio(
    path: Path,
    *,
    seconds: float = 2.0,
    freq_hz: float = 440.0,
    sample_rate: int = 48000,
    amplitude: float = 0.25,
) -> None:
    """Write a mono 16-bit WAV containing a sine tone."""
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    samples = np.sin(2.0 * math.pi * freq_hz * t) * amplitude
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def make_silent_audio(
    path: Path,
    *,
    seconds: float = 2.0,
    sample_rate: int = 48000,
) -> None:
    """Write a mono 16-bit WAV of pure silence."""
    n = int(seconds * sample_rate)
    pcm = np.zeros(n, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
```

### Step 4: Run, see pass

```
pytest tests/unit/test_fixture_helpers.py -v
```
Expected: 4 PASSED.

### Step 5: Commit

```
git add tests/_fixtures.py tests/unit/test_fixture_helpers.py
git commit -m "test(fixtures): synthetic media helpers (video, motion video, tone, silence)"
```

---

## Task 2: `CheckResult` dataclass + module skeleton

**Files:**
- Create: `src/platinum/utils/validate.py`
- Test: `tests/unit/test_validate.py`

### Step 1: Write the failing test

Create `tests/unit/test_validate.py`:

```python
"""Tests for utils/validate.py quality-gate primitives."""

from __future__ import annotations

import pytest

from platinum.utils.validate import CheckResult


def test_check_result_is_frozen_and_carries_fields() -> None:
    r = CheckResult(passed=True, metric=0.5, threshold=0.4, reason="ok")
    assert r.passed and r.metric == 0.5 and r.threshold == 0.4 and r.reason == "ok"
    with pytest.raises(Exception):
        r.passed = False  # type: ignore[misc]
```

### Step 2: Run, see fail

```
pytest tests/unit/test_validate.py -v
```
Expected: ImportError -- `validate` module not present.

### Step 3: Implement

Create `src/platinum/utils/validate.py`:

```python
"""Quality-gate primitives.

Five sync `check_*` functions returning a uniform `CheckResult`. Used by
Sessions 6/8/13 to reject AI-generated assets before human review.

`passed=True` always means the asset cleared the gate.
"""

from __future__ import annotations

from dataclasses import dataclass


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
```

### Step 4: Run, see pass

```
pytest tests/unit/test_validate.py -v
```
Expected: 1 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/validate.py tests/unit/test_validate.py
git commit -m "feat(validate): CheckResult dataclass scaffold"
```

---

## Task 3: `check_duration_match`

**Files:**
- Modify: `src/platinum/utils/validate.py`
- Modify: `tests/unit/test_validate.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_validate.py`:

```python
from pathlib import Path

from tests._fixtures import make_test_audio


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
    assert "tolerance" in r.reason.lower() or "exceeds" in r.reason.lower()


def test_check_duration_match_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_duration_match
    with pytest.raises(FileNotFoundError):
        check_duration_match(tmp_path / "nope.wav", target_seconds=1.0, tolerance_seconds=0.1)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_validate.py -v
```
Expected: 4 import-time failures or AttributeError -- `check_duration_match` not defined.

### Step 3: Implement

Append to `src/platinum/utils/validate.py`:

```python
import json
import shutil
import subprocess
from pathlib import Path


def _ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe. Raises FileNotFoundError if file or ffprobe missing."""
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
            f"passed: {measured:.2f}s within {tolerance_seconds:.2f}s of target {target_seconds:.2f}s"
        )
    else:
        reason = (
            f"failed: {measured:.2f}s deviates {delta:.2f}s from target "
            f"{target_seconds:.2f}s (tolerance {tolerance_seconds:.2f}s)"
        )
    return CheckResult(passed=passed, metric=measured, threshold=target_seconds, reason=reason)
```

### Step 4: Run, see pass

```
pytest tests/unit/test_validate.py -v
```
Expected: 5 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/validate.py tests/unit/test_validate.py
git commit -m "feat(validate): check_duration_match via ffprobe"
```

---

## Task 4: `check_audio_levels`

**Files:**
- Modify: `src/platinum/utils/validate.py`
- Modify: `tests/unit/test_validate.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_validate.py`:

```python
from tests._fixtures import make_silent_audio


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
```

### Step 2: Run, see fail

```
pytest tests/unit/test_validate.py -k audio_levels -v
```
Expected: AttributeError -- `check_audio_levels` / `_measure_lufs` not defined.

### Step 3: Implement

Append to `src/platinum/utils/validate.py`:

```python
import math
import re


def _measure_lufs(path: Path) -> float:
    """Return integrated LUFS via ffmpeg loudnorm. -inf for silent audio."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not on PATH.")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(path),
            "-af", "loudnorm=print_format=json",
            "-f", "null", "-",
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
            f"failed: {measured:.1f} LUFS deviates {delta:.1f} dB from target {target_lufs:.1f} "
            f"(tolerance {tolerance_db:.1f} dB)"
        )
    return CheckResult(passed=passed, metric=measured, threshold=target_lufs, reason=reason)
```

### Step 4: Run, see pass

```
pytest tests/unit/test_validate.py -v
```
Expected: 9 PASSED total.

### Step 5: Commit

```
git add src/platinum/utils/validate.py tests/unit/test_validate.py
git commit -m "feat(validate): check_audio_levels via ffmpeg loudnorm"
```

---

## Task 5: `check_black_frames`

**Files:**
- Modify: `src/platinum/utils/validate.py`
- Modify: `tests/unit/test_validate.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_validate.py`:

```python
from tests._fixtures import make_test_video, make_test_video_with_motion


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
```

### Step 2: Run, see fail

```
pytest tests/unit/test_validate.py -k black_frames -v
```
Expected: AttributeError.

### Step 3: Implement

Append to `src/platinum/utils/validate.py`:

```python
import cv2  # noqa: E402  -- placed near related imports below
import numpy as np  # noqa: E402


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
```

(Move `import cv2` and `import numpy as np` to the top-of-file import block during refactor; both needed for upcoming tasks.)

### Step 4: Run, see pass

```
pytest tests/unit/test_validate.py -v
```
Expected: 13 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/validate.py tests/unit/test_validate.py
git commit -m "feat(validate): check_black_frames via cv2 frame-mean threshold"
```

---

## Task 6: `check_motion`

**Files:**
- Modify: `src/platinum/utils/validate.py`
- Modify: `tests/unit/test_validate.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_validate.py`:

```python
def test_check_motion_static_video_fails(tmp_path: Path) -> None:
    from platinum.utils.validate import check_motion
    video = tmp_path / "static.mp4"
    make_test_video(video, n_frames=30, fps=24, color=(80, 80, 80), size=(64, 64))
    r = check_motion(video, min_flow_magnitude=0.5)
    assert not r.passed


def test_check_motion_moving_video_passes(tmp_path: Path) -> None:
    from platinum.utils.validate import check_motion
    video = tmp_path / "moving.mp4"
    make_test_video_with_motion(video, n_frames=30, fps=24, size=(64, 64))
    r = check_motion(video, min_flow_magnitude=0.5)
    assert r.passed
    assert r.metric > 0.5


def test_check_motion_corrupt_video_fails(tmp_path: Path) -> None:
    from platinum.utils.validate import check_motion
    video = tmp_path / "corrupt.mp4"
    video.write_bytes(b"not a real mp4")
    r = check_motion(video, min_flow_magnitude=0.5)
    assert not r.passed
    assert "cannot read" in r.reason.lower() or "frames" in r.reason.lower()


def test_check_motion_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_motion
    with pytest.raises(FileNotFoundError):
        check_motion(tmp_path / "nope.mp4", min_flow_magnitude=0.5)
```

### Step 2: Run, see fail

```
pytest tests/unit/test_validate.py -k motion -v
```
Expected: AttributeError.

### Step 3: Implement

Append to `src/platinum/utils/validate.py`:

```python
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
                    flow = cv2.calcOpticalFlowFarneback(
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
        reason = f"passed: mean flow magnitude {mean_mag:.2f} >= min {min_flow_magnitude:.2f}"
    else:
        reason = f"failed: mean flow magnitude {mean_mag:.2f} below min {min_flow_magnitude:.2f}"
    return CheckResult(
        passed=passed, metric=mean_mag, threshold=min_flow_magnitude, reason=reason
    )
```

### Step 4: Run, see pass

```
pytest tests/unit/test_validate.py -v
```
Expected: 17 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/validate.py tests/unit/test_validate.py
git commit -m "feat(validate): check_motion via cv2 Farneback optical flow"
```

---

## Task 7: `check_hand_anomalies` (with `mp_hands_factory` injection)

**Files:**
- Modify: `src/platinum/utils/validate.py`
- Modify: `tests/unit/test_validate.py`

### Step 1: Write the failing tests

Append to `tests/unit/test_validate.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image


def _write_solid_image(path: Path, size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (10, 10, 10)) -> None:
    Image.new("RGB", size, color=color).save(path)


def _make_fake_hands_factory(landmarks_per_hand: list[int] | None):
    """Return a factory that produces a Hands-stub returning configured results."""
    def factory():
        instance = MagicMock()
        if landmarks_per_hand is None:
            result = SimpleNamespace(multi_hand_landmarks=None)
        else:
            hands = []
            for n in landmarks_per_hand:
                lm = SimpleNamespace(landmark=[SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(n)])
                hands.append(lm)
            result = SimpleNamespace(multi_hand_landmarks=hands)
        instance.process.return_value = result
        instance.close.return_value = None
        return instance
    return factory


def test_check_hand_anomalies_no_hands_passes(tmp_path: Path) -> None:
    from platinum.utils.validate import check_hand_anomalies
    img = tmp_path / "blank.png"
    _write_solid_image(img)
    r = check_hand_anomalies(img, mp_hands_factory=_make_fake_hands_factory(None))
    assert r.passed
    assert "no hands" in r.reason.lower()


def test_check_hand_anomalies_valid_hands_passes(tmp_path: Path) -> None:
    from platinum.utils.validate import check_hand_anomalies
    img = tmp_path / "valid.png"
    _write_solid_image(img)
    r = check_hand_anomalies(img, mp_hands_factory=_make_fake_hands_factory([21, 21]))
    assert r.passed


def test_check_hand_anomalies_extra_landmarks_fails(tmp_path: Path) -> None:
    from platinum.utils.validate import check_hand_anomalies
    img = tmp_path / "anom.png"
    _write_solid_image(img)
    r = check_hand_anomalies(img, mp_hands_factory=_make_fake_hands_factory([21, 22]))
    assert not r.passed
    assert "22" in r.reason or "anomaly" in r.reason.lower()


def test_check_hand_anomalies_missing_file(tmp_path: Path) -> None:
    from platinum.utils.validate import check_hand_anomalies
    with pytest.raises(FileNotFoundError):
        check_hand_anomalies(tmp_path / "nope.png", mp_hands_factory=_make_fake_hands_factory(None))
```

### Step 2: Run, see fail

```
pytest tests/unit/test_validate.py -k hand -v
```
Expected: AttributeError.

### Step 3: Implement

Append to `src/platinum/utils/validate.py`:

```python
from collections.abc import Callable
from typing import Any

EXPECTED_LANDMARKS_PER_HAND = 21


def _default_mp_hands_factory() -> Any:
    """Late-bound mediapipe Hands factory; resolved at call time, not import time."""
    import mediapipe as mp  # noqa: PLC0415  -- intentional lazy import

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
    hands = factory()
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
            reason=f"failed: hand anomaly (landmark counts {counts}, expected {EXPECTED_LANDMARKS_PER_HAND})",
        )
    return CheckResult(
        passed=True,
        metric=float(EXPECTED_LANDMARKS_PER_HAND),
        threshold=float(EXPECTED_LANDMARKS_PER_HAND),
        reason=f"passed: {len(counts)} hand(s) with valid landmark counts",
    )
```

### Step 4: Run, see pass

```
pytest tests/unit/test_validate.py -v
```
Expected: 21 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/validate.py tests/unit/test_validate.py
git commit -m "feat(validate): check_hand_anomalies via mediapipe (late-bound factory)"
```

---

## Task 8: `AestheticScorer` Protocol + `FakeAestheticScorer`

**Files:**
- Create: `src/platinum/utils/aesthetics.py`
- Test: `tests/unit/test_aesthetics.py`

### Step 1: Write the failing tests

Create `tests/unit/test_aesthetics.py`:

```python
"""Tests for utils/aesthetics.py."""

from __future__ import annotations

import inspect
from pathlib import Path

from platinum.utils.aesthetics import AestheticScorer, FakeAestheticScorer


async def test_fake_scorer_returns_fixed_score(tmp_path: Path) -> None:
    scorer = FakeAestheticScorer(fixed_score=7.25)
    img = tmp_path / "x.png"
    img.write_bytes(b"")
    score = await scorer.score(img)
    assert score == 7.25


async def test_fake_scorer_satisfies_protocol() -> None:
    scorer = FakeAestheticScorer(fixed_score=5.0)
    assert isinstance(scorer, AestheticScorer)


def test_fake_scorer_score_is_awaitable() -> None:
    scorer = FakeAestheticScorer(fixed_score=5.0)
    coro = scorer.score(Path("ignored"))
    assert inspect.iscoroutine(coro)
    coro.close()
```

### Step 2: Run, see fail

```
pytest tests/unit/test_aesthetics.py -v
```
Expected: ImportError -- `aesthetics` module not present.

### Step 3: Implement

Create `src/platinum/utils/aesthetics.py`:

```python
"""Aesthetic scoring interface.

Real LAION-Aesthetics v2 implementation lives on the cloud GPU and lands in
Session 6. This module ships only the Protocol and a deterministic fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class AestheticScorer(Protocol):
    """Score an image for cinematic quality on a 0.0-10.0 scale."""

    async def score(self, image_path: Path) -> float: ...


@dataclass(frozen=True, slots=True)
class FakeAestheticScorer:
    """Deterministic scorer for tests."""

    fixed_score: float

    async def score(self, image_path: Path) -> float:
        return self.fixed_score
```

### Step 4: Run, see pass

```
pytest tests/unit/test_aesthetics.py -v
```
Expected: 3 PASSED.

### Step 5: Commit

```
git add src/platinum/utils/aesthetics.py tests/unit/test_aesthetics.py
git commit -m "feat(aesthetics): AestheticScorer Protocol + FakeAestheticScorer (Session 6 fills real impl)"
```

---

## Task 9: `quality_gates` block in `atmospheric_horror.yaml` + integration test

**Files:**
- Modify: `config/tracks/atmospheric_horror.yaml`
- Create: `tests/integration/test_quality_gates_config.py`

### Step 1: Write the failing test

Create `tests/integration/test_quality_gates_config.py`:

```python
"""Integration tests: quality_gates block round-trips through Config.track()."""

from __future__ import annotations

from pathlib import Path

import pytest

from platinum.config import Config

EXPECTED_KEYS = {
    "aesthetic_min_score",
    "black_frame_max_ratio",
    "luminance_threshold",
    "motion_min_flow_magnitude",
    "audio_target_lufs",
    "audio_lufs_tolerance_db",
    "duration_tolerance_seconds",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_atmospheric_horror_yaml_loads_quality_gates() -> None:
    cfg = Config(root=REPO_ROOT)
    track = cfg.track("atmospheric_horror")
    assert "quality_gates" in track, "atmospheric_horror.yaml is missing the quality_gates block"
    gates = track["quality_gates"]
    missing = EXPECTED_KEYS - set(gates.keys())
    assert not missing, f"atmospheric_horror.quality_gates missing keys: {missing}"
    for key in EXPECTED_KEYS:
        assert isinstance(gates[key], (int, float)), f"{key} must be numeric"
    assert 0.0 <= gates["aesthetic_min_score"] <= 10.0
    assert 0.0 <= gates["black_frame_max_ratio"] <= 1.0
    assert gates["audio_target_lufs"] < 0
```

### Step 2: Run, see fail

```
pytest tests/integration/test_quality_gates_config.py -v
```
Expected: AssertionError -- `quality_gates` block not yet in YAML.

### Step 3: Add the block

Edit `config/tracks/atmospheric_horror.yaml`. Find the existing `track:` block and insert before the closing of `track:` (after `publish:` is fine):

```yaml
  # Quality gates -- automated thresholds enforced before human review (Session 5).
  quality_gates:
    aesthetic_min_score: 6.0          # LAION-Aesthetics v2 (0-10); horror runs ~0.5 lower than bright tracks
    black_frame_max_ratio: 0.02       # max share of near-black frames; horror has intentional fade-to-blacks
    luminance_threshold: 8.0          # 0-255 mean below = "black frame"
    motion_min_flow_magnitude: 0.5    # Farneback dense flow, pixels/frame; below = still image
    audio_target_lufs: -16.0          # YouTube narration standard
    audio_lufs_tolerance_db: 1.5      # +/- around target
    duration_tolerance_seconds: 0.5   # narration audio vs. scene target
```

### Step 4: Run, see pass

```
pytest tests/integration/test_quality_gates_config.py -v
```
Expected: 1 PASSED.

### Step 5: Commit

```
git add config/tracks/atmospheric_horror.yaml tests/integration/test_quality_gates_config.py
git commit -m "feat(config): quality_gates block in atmospheric_horror track"
```

---

## Task 10: `quality_gates` block in remaining four track YAMLs + sweep test

**Files:**
- Modify: `config/tracks/folktales_world_myths.yaml`
- Modify: `config/tracks/childrens_fables.yaml`
- Modify: `config/tracks/scifi_concept.yaml`
- Modify: `config/tracks/slice_of_life.yaml`
- Modify: `tests/integration/test_quality_gates_config.py`

### Step 1: Write the failing test

Append to `tests/integration/test_quality_gates_config.py`:

```python
ALL_TRACKS = [
    "atmospheric_horror",
    "folktales_world_myths",
    "childrens_fables",
    "scifi_concept",
    "slice_of_life",
]


@pytest.mark.parametrize("track_id", ALL_TRACKS)
def test_all_tracks_have_quality_gates(track_id: str) -> None:
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    assert "quality_gates" in track, f"{track_id} missing quality_gates"
    gates = track["quality_gates"]
    missing = EXPECTED_KEYS - set(gates.keys())
    assert not missing, f"{track_id}.quality_gates missing keys: {missing}"
```

### Step 2: Run, see fail

```
pytest tests/integration/test_quality_gates_config.py -v
```
Expected: 4 failures (one per non-horror track).

### Step 3: Insert the block in each track YAML

For each of the four files, add the `quality_gates:` block under `track:`. Track-tuned values:

**`folktales_world_myths.yaml`** -- moderate bar:
```yaml
  quality_gates:
    aesthetic_min_score: 6.5
    black_frame_max_ratio: 0.01
    luminance_threshold: 12.0
    motion_min_flow_magnitude: 0.6
    audio_target_lufs: -16.0
    audio_lufs_tolerance_db: 1.5
    duration_tolerance_seconds: 0.5
```

**`childrens_fables.yaml`** -- bright tolerates a higher aesthetic floor and very little blackness:
```yaml
  quality_gates:
    aesthetic_min_score: 6.5
    black_frame_max_ratio: 0.005
    luminance_threshold: 16.0
    motion_min_flow_magnitude: 0.8
    audio_target_lufs: -16.0
    audio_lufs_tolerance_db: 1.5
    duration_tolerance_seconds: 0.5
```

**`scifi_concept.yaml`** -- can be dark (space) but wants strong motion:
```yaml
  quality_gates:
    aesthetic_min_score: 6.5
    black_frame_max_ratio: 0.03
    luminance_threshold: 8.0
    motion_min_flow_magnitude: 0.7
    audio_target_lufs: -16.0
    audio_lufs_tolerance_db: 1.5
    duration_tolerance_seconds: 0.5
```

**`slice_of_life.yaml`** -- naturalistic, moderate everything:
```yaml
  quality_gates:
    aesthetic_min_score: 6.0
    black_frame_max_ratio: 0.01
    luminance_threshold: 14.0
    motion_min_flow_magnitude: 0.5
    audio_target_lufs: -16.0
    audio_lufs_tolerance_db: 1.5
    duration_tolerance_seconds: 0.5
```

### Step 4: Run, see pass

```
pytest tests/integration/test_quality_gates_config.py -v
```
Expected: 6 PASSED (1 horror + 5 parametrized).

### Step 5: Commit

```
git add config/tracks/*.yaml tests/integration/test_quality_gates_config.py
git commit -m "feat(config): quality_gates block in all five track YAMLs"
```

---

## Task 11: ffmpeg-real round-trip integration test for audio levels

**Files:**
- Modify: `tests/integration/test_quality_gates_config.py`

### Step 1: Write the failing test

Append:

```python
def test_check_audio_levels_round_trips_with_real_ffmpeg(tmp_path: Path) -> None:
    """Generate a tone, measure with loudnorm, verify check_audio_levels matches."""
    from platinum.utils.validate import _measure_lufs, check_audio_levels
    from tests._fixtures import make_test_audio

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=3.0, freq_hz=1000.0, amplitude=0.25)
    measured = _measure_lufs(audio)
    assert measured > -50.0  # not silent
    assert measured < 0.0    # not clipping above 0 dB FS
    r = check_audio_levels(audio, target_lufs=measured, tolerance_db=0.5)
    assert r.passed
    assert "passed" in r.reason
```

### Step 2: Run, see pass (already implemented in Task 4)

```
pytest tests/integration/test_quality_gates_config.py -v
```
Expected: 7 PASSED. This test uses real ffmpeg, no mocks. It's the only "real binary" integration test we ship in Session 5.

### Step 3: Commit

```
git add tests/integration/test_quality_gates_config.py
git commit -m "test(integration): real-ffmpeg LUFS round-trip for check_audio_levels"
```

---

## Task 12: Quality sweep + smoke run + memory update

**Files:**
- Verify: full repo
- Modify: `C:/Users/claws/.claude/projects/C--Users-claws-OneDrive-Desktop-platinum/memory/project_platinum.md` -- append a Session 5 review section

### Step 1: Run the full quality bar

```
pytest -q
```
Expected: ~180 tests, 0 failures, 0 skips, run time under 20s.

```
ruff check src tests
```
Expected: All checks passed.

```
mypy src
```
Expected: Success: no issues found (or accept previously-deferred items not introduced this session).

### Step 2: Smoke run

Generate a test asset and run the primitives end-to-end from a one-off shell:

```
python - <<'PY'
import tempfile
from pathlib import Path
from tests._fixtures import make_test_audio, make_test_video_with_motion
from platinum.utils.validate import (
    check_audio_levels, check_black_frames, check_duration_match, check_motion,
)

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    audio = d / "tone.wav"
    video = d / "moving.mp4"
    make_test_audio(audio, seconds=3.0, freq_hz=440.0, amplitude=0.25)
    make_test_video_with_motion(video, n_frames=24, fps=24, size=(64, 64))
    print(check_duration_match(audio, target_seconds=3.0, tolerance_seconds=0.1).reason)
    print(check_audio_levels(audio, target_lufs=-12.0, tolerance_db=4.0).reason)
    print(check_black_frames(video, max_black_ratio=0.05).reason)
    print(check_motion(video, min_flow_magnitude=0.5).reason)
PY
```

Expected: four `passed: ...` lines.

### Step 3: Append Session 5 review to project memory

Open `C:/Users/claws/.claude/projects/C--Users-claws-OneDrive-Desktop-platinum/memory/project_platinum.md` and append a `Session 5 review` block summarising:
- Files added (`utils/aesthetics.py`, `utils/validate.py`, `tests/unit/test_aesthetics.py`, `tests/unit/test_validate.py`, `tests/unit/test_fixture_helpers.py`, `tests/integration/test_quality_gates_config.py`).
- Files modified (5 track YAMLs, `tests/_fixtures.py`).
- Test count delta (~25 new tests).
- Decisions locked in (LAION deferred, per-track YAML thresholds, mock-mediapipe, uniform CheckResult).
- Lessons reinforced (late binding for testability, Protocol + Fake = clean injection, no Stage wiring at leaf level).
- Status update: Session 5 complete, Session 6 next.

### Step 4: Final commit

```
git add -A
git commit -m "$(cat <<'EOF'
docs: Session 5 complete -- aesthetic scoring + validation utilities

Five sync check_* primitives shipped (CheckResult, black frames, motion,
hand anomalies, audio LUFS, duration match). AestheticScorer Protocol +
FakeAestheticScorer; real impl deferred to Session 6 on vast.ai GPU.
quality_gates block added to all five track YAMLs.

Tests: ~180 total (~25 new), 0 fail, 0 skip. ruff + mypy clean.
EOF
)"
```

### Step 5: Push not required

`git push` is the user's call; the plan does not push automatically.

---

## Done criteria

- [ ] Tasks 1-12 each have a green commit landed on `main` (or the active worktree branch).
- [ ] `pytest -q` shows ~180 tests passing, 0 failures, 0 skips.
- [ ] `ruff check src tests` clean.
- [ ] `mypy src` clean (no new errors introduced this session).
- [ ] `config/tracks/*.yaml` all carry a `quality_gates:` block.
- [ ] `src/platinum/utils/aesthetics.py` and `src/platinum/utils/validate.py` exist with the documented surface.
- [ ] Project memory updated with Session 5 review.
- [ ] Session 6 (cloud provisioning + keyframe generator) can compose the leaf utilities with zero refactor of Session 5 code.
