"""Fixture recorder/replayer for offline LLM tests.

Replay mode: read JSON from disk, return its `response` field.
Record mode: call the live backend, save (request, response) to disk.

Path scheme: tests/fixtures/anthropic/<stage>/<test_name>__<attempt>.json
"""

from __future__ import annotations

import json
import math
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

LiveCall = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class FixtureMissingError(FileNotFoundError):
    """Raised in replay mode when no fixture matches the request."""


class FixtureRecorder:
    """Per-call recorder. One instance per (stage, test, attempt).

    In replay mode, returns the saved response. In record mode, invokes
    `live` and writes the (request, response) pair to disk on the fly.
    """

    def __init__(
        self,
        *,
        path: Path,
        mode: Literal["replay", "record"] = "replay",
        live: LiveCall | None = None,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.live = live

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "replay":
            if not self.path.exists():
                raise FixtureMissingError(
                    f"No fixture at {self.path}. Re-run with PLATINUM_RECORD_FIXTURES=1 "
                    "to capture against the live API."
                )
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data["response"]

        # record mode
        if self.live is None:
            raise RuntimeError("FixtureRecorder(mode='record') requires a `live` callable.")
        response = await self.live(request)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"request": request, "response": response}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return response


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


def make_synthetic_png(
    path: Path,
    *,
    kind: Literal["grey", "checkerboard", "gradient"] = "grey",
    value: int = 128,
    size: tuple[int, int] = (64, 64),
    block: int = 8,
) -> None:
    """Write a small synthetic PNG to `path`.

    kind="grey": solid RGB at (value, value, value).
    kind="checkerboard": black/white blocks of `block` pixels.
    kind="gradient": smooth horizontal gradient from black to white.
    """
    w, h = size
    if kind == "grey":
        Image.new("RGB", size, color=(value, value, value)).save(path)
        return
    if kind == "checkerboard":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, block):
            for x in range(0, w, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    arr[y : y + block, x : x + block] = 255
        Image.fromarray(arr, "RGB").save(path)
        return
    if kind == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            v = int(255 * x / max(w - 1, 1))
            arr[:, x] = (v, v, v)
        Image.fromarray(arr, "RGB").save(path)
        return
    raise ValueError(f"unknown kind: {kind!r}")


def make_fake_hands_factory(landmarks_per_hand: list[int] | None) -> Callable[[], Any]:
    """Return a factory that yields a Hands-stub returning configured results.

    Used by tests that exercise check_hand_anomalies (and the keyframe
    generator's anatomy gate) without mediapipe. Pass None to simulate
    "no hands detected"; pass [21, 21] for two valid hands; pass [21, 22]
    to simulate an anomaly.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    def factory() -> Any:
        instance = MagicMock()
        if landmarks_per_hand is None:
            result = SimpleNamespace(multi_hand_landmarks=None)
        else:
            hands = []
            for n in landmarks_per_hand:
                lm = SimpleNamespace(
                    landmark=[SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(n)]
                )
                hands.append(lm)
            result = SimpleNamespace(multi_hand_landmarks=hands)
        instance.process.return_value = result
        instance.close.return_value = None
        return instance

    return factory
