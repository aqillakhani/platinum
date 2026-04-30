"""Video generator pipeline -- per-scene Wan 2.2 I2V (S8 Phase A).

For each Scene with a populated keyframe_path:
  1. Upload keyframe to ComfyUI.
  2. Submit Wan 2.2 I2V workflow with keyframe + visual_prompt as conditioning.
  3. Run quality gates (duration_match, black_frames, motion).
  4. On content failure: retry ONCE with new seed.
  5. On 2nd content fail or any infrastructure failure: raise VideoGenerationError.

Pure functions (`generate_video_for_scene`, `generate_video`) take all
dependencies as args; the impure `VideoGeneratorStage` pulls dependencies
from `ctx`. Per-scene atomic save with resume via Scene.video_path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoReport:
    """Per-scene generation report. Persisted onto Scene fields by the Stage."""

    scene_index: int
    success: bool
    mp4_path: Path | None
    duration_seconds: float
    gates_passed: dict[str, bool]
    retry_used: int


class VideoGenerationError(RuntimeError):
    """Raised when a scene's video generation cannot complete.

    Carries a structured reason and a `retryable` flag distinguishing
    infrastructure failures (network, OOM, missing weights -- not retryable
    by re-seeding) from content failures (gates fail twice -- in principle
    retryable by the user with fresh seeds).
    """

    def __init__(
        self,
        *,
        scene_index: int,
        reason: str,
        retryable: bool = False,
    ) -> None:
        self.scene_index = scene_index
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"scene_index={scene_index}: {reason} (retryable={retryable})")


def _seed_for_scene(scene_index: int, *, retry: int = 0) -> int:
    """Deterministic seed for a scene's video generation.

    seed = scene_index * 1000 + retry. With retry in {0, 1} only two seeds
    are ever used per scene (initial + one retry).
    """
    return scene_index * 1000 + retry
