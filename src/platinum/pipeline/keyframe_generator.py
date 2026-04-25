"""Keyframe generator pipeline -- per-scene Flux candidate generation + selection.

For each Scene:
  1. Generate N candidates (default 3) via ComfyClient at deterministic seeds.
  2. Score each via AestheticScorer; check_hand_anomalies for anatomy gate.
  3. Select the highest-scoring candidate that passes both gates;
     fall back to candidate 0 if none qualify.

Pure functions (`generate_for_scene`, `generate`) take all dependencies as
args; the impure `KeyframeGeneratorStage` pulls dependencies from `ctx`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KeyframeReport:
    """Per-scene generation report. Persisted onto Scene fields by the Stage."""

    scene_index: int
    candidates: list[Path]
    scores: list[float]
    anatomy_passed: list[bool]
    selected_index: int
    selected_via_fallback: bool


class KeyframeGenerationError(RuntimeError):
    """Raised when ALL candidates for a scene threw during generation.

    Carries the per-candidate exception list so the Stage can record a
    useful error string in the StageRun log.
    """

    def __init__(self, *, scene_index: int, exceptions: Sequence[BaseException]) -> None:
        self.scene_index = scene_index
        self.exceptions = list(exceptions)
        joined = "; ".join(f"{type(e).__name__}: {e}" for e in self.exceptions)
        super().__init__(f"all candidates failed for scene_index={scene_index}: {joined}")
