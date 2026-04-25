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
