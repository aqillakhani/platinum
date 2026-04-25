"""Aesthetic scoring interface.

Real LAION-Aesthetics v2 implementation lives on the cloud GPU and lands in
Session 6. This module ships only the Protocol and a deterministic fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class MappedFakeScorer:
    """Test scorer: returns score per image_path; falls back to default for unmapped paths.

    Used by tests that need different scores for different candidates within the same
    test (e.g., to drive selection logic). The plain FakeAestheticScorer with a single
    fixed_score cannot do this.
    """

    scores_by_path: dict[Path, float] = field(default_factory=dict)
    default: float = 0.0

    async def score(self, image_path: Path) -> float:
        return self.scores_by_path.get(image_path, self.default)


class RemoteAestheticScorer:
    """LAION-Aesthetics v2 scorer that runs on the vast.ai box via SSH+script.

    Implementation deferred to Session 6.1. Construction is wired (so the
    production code path in KeyframeGeneratorStage can reference this class
    by name) but raises NotImplementedError so tests / dry-runs fail loudly
    with a clear pointer rather than silently returning zeros.
    """

    def __init__(
        self,
        *,
        host: str,
        ssh_user: str = "root",
        ssh_key_path: Path | None = None,
    ) -> None:
        raise NotImplementedError(
            "Session 6.1: implement SSH+script LAION scorer. "
            "Until then, inject FakeAestheticScorer / MappedFakeScorer in tests."
        )

    async def score(self, image_path: Path) -> float:
        raise NotImplementedError
