"""Aesthetic scoring interface.

Three layers:
- AestheticScorer Protocol -- the contract.
- FakeAestheticScorer / MappedFakeScorer -- deterministic fakes for tests.
- RemoteAestheticScorer -- httpx-based client for the FastAPI score_server
  running on the vast.ai box (see scripts/score_server/server.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx


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
    """LAION-Aesthetics v2 scorer. Calls a FastAPI score_server running on the
    vast.ai box (see scripts/score_server/server.py).

    Lifecycle: instantiated once per Stage.run; closed via aclose() at end of stage.
    Mirrors HttpComfyClient -- transport seam for httpx.MockTransport in unit tests.
    """

    def __init__(
        self,
        *,
        host: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not host:
            raise ValueError(
                "RemoteAestheticScorer requires a host "
                "(set PLATINUM_AESTHETICS_HOST in .env or aesthetics.host in settings.yaml)"
            )
        self._host = host.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=timeout,
            transport=transport,
        )

    async def score(self, image_path: Path) -> float:
        with image_path.open("rb") as fh:
            files = {"image": (image_path.name, fh, "image/png")}
            resp = await self._client.post("/score", files=files)
        resp.raise_for_status()
        body = resp.json()
        score = float(body["score"])
        if not math.isfinite(score):
            raise ValueError(f"score_server returned non-finite score: {body!r}")
        return score

    async def aclose(self) -> None:
        await self._client.aclose()
