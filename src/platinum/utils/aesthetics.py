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
    """Score an image for cinematic quality on a 0.0-10.0 scale.

    Also exposes a CLIP image-text cosine similarity in [-1, 1] for use as
    the content-fidelity gate (S7.1.A3): rejects candidates whose render
    drifts from the visual_prompt's semantic intent.
    """

    async def score(self, image_path: Path) -> float: ...

    async def clip_similarity(self, image_path: Path, text: str) -> float: ...


@dataclass(frozen=True, slots=True)
class FakeAestheticScorer:
    """Deterministic scorer for tests."""

    fixed_score: float
    fixed_clip_similarity: float = 0.5

    async def score(self, image_path: Path) -> float:
        return self.fixed_score

    async def clip_similarity(self, image_path: Path, text: str) -> float:
        return self.fixed_clip_similarity


@dataclass(frozen=True, slots=True)
class MappedFakeScorer:
    """Test scorer: returns score per image_path; falls back to default for unmapped paths.

    Used by tests that need different scores for different candidates within the same
    test (e.g., to drive selection logic). The plain FakeAestheticScorer with a single
    fixed_score cannot do this.

    Mirrors the per-path mapping for clip_similarity so tests can drive the
    A3.3 content-fidelity gate by candidate path.
    """

    scores_by_path: dict[Path, float] = field(default_factory=dict)
    default: float = 0.0
    clip_similarities_by_path: dict[Path, float] = field(default_factory=dict)
    clip_similarity_default: float = 0.5

    async def score(self, image_path: Path) -> float:
        return self.scores_by_path.get(image_path, self.default)

    async def clip_similarity(self, image_path: Path, text: str) -> float:
        return self.clip_similarities_by_path.get(
            image_path, self.clip_similarity_default
        )


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

    async def clip_similarity(self, image_path: Path, text: str) -> float:
        with image_path.open("rb") as fh:
            files = {"image": (image_path.name, fh, "image/png")}
            data = {"text": text}
            resp = await self._client.post("/clip-sim", files=files, data=data)
        resp.raise_for_status()
        body = resp.json()
        sim = float(body["similarity"])
        if not math.isfinite(sim):
            raise ValueError(f"score_server returned non-finite similarity: {body!r}")
        return sim

    async def aclose(self) -> None:
        await self._client.aclose()
