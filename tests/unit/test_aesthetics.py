"""Tests for utils/aesthetics.py."""

from __future__ import annotations

import inspect
from pathlib import Path

import httpx
import pytest

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


async def test_mapped_fake_scorer_returns_mapped_value(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import MappedFakeScorer
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    img_a.write_bytes(b"")
    img_b.write_bytes(b"")
    scorer = MappedFakeScorer(scores_by_path={img_a: 7.5, img_b: 3.0}, default=0.0)
    assert await scorer.score(img_a) == 7.5
    assert await scorer.score(img_b) == 3.0


async def test_mapped_fake_scorer_returns_default_for_unmapped(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import MappedFakeScorer
    unknown = tmp_path / "unknown.png"
    unknown.write_bytes(b"")
    scorer = MappedFakeScorer(scores_by_path={}, default=4.2)
    assert await scorer.score(unknown) == 4.2


async def test_remote_scorer_happy_path(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["content"] = request.content
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"score": 6.42})

    transport = httpx.MockTransport(handler)
    scorer = RemoteAestheticScorer(host="http://test:8189", transport=transport)
    image = tmp_path / "candidate_0.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    try:
        score = await scorer.score(image)
    finally:
        await scorer.aclose()

    assert score == 6.42
    assert captured["url"] == "http://test:8189/score"
    assert captured["method"] == "POST"
    assert b"candidate_0.png" in captured["content"]  # multipart filename
    assert str(captured["content_type"]).startswith("multipart/form-data")


def test_remote_scorer_missing_host_raises() -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    with pytest.raises(ValueError, match="PLATINUM_AESTHETICS_HOST"):
        RemoteAestheticScorer(host="")
