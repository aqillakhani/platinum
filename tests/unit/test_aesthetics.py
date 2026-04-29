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


async def test_remote_scorer_non_200_raises(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    scorer = RemoteAestheticScorer(
        host="http://test:8189", transport=httpx.MockTransport(handler)
    )
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await scorer.score(image)
    finally:
        await scorer.aclose()


async def test_remote_scorer_aclose_idempotent() -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    scorer = RemoteAestheticScorer(host="http://test:8189")
    await scorer.aclose()
    await scorer.aclose()  # second call must not raise


async def test_remote_scorer_nan_raises(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    async def handler(request: httpx.Request) -> httpx.Response:
        # Send literal NaN in the JSON body. Python's json.loads is lenient
        # about NaN by default; our scorer must catch it via math.isfinite.
        return httpx.Response(
            200,
            content=b'{"score": NaN}',
            headers={"content-type": "application/json"},
        )

    scorer = RemoteAestheticScorer(
        host="http://test:8189", transport=httpx.MockTransport(handler)
    )
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        with pytest.raises(ValueError, match="non-finite"):
            await scorer.score(image)
    finally:
        await scorer.aclose()


async def test_fake_scorer_clip_similarity_returns_fixed(tmp_path: Path) -> None:
    """S7.1.A3.2: FakeAestheticScorer.clip_similarity returns fixed_clip_similarity."""
    scorer = FakeAestheticScorer(fixed_score=7.0, fixed_clip_similarity=0.35)
    img = tmp_path / "x.png"
    img.write_bytes(b"")
    sim = await scorer.clip_similarity(img, "a candle in dark hallway")
    assert sim == 0.35


async def test_fake_scorer_clip_similarity_default_is_neutral(tmp_path: Path) -> None:
    """fixed_clip_similarity defaults to 0.5 -- a neutral pass for tests that
    don't care about the gate."""
    scorer = FakeAestheticScorer(fixed_score=7.0)
    img = tmp_path / "x.png"
    img.write_bytes(b"")
    assert await scorer.clip_similarity(img, "irrelevant") == 0.5


async def test_mapped_fake_scorer_clip_similarity_returns_mapped(tmp_path: Path) -> None:
    """S7.1.A3.2: MappedFakeScorer maps path -> similarity for per-candidate control."""
    from platinum.utils.aesthetics import MappedFakeScorer

    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    img_a.write_bytes(b"")
    img_b.write_bytes(b"")
    scorer = MappedFakeScorer(
        scores_by_path={},
        default=0.0,
        clip_similarities_by_path={img_a: 0.40, img_b: 0.05},
        clip_similarity_default=0.20,
    )
    assert await scorer.clip_similarity(img_a, "x") == 0.40
    assert await scorer.clip_similarity(img_b, "x") == 0.05
    unknown = tmp_path / "unknown.png"
    unknown.write_bytes(b"")
    assert await scorer.clip_similarity(unknown, "x") == 0.20


async def test_remote_scorer_clip_similarity_happy_path(tmp_path: Path) -> None:
    """S7.1.A3.2: RemoteAestheticScorer.clip_similarity hits /clip-sim with
    multipart image + text form field."""
    from platinum.utils.aesthetics import RemoteAestheticScorer

    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["content"] = request.content
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"similarity": 0.31})

    scorer = RemoteAestheticScorer(
        host="http://test:8189", transport=httpx.MockTransport(handler)
    )
    image = tmp_path / "candidate_2.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    try:
        sim = await scorer.clip_similarity(image, "a candle in dark hallway")
    finally:
        await scorer.aclose()

    assert sim == 0.31
    assert captured["url"] == "http://test:8189/clip-sim"
    assert captured["method"] == "POST"
    assert b"candidate_2.png" in captured["content"]
    assert b"a candle in dark hallway" in captured["content"]  # text form field
    assert str(captured["content_type"]).startswith("multipart/form-data")


async def test_remote_scorer_clip_similarity_non_200_raises(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    scorer = RemoteAestheticScorer(
        host="http://test:8189", transport=httpx.MockTransport(handler)
    )
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await scorer.clip_similarity(image, "anything")
    finally:
        await scorer.aclose()


async def test_remote_scorer_clip_similarity_nan_raises(tmp_path: Path) -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"similarity": NaN}',
            headers={"content-type": "application/json"},
        )

    scorer = RemoteAestheticScorer(
        host="http://test:8189", transport=httpx.MockTransport(handler)
    )
    image = tmp_path / "x.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        with pytest.raises(ValueError, match="non-finite"):
            await scorer.clip_similarity(image, "x")
    finally:
        await scorer.aclose()
