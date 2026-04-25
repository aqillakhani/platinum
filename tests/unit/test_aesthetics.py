"""Tests for utils/aesthetics.py."""

from __future__ import annotations

import inspect
from pathlib import Path

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


def test_remote_aesthetic_scorer_init_raises_with_session_pointer() -> None:
    from platinum.utils.aesthetics import RemoteAestheticScorer
    with pytest.raises(NotImplementedError) as exc:
        RemoteAestheticScorer(host="example.com", ssh_user="root", ssh_key_path=None)
    assert "Session 6.1" in str(exc.value)
