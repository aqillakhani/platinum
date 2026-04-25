"""Tests for utils/aesthetics.py."""

from __future__ import annotations

import inspect
from pathlib import Path

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
