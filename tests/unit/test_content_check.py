"""Tests for utils/content_check.py -- Claude vision content gate (S7.1.A4)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


def test_content_check_result_is_frozen_dataclass() -> None:
    """ContentCheckResult is a frozen dataclass with the documented fields."""
    from platinum.utils.content_check import ContentCheckResult

    r = ContentCheckResult(
        score=8,
        missing=["fog"],
        rationale="image shows the candle but lacks fog described in prompt",
        raw_response='{"score": 8}',
    )
    assert r.score == 8
    assert r.missing == ["fog"]
    assert "fog" in r.rationale
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        r.score = 9  # type: ignore[misc]


def test_content_check_result_required_fields() -> None:
    """All four fields (score, missing, rationale, raw_response) must be present."""
    from platinum.utils.content_check import ContentCheckResult

    r = ContentCheckResult(score=10, missing=[], rationale="", raw_response="{}")
    assert r.score == 10
    assert r.missing == []
    assert r.rationale == ""
    assert r.raw_response == "{}"


async def test_content_checker_protocol_can_be_satisfied() -> None:
    """The ContentChecker Protocol is runtime_checkable and accepts a duck."""
    from platinum.utils.content_check import ContentChecker, ContentCheckResult

    class _DuckChecker:
        async def check(
            self, *, prompt: str, image_path: Path
        ) -> ContentCheckResult:
            return ContentCheckResult(
                score=10, missing=[], rationale="ok", raw_response="{}"
            )

    duck = _DuckChecker()
    assert isinstance(duck, ContentChecker)
    result = await duck.check(prompt="anything", image_path=Path("/tmp/x.png"))
    assert result.score == 10


async def test_fake_content_checker_returns_mapped_score(tmp_path: Path) -> None:
    """S7.1.A4.2: FakeContentChecker maps image_path -> score for per-candidate
    control."""
    from platinum.utils.content_check import FakeContentChecker

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"")
    b.write_bytes(b"")
    fake = FakeContentChecker(scores={a: 8, b: 3})
    result_a = await fake.check(prompt="anything", image_path=a)
    result_b = await fake.check(prompt="anything", image_path=b)
    assert result_a.score == 8
    assert result_b.score == 3


async def test_fake_content_checker_returns_default_for_unknown(tmp_path: Path) -> None:
    """Unknown paths fall back to the default_score (default 8 = pass)."""
    from platinum.utils.content_check import FakeContentChecker

    unknown = tmp_path / "unknown.png"
    unknown.write_bytes(b"")
    fake = FakeContentChecker(scores={}, default_score=7)
    result = await fake.check(prompt="anything", image_path=unknown)
    assert result.score == 7


async def test_fake_content_checker_default_score_is_eight(tmp_path: Path) -> None:
    """default_score defaults to 8 = neutral pass for tests that don't care."""
    from platinum.utils.content_check import FakeContentChecker

    img = tmp_path / "x.png"
    img.write_bytes(b"")
    fake = FakeContentChecker()
    result = await fake.check(prompt="anything", image_path=img)
    assert result.score == 8


async def test_fake_content_checker_missing_elements_per_path(tmp_path: Path) -> None:
    """missing_by_path maps image_path -> list of missing elements."""
    from platinum.utils.content_check import FakeContentChecker

    img = tmp_path / "img.png"
    img.write_bytes(b"")
    fake = FakeContentChecker(
        scores={img: 5},
        missing_by_path={img: ["chains", "fog"]},
    )
    result = await fake.check(prompt="...", image_path=img)
    assert result.missing == ["chains", "fog"]


async def test_fake_content_checker_satisfies_protocol(tmp_path: Path) -> None:
    """FakeContentChecker is a structural ContentChecker."""
    from platinum.utils.content_check import ContentChecker, FakeContentChecker

    fake = FakeContentChecker()
    assert isinstance(fake, ContentChecker)


async def test_fake_content_checker_records_call_count(tmp_path: Path) -> None:
    """call_count exposes how many times check() was invoked -- tests that
    need to verify the gate skipped a candidate (e.g. when content_gate=off)
    rely on this."""
    from platinum.utils.content_check import FakeContentChecker

    img = tmp_path / "x.png"
    img.write_bytes(b"")
    fake = FakeContentChecker()
    assert fake.call_count == 0
    await fake.check(prompt="x", image_path=img)
    await fake.check(prompt="x", image_path=img)
    assert fake.call_count == 2
