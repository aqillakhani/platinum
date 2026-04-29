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
