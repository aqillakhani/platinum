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


def _content_check_fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "anthropic"
        / "content_check"
        / "test_records_and_replays__1.json"
    )


async def test_claude_content_checker_records_and_replays(tmp_path: Path) -> None:
    """S7.1.A4.3: ClaudeContentChecker hits Anthropic vision and round-trips
    through the project FixtureRecorder.

    Replay mode (default): reads the saved fixture and asserts the parsed
    ContentCheckResult shape.

    Record mode (PLATINUM_RECORD_FIXTURES=1): hits the live Anthropic API,
    saves the response to tests/fixtures/anthropic/content_check/. ~$0.005
    per record. Use a synthetic checkerboard PNG so we never commit any PII.
    """
    import os

    from platinum.models.db import create_all
    from platinum.utils.claude import _live_call
    from platinum.utils.content_check import ClaudeContentChecker
    from tests._fixtures import FixtureRecorder, make_synthetic_png

    fixture_path = _content_check_fixture_path()
    record = os.environ.get("PLATINUM_RECORD_FIXTURES") == "1"

    db_path = tmp_path / "p.db"
    create_all(db_path)

    img_path = tmp_path / "candidate.png"
    make_synthetic_png(img_path, kind="checkerboard", size=(256, 256), block=32)

    if record:
        # Trigger secrets/.env load so ANTHROPIC_API_KEY reaches resolve_api_key().
        from platinum.config import Config
        Config()

        async def live(request: dict) -> dict:
            return await _live_call(request, client_factory=None)
        recorder = FixtureRecorder(path=fixture_path, mode="record", live=live)
    else:
        recorder = FixtureRecorder(path=fixture_path, mode="replay")

    checker = ClaudeContentChecker(recorder=recorder, db_path=db_path)
    result = await checker.check(
        prompt="a wide checkerboard pattern of black and white squares",
        image_path=img_path,
    )

    assert isinstance(result.score, int)
    assert 1 <= result.score <= 10
    assert isinstance(result.missing, list)
    assert isinstance(result.rationale, str)
    assert result.raw_response  # non-empty JSON


async def test_claude_content_checker_satisfies_protocol() -> None:
    from platinum.utils.content_check import ClaudeContentChecker, ContentChecker

    checker = ClaudeContentChecker()
    assert isinstance(checker, ContentChecker)


async def test_claude_content_checker_parses_tool_input(tmp_path: Path) -> None:
    """ClaudeContentChecker.check maps the submit_content_check tool_input
    into ContentCheckResult fields verbatim."""
    import json as _json

    from platinum.models.db import create_all
    from platinum.utils.content_check import ClaudeContentChecker
    from tests._fixtures import make_synthetic_png

    db_path = tmp_path / "p.db"
    create_all(db_path)

    img = tmp_path / "x.png"
    make_synthetic_png(img, kind="grey", value=128)

    async def synth(request: dict) -> dict:
        return {
            "id": "msg_synth",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_content_check",
                    "input": {
                        "score": 7,
                        "missing": ["fog", "chains"],
                        "rationale": "atmospheric but specific elements absent",
                    },
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    checker = ClaudeContentChecker(recorder=synth, db_path=db_path)
    result = await checker.check(prompt="a chained man in fog", image_path=img)
    assert result.score == 7
    assert result.missing == ["fog", "chains"]
    assert "atmospheric" in result.rationale
    assert _json.loads(result.raw_response)["score"] == 7


async def test_claude_content_checker_haiku_pricing_resolves(tmp_path: Path) -> None:
    """Haiku 4.5 cost calculation must not raise -- wire-check for
    _PRICING_USD_PER_MTOK entry added alongside ClaudeContentChecker."""
    from platinum.models.db import create_all
    from platinum.utils.content_check import ClaudeContentChecker
    from tests._fixtures import make_synthetic_png

    db_path = tmp_path / "p.db"
    create_all(db_path)
    img = tmp_path / "x.png"
    make_synthetic_png(img, kind="grey", value=64)

    async def synth(_: dict) -> dict:
        return {
            "id": "msg",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_content_check",
                    "input": {"score": 9, "missing": [], "rationale": "good"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    checker = ClaudeContentChecker(recorder=synth, db_path=db_path)
    result = await checker.check(prompt="x", image_path=img)
    assert result.score == 9
