"""Unit tests for utils/claude.py."""

from __future__ import annotations

from platinum.utils.claude import calculate_cost_usd


def test_calculate_cost_opus_input_only() -> None:
    # 1M input tokens at $15/M = $15.00
    cost = calculate_cost_usd(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    assert cost == 15.0


def test_calculate_cost_opus_output() -> None:
    # 1M output tokens at $75/M = $75.00
    cost = calculate_cost_usd(
        model="claude-opus-4-7",
        input_tokens=0,
        output_tokens=1_000_000,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    assert cost == 75.0


def test_calculate_cost_cache_read_discounted() -> None:
    # 1M cache-read tokens at $1.50/M (10% of $15) = $1.50
    cost = calculate_cost_usd(
        model="claude-opus-4-7",
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=0,
    )
    assert cost == 1.5


def test_calculate_cost_cache_creation_premium() -> None:
    # 1M cache-creation tokens at $18.75/M (125% of $15)
    cost = calculate_cost_usd(
        model="claude-opus-4-7",
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=1_000_000,
    )
    assert cost == 18.75


def test_calculate_cost_unknown_model_raises() -> None:
    import pytest
    with pytest.raises(KeyError, match="claude-haiku-99"):
        calculate_cost_usd(
            model="claude-haiku-99",
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )


def test_claude_usage_is_frozen_dataclass() -> None:
    from platinum.utils.claude import ClaudeUsage
    u = ClaudeUsage(
        model="claude-opus-4-7",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        cost_usd=0.0,
    )
    import dataclasses
    assert dataclasses.is_dataclass(u)
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.input_tokens = 200  # type: ignore[misc]


def test_claude_result_holds_tool_input_and_usage() -> None:
    from platinum.utils.claude import ClaudeResult, ClaudeUsage
    r = ClaudeResult(
        tool_input={"hello": "world"},
        text="",
        usage=ClaudeUsage(
            model="claude-opus-4-7",
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
            cost_usd=0.001,
        ),
        raw={"id": "msg_123"},
    )
    assert r.tool_input == {"hello": "world"}
    assert r.usage.cost_usd == 0.001
