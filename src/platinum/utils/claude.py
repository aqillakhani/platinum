"""Anthropic SDK wrapper: tool-use, prompt caching, retry, cost tracking, fixture recording.

Only file in platinum that imports `anthropic`. Single integration point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ClaudeUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class ClaudeResult:
    tool_input: dict[str, Any]
    text: str
    usage: ClaudeUsage
    raw: dict[str, Any]


@dataclass(frozen=True)
class RecordedCall:
    """Captured (request, response) pair for fixture replay."""

    request: dict[str, Any]
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"request": dict(self.request), "response": dict(self.response)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecordedCall:
        return cls(request=dict(d["request"]), response=dict(d["response"]))


@runtime_checkable
class Recorder(Protocol):
    """Protocol for fixture record/replay or any synthetic stand-in.

    Tests inject a Recorder; production calls leave it None and `claude.call`
    talks to the real SDK.
    """

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]: ...


# Pricing per million tokens, in USD: (input, output).
# Cache reads bill at 10% of input rate; cache creation at 125%.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
}


def calculate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
) -> float:
    """Compute USD cost for one Anthropic API call.

    Cache reads are billed at 10% of the input rate; cache creation at 125%.
    """
    if model not in _PRICING_USD_PER_MTOK:
        raise KeyError(f"No pricing entry for model: {model!r}")
    in_rate, out_rate = _PRICING_USD_PER_MTOK[model]
    cost = (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_read_input_tokens * in_rate * 0.10
        + cache_creation_input_tokens * in_rate * 1.25
    ) / 1_000_000
    return round(cost, 6)
