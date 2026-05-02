"""Anthropic SDK wrapper: tool-use, prompt caching, retry, cost tracking, fixture recording.

Only file in platinum that imports `anthropic`. Single integration point.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import anthropic

from platinum.models.db import ApiUsageRow, sync_session
from platinum.utils.retry import retry

logger = logging.getLogger(__name__)


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
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
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


def resolve_api_key() -> str:
    """Return the Anthropic API key from env, or raise a clear error.

    Loaded from `secrets/.env` by `platinum.config.Config` at process start.
    Raising here -- before any HTTP call -- gives a much clearer signal than
    Anthropic's generic 401.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in environment or secrets/.env. "
            "Get a key at console.anthropic.com and add it to secrets/.env."
        )
    return key


class ClaudeProtocolError(RuntimeError):
    """Anthropic returned a response shape we don't understand."""


def _attach_cache_control(system: list[dict]) -> list[dict]:
    """Return a copy of `system` with cache_control on the final block.

    Anthropic accepts up to 4 cache breakpoints; we put one at the end of
    the system blocks so all preceding text (including track style guides)
    is cached together.
    """
    if not system:
        return []
    out: list[dict] = [dict(b) for b in system]
    out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out


def _extract_tool_use(response: dict) -> tuple[dict, str]:
    """Return (tool_input, fallback_text) from a Claude response or raise."""
    content = response.get("content", [])
    text_chunks: list[str] = []
    for block in content:
        if block.get("type") == "tool_use":
            return dict(block.get("input", {})), ""
        if block.get("type") == "text":
            text_chunks.append(block.get("text", ""))
    raise ClaudeProtocolError(
        f"Expected tool_use in response.content, got: "
        f"types={[b.get('type') for b in content]}, text={'; '.join(text_chunks)[:200]}"
    )


def _write_usage_row(
    *,
    db_path: Path,
    story_id: str | None,
    usage: ClaudeUsage,
    when: datetime,
) -> None:
    """Best-effort ApiUsageRow write. Never fails the call."""
    try:
        with sync_session(db_path) as session:
            session.add(
                ApiUsageRow(
                    story_id=story_id,
                    provider="anthropic",
                    model=usage.model,
                    input_tokens=usage.input_tokens
                    + usage.cache_creation_input_tokens
                    + usage.cache_read_input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=usage.cost_usd,
                    ts=when,
                )
            )
    except Exception as exc:
        logger.warning("Failed to record ApiUsageRow (cost tracking only): %s", exc)


async def call(
    *,
    model: str,
    system: list[dict],
    messages: list[dict],
    tool: dict,
    story_id: str | None,
    stage: str,
    db_path: Path,
    recorder: Recorder | None = None,
    client_factory: Callable[[], anthropic.AsyncAnthropic] | None = None,
    max_tokens: int = 8192,
) -> ClaudeResult:
    """One Claude call in tool-use forced mode.

    With `recorder=None` (production), talks to the real SDK [Task 8].
    With a recorder injected (tests / fixture-replay), the recorder owns
    the request -> response transformation.

    ``max_tokens`` defaults to 8192 (current visual_prompts ceiling). The
    story_bible stage (S8.B) overrides to 16000 because the prototype
    validated 8000 truncated continuity blocks; bumping the default would
    raise spend on every other call unnecessarily.
    """
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _attach_cache_control(system),
        "messages": messages,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
    }

    if recorder is not None:
        response = await recorder(request)
    else:
        response = await _live_call(request, client_factory=client_factory)

    tool_input, fallback = _extract_tool_use(response)
    raw_usage = response.get("usage", {})
    usage = ClaudeUsage(
        model=model,
        input_tokens=int(raw_usage.get("input_tokens", 0)),
        output_tokens=int(raw_usage.get("output_tokens", 0)),
        cache_creation_input_tokens=int(raw_usage.get("cache_creation_input_tokens", 0)),
        cache_read_input_tokens=int(raw_usage.get("cache_read_input_tokens", 0)),
        cost_usd=calculate_cost_usd(
            model=model,
            input_tokens=int(raw_usage.get("input_tokens", 0)),
            output_tokens=int(raw_usage.get("output_tokens", 0)),
            cache_creation_input_tokens=int(raw_usage.get("cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(raw_usage.get("cache_read_input_tokens", 0)),
        ),
    )
    _write_usage_row(db_path=db_path, story_id=story_id, usage=usage, when=datetime.now())
    return ClaudeResult(tool_input=tool_input, text=fallback, usage=usage, raw=response)


@retry(
    max_retries=4,
    base_delay=1.0,
    max_delay=16.0,
    exceptions=(anthropic.RateLimitError, anthropic.APIStatusError),
)
async def _live_call(
    request: dict,
    *,
    client_factory: Callable[[], anthropic.AsyncAnthropic] | None,
) -> dict:
    """Hit AsyncAnthropic; raise on auth/4xx, retry on 429/5xx."""
    api_key = resolve_api_key()
    factory = client_factory or (lambda: anthropic.AsyncAnthropic(api_key=api_key))
    client = factory()
    msg = await client.messages.create(**request)
    return msg.model_dump()
