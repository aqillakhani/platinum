"""Unit tests for utils/claude.py."""

from __future__ import annotations

import pytest

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


def test_recorded_call_round_trips_through_dict() -> None:
    from platinum.utils.claude import RecordedCall
    rc = RecordedCall(
        request={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]},
        response={"id": "msg_1", "content": [{"type": "tool_use", "input": {"x": 1}}]},
    )
    d = rc.to_dict()
    assert d["request"]["model"] == "claude-opus-4-7"
    rc2 = RecordedCall.from_dict(d)
    assert rc2 == rc


def test_recorder_protocol_accepts_synthetic_recorder() -> None:
    """Anything with an awaitable __call__ that takes (request) -> response satisfies Recorder."""
    from platinum.utils.claude import Recorder

    class FakeRec:
        async def __call__(self, request: dict) -> dict:
            return {"id": "fake", "content": []}

    rec: Recorder = FakeRec()  # would fail static-typing if not satisfying the protocol
    assert callable(rec)


def test_resolve_api_key_reads_env(monkeypatch) -> None:
    from platinum.utils.claude import resolve_api_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    assert resolve_api_key() == "sk-test-123"


def test_resolve_api_key_missing_raises_clear_error(monkeypatch) -> None:
    from platinum.utils.claude import resolve_api_key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        resolve_api_key()


@pytest.mark.asyncio
async def test_call_uses_recorder_response_and_returns_claude_result(tmp_path) -> None:
    from platinum.utils.claude import call

    async def synthetic_recorder(request: dict) -> dict:
        return {
            "id": "msg_synthetic",
            "content": [
                {"type": "tool_use", "name": request["tool_choice"]["name"],
                 "input": {"title": "T", "synopsis": "S"}}
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100, "output_tokens": 50,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            },
        }

    db_path = tmp_path / "test.db"
    from platinum.models.db import create_all
    create_all(db_path)

    result = await call(
        model="claude-opus-4-7",
        system=[{"type": "text", "text": "You are an editor."}],
        messages=[{"role": "user", "content": "Hello"}],
        tool={"name": "submit_story", "input_schema": {"type": "object"}},
        story_id="story_test_001",
        stage="story_adapter",
        db_path=db_path,
        recorder=synthetic_recorder,
    )
    assert result.tool_input == {"title": "T", "synopsis": "S"}
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cost_usd > 0


@pytest.mark.asyncio
async def test_call_raises_protocol_error_when_no_tool_use(tmp_path) -> None:
    from platinum.utils.claude import ClaudeProtocolError, call

    async def text_only(request: dict) -> dict:
        return {
            "id": "msg_text",
            "content": [{"type": "text", "text": "Sorry, no tool use today."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 5,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    db_path = tmp_path / "p.db"
    from platinum.models.db import create_all
    create_all(db_path)

    with pytest.raises(ClaudeProtocolError, match="tool_use"):
        await call(
            model="claude-opus-4-7",
            system=[{"type": "text", "text": ""}],
            messages=[{"role": "user", "content": "x"}],
            tool={"name": "t", "input_schema": {}},
            story_id=None, stage="story_adapter",
            db_path=db_path, recorder=text_only,
        )


@pytest.mark.asyncio
async def test_call_request_carries_tool_choice_forced(tmp_path) -> None:
    from platinum.utils.claude import call

    captured = {}

    async def capture(request: dict) -> dict:
        captured.update(request)
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "t", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    db_path = tmp_path / "p.db"
    from platinum.models.db import create_all
    create_all(db_path)

    await call(
        model="claude-opus-4-7",
        system=[{"type": "text", "text": "S"}],
        messages=[{"role": "user", "content": "M"}],
        tool={"name": "t", "input_schema": {"type": "object"}},
        story_id=None, stage="visual_prompts",
        db_path=db_path, recorder=capture,
    )
    assert captured["tool_choice"] == {"type": "tool", "name": "t"}
    assert captured["tools"] == [{"name": "t", "input_schema": {"type": "object"}}]
    assert captured["model"] == "claude-opus-4-7"
    assert captured["system"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_call_writes_api_usage_row(tmp_path) -> None:
    from platinum.models.db import ApiUsageRow, create_all, sync_session
    from platinum.utils.claude import call

    async def synth(request):
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "t", "input": {"a": 1}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1000, "output_tokens": 500,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    from datetime import datetime

    from platinum.models.db import StoryRow

    db_path = tmp_path / "p.db"
    create_all(db_path)
    with sync_session(db_path) as s:
        s.add(StoryRow(id="story_x", track="atmospheric_horror", status="pending",
                        created_at=datetime.now(), updated_at=datetime.now()))

    await call(
        model="claude-opus-4-7",
        system=[{"type": "text", "text": "S"}],
        messages=[{"role": "user", "content": "M"}],
        tool={"name": "t", "input_schema": {}},
        story_id="story_x", stage="story_adapter",
        db_path=db_path, recorder=synth,
    )

    with sync_session(db_path) as s:
        rows = s.query(ApiUsageRow).all()
        assert len(rows) == 1
        assert rows[0].model == "claude-opus-4-7"
        assert rows[0].input_tokens == 1000
        assert rows[0].output_tokens == 500
        assert rows[0].provider == "anthropic"
        assert rows[0].story_id == "story_x"
        assert rows[0].cost_usd > 0


@pytest.mark.asyncio
async def test_default_live_recorder_calls_async_anthropic(monkeypatch, tmp_path) -> None:
    """When recorder=None and a fake AsyncAnthropic is injected via
    client_factory, call() should hit it."""
    from platinum.models.db import create_all
    from platinum.utils.claude import call

    captured = {}

    class FakeMessage:
        def model_dump(self):
            return {
                "id": "msg_live",
                "content": [{"type": "tool_use", "name": "t", "input": {"ok": True}}],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 10, "output_tokens": 5,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                },
            }

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeMessage()

    class FakeAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    db_path = tmp_path / "p.db"
    create_all(db_path)

    result = await call(
        model="claude-opus-4-7",
        system=[{"type": "text", "text": "S"}],
        messages=[{"role": "user", "content": "M"}],
        tool={"name": "t", "input_schema": {"type": "object"}},
        story_id=None, stage="story_adapter",
        db_path=db_path,
        client_factory=lambda: FakeAnthropic(),
    )
    assert result.tool_input == {"ok": True}
    assert captured["model"] == "claude-opus-4-7"
    assert captured["tool_choice"] == {"type": "tool", "name": "t"}


@pytest.mark.asyncio
async def test_call_retries_on_rate_limit(monkeypatch, tmp_path) -> None:
    """Retry decorator should kick in for RateLimitError."""
    import anthropic

    from platinum.models.db import create_all
    from platinum.utils.claude import call

    attempt = {"n": 0}

    class FakeMessages:
        async def create(self, **kwargs):
            attempt["n"] += 1
            if attempt["n"] < 3:
                raise anthropic.RateLimitError(
                    message="rate limited",
                    response=type("R", (), {"status_code": 429, "headers": {}, "request": None})(),
                    body=None,
                )
            return type("Msg", (), {
                "model_dump": lambda self: {
                    "id": "ok", "content": [{"type": "tool_use", "name": "t", "input": {}}],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                }
            })()

    class FakeAnthropic:
        def __init__(self, **_): self.messages = FakeMessages()

    async def noop_sleep(_seconds):
        pass

    monkeypatch.setattr("platinum.utils.retry.asyncio.sleep", noop_sleep)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    db_path = tmp_path / "p.db"
    create_all(db_path)

    result = await call(
        model="claude-opus-4-7",
        system=[{"type": "text", "text": "S"}],
        messages=[{"role": "user", "content": "M"}],
        tool={"name": "t", "input_schema": {}},
        story_id=None, stage="story_adapter",
        db_path=db_path,
        client_factory=lambda: FakeAnthropic(),
    )
    assert attempt["n"] == 3
    assert result.tool_input == {}


@pytest.mark.asyncio
async def test_missing_api_key_raises_before_request(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from platinum.models.db import create_all
    from platinum.utils.claude import call

    db_path = tmp_path / "p.db"
    create_all(db_path)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await call(
            model="claude-opus-4-7",
            system=[{"type": "text", "text": ""}],
            messages=[{"role": "user", "content": ""}],
            tool={"name": "t", "input_schema": {}},
            story_id=None, stage="story_adapter",
            db_path=db_path,
        )
