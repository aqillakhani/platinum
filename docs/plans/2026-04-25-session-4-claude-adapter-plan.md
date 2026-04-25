# Session 4 — Claude integration + story adapter (implementation plan)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Build `python -m platinum adapt` that takes a curator-approved Story and produces, via three Claude (Opus 4.7) calls, a polished narration script + scene list + per-scene visual prompts. Persist into `story.json` and SQLite.

**Architecture:** One Anthropic SDK wrapper (`utils/claude.py`) doing tool-use, prompt caching, retry, cost tracking, and fixture recording. Three pipeline modules with three `Stage` subclasses (`story_adapter`, `scene_breakdown`, `visual_prompts`). Tests run offline against recorded fixtures; live API only when (re)recording.

**Tech Stack:** Python 3.11, `anthropic>=0.40` (async client), `jinja2`, `sqlalchemy`, `pytest`+`pytest-asyncio` (auto mode already enabled).

**Design doc:** `docs/plans/2026-04-25-session-4-claude-adapter-design.md` — read first if any task feels under-specified.

---

## Pre-flight context (read before Task 1)

### Existing code you will integrate with

- `src/platinum/models/story.py` — `Story`, `Source`, `Adapted` (with `arc` dict), `Scene`, `StageRun`, `StageStatus`. **Do not modify.**
- `src/platinum/models/db.py` — has `ApiUsageRow` table with `story_id, provider, model, input_tokens, output_tokens, cost_usd, ts`. Use `sync_session(db_path)` ctx manager for writes.
- `src/platinum/pipeline/stage.py` — `Stage` ABC, async `run`, must declare `name` class attribute.
- `src/platinum/pipeline/orchestrator.py` — already awaits `stage.run`; uses `CANONICAL_STAGE_NAMES` list which already contains `"story_adapter"`, `"scene_breakdown"`, `"visual_prompts"` at positions 3/4/5.
- `src/platinum/pipeline/context.py` — `PipelineContext.db_path`, `story_path(story)`.
- `src/platinum/utils/retry.py` — async-only `retry(...)` decorator. Use it on `claude.call` directly.
- `src/platinum/config.py` — `Config.track(id)` returns track dict; `Config.prompts_dir` -> `config/prompts/`.

### Existing test fixtures (in `tests/conftest.py`)

- `tmp_project` — minimal project layout under `tmp_path`.
- `config` — `Config(root=tmp_project)`.
- `context` — `PipelineContext`.
- `source`, `story` — pre-built `Source` and `Story` objects.

### Conventions established in earlier sessions

- ASCII only in any string that flows to a Windows console (Rich panels, Typer help, prompts, error messages). No `>>`, no smart quotes, no em dashes in user-facing strings.
- Pure-core / impure-shell: pipeline modules export pure-ish functions with injectable dependencies; the Stage subclasses do the I/O wiring.
- Late binding for testability: do not capture `subprocess.run` / `client_factory` in default args; use `runner=None`, resolve at call time.
- `pyproject.toml` already has `anthropic>=0.40`. No new deps needed.
- `pyproject.toml` already has `[tool.pytest.ini_options] asyncio_mode = "auto"`. Plain `def test_x():` for sync tests, `async def test_x():` for async.
- Atomic Story writes via `Story.save` already exist. Don't reinvent.

### Cost-tracking pricing facts

- `claude-opus-4-7`: $15/M input tokens, $75/M output tokens.
- Cache reads bill at 10% of input rate ($1.50/M).
- Cache writes bill at 125% of input rate ($18.75/M) — only on the call that creates the cache.

---

## Task 1: Pricing table + cost calculation

**Files:**
- Create: `src/platinum/utils/claude.py`
- Test: `tests/unit/test_claude_util.py`

**Step 1: Write failing tests.**

Append to a fresh file `tests/unit/test_claude_util.py`:

```python
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
```

**Step 2: Run tests to verify failure.**

```
pytest tests/unit/test_claude_util.py -q
```
Expected: 5 errors / `ModuleNotFoundError: No module named 'platinum.utils.claude'`.

**Step 3: Implement.**

Create `src/platinum/utils/claude.py` with just enough to pass:

```python
"""Anthropic SDK wrapper: tool-use, prompt caching, retry, cost tracking, fixture recording.

Only file in platinum that imports `anthropic`. Single integration point.
"""

from __future__ import annotations


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
```

**Step 4: Run tests to verify pass.**

```
pytest tests/unit/test_claude_util.py -q
```
Expected: `5 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/claude.py tests/unit/test_claude_util.py
git commit -m "feat(claude): pricing table + calculate_cost_usd

Opus 4.7 base/cache/creation rates per Anthropic public pricing.
Tests cover input, output, cache-read discount, cache-creation premium,
unknown-model error."
```

---

## Task 2: `ClaudeUsage` and `ClaudeResult` dataclasses

**Files:**
- Modify: `src/platinum/utils/claude.py` (add dataclasses)
- Test: `tests/unit/test_claude_util.py` (extend)

**Step 1: Write failing test.** Append to `tests/unit/test_claude_util.py`:

```python
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
```

**Step 2: Run.** Expected: `ImportError`.

**Step 3: Implement.** Add to `src/platinum/utils/claude.py` near the top:

```python
from dataclasses import dataclass
from typing import Any


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
```

**Step 4: Run.** Expected: `7 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/claude.py tests/unit/test_claude_util.py
git commit -m "feat(claude): ClaudeUsage and ClaudeResult dataclasses

Frozen dataclasses for SDK call output. tool_input is the parsed dict from
the tool-use response; raw is the full SDK response (used by the recorder
to capture fixtures)."
```

---

## Task 3: Recorder protocol + `RecordedCall` dataclass

**Files:**
- Modify: `src/platinum/utils/claude.py`
- Test: `tests/unit/test_claude_util.py`

**Step 1: Write failing test.** Append:

```python
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
```

**Step 2: Run.** Expected: `ImportError`.

**Step 3: Implement.** Add to `src/platinum/utils/claude.py`:

```python
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RecordedCall:
    """Captured (request, response) pair for fixture replay."""

    request: dict[str, Any]
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"request": dict(self.request), "response": dict(self.response)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecordedCall":
        return cls(request=dict(d["request"]), response=dict(d["response"]))


@runtime_checkable
class Recorder(Protocol):
    """Protocol for fixture record/replay or any synthetic stand-in.

    Tests inject a Recorder; production calls leave it None and `claude.call`
    talks to the real SDK.
    """

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]: ...
```

**Step 4: Run.** Expected: `9 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/claude.py tests/unit/test_claude_util.py
git commit -m "feat(claude): Recorder protocol + RecordedCall dataclass

Dependency-injection seam for tests. Production passes recorder=None; tests
pass a FixtureRecorder. RecordedCall round-trips via to_dict/from_dict so
fixtures persist as plain JSON."
```

---

## Task 4: `FixtureRecorder` (replay-only path)

**Files:**
- Create: `tests/_fixtures.py`
- Test: `tests/unit/test_recorder.py`

**Step 1: Write failing tests.** Create `tests/unit/test_recorder.py`:

```python
"""Unit tests for the fixture recorder used in offline LLM tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_replay_returns_saved_response(tmp_path: Path) -> None:
    from tests._fixtures import FixtureRecorder

    fixture = {
        "request": {"model": "claude-opus-4-7", "messages": []},
        "response": {"id": "msg_replay", "content": [{"type": "tool_use", "input": {"ok": True}}]},
    }
    fp = tmp_path / "fixture.json"
    fp.write_text(json.dumps(fixture), encoding="utf-8")

    rec = FixtureRecorder(path=fp, mode="replay")
    response = await rec(fixture["request"])
    assert response["id"] == "msg_replay"


@pytest.mark.asyncio
async def test_replay_missing_fixture_raises_with_record_hint(tmp_path: Path) -> None:
    from tests._fixtures import FixtureMissingError, FixtureRecorder

    rec = FixtureRecorder(path=tmp_path / "missing.json", mode="replay")
    with pytest.raises(FixtureMissingError, match="PLATINUM_RECORD_FIXTURES=1"):
        await rec({"model": "claude-opus-4-7"})
```

**Step 2: Run.** Expected: 2 errors / module not found.

**Step 3: Implement.** Create `tests/_fixtures.py`:

```python
"""Fixture recorder/replayer for offline LLM tests.

Replay mode: read JSON from disk, return its `response` field.
Record mode: call the live backend, save (request, response) to disk.

Path scheme: tests/fixtures/anthropic/<stage>/<test_name>__<attempt>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

LiveCall = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class FixtureMissingError(FileNotFoundError):
    """Raised in replay mode when no fixture matches the request."""


class FixtureRecorder:
    """Per-call recorder. One instance per (stage, test, attempt).

    In replay mode, returns the saved response. In record mode, invokes
    `live` and writes the (request, response) pair to disk on the fly.
    """

    def __init__(
        self,
        *,
        path: Path,
        mode: Literal["replay", "record"] = "replay",
        live: LiveCall | None = None,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.live = live

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "replay":
            if not self.path.exists():
                raise FixtureMissingError(
                    f"No fixture at {self.path}. Re-run with PLATINUM_RECORD_FIXTURES=1 "
                    "to capture against the live API."
                )
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data["response"]

        # record mode
        if self.live is None:
            raise RuntimeError("FixtureRecorder(mode='record') requires a `live` callable.")
        response = await self.live(request)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"request": request, "response": response}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return response
```

**Step 4: Run.** Expected: `2 passed`.

**Step 5: Commit.**

```bash
git add tests/_fixtures.py tests/unit/test_recorder.py
git commit -m "feat(tests): FixtureRecorder replay-mode

Replay mode reads (request, response) JSON from disk and returns response;
missing fixture raises FixtureMissingError with a clear hint pointing to
PLATINUM_RECORD_FIXTURES=1. Record mode is wired but exercised in next task."
```

---

## Task 5: `FixtureRecorder` record-mode

**Files:**
- Modify: `tests/unit/test_recorder.py`

**Step 1: Write failing test.** Append:

```python
@pytest.mark.asyncio
async def test_record_writes_fixture_file(tmp_path: Path) -> None:
    from tests._fixtures import FixtureRecorder

    captured = []

    async def fake_live(req: dict) -> dict:
        captured.append(req)
        return {"id": "msg_recorded", "content": [{"type": "tool_use", "input": {"a": 1}}]}

    fp = tmp_path / "anthropic" / "story_adapter" / "test__1.json"
    rec = FixtureRecorder(path=fp, mode="record", live=fake_live)
    response = await rec({"model": "claude-opus-4-7", "n": 42})

    assert response["id"] == "msg_recorded"
    assert fp.exists()
    saved = json.loads(fp.read_text(encoding="utf-8"))
    assert saved["request"]["n"] == 42
    assert saved["response"]["id"] == "msg_recorded"
    assert captured == [{"model": "claude-opus-4-7", "n": 42}]


@pytest.mark.asyncio
async def test_record_mode_without_live_raises(tmp_path: Path) -> None:
    from tests._fixtures import FixtureRecorder

    rec = FixtureRecorder(path=tmp_path / "x.json", mode="record", live=None)
    with pytest.raises(RuntimeError, match="record.*requires.*live"):
        await rec({})
```

**Step 2: Run.** Expected: pass (already implemented in Task 4).

**Step 3 / 4 / 5:** No new code needed; commit the additional tests:

```bash
git add tests/unit/test_recorder.py
git commit -m "test(recorder): record-mode integration cases

Cover the record-mode happy path (fake live -> file written; nested dirs
auto-created) and the missing-live error path."
```

---

## Task 6: API-key resolver helper

**Files:**
- Modify: `src/platinum/utils/claude.py`
- Modify: `tests/unit/test_claude_util.py`

**Step 1: Write failing tests.** Append to `tests/unit/test_claude_util.py`:

```python
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
```

**Step 2: Run.** Expected: ImportError.

**Step 3: Implement.** Add to `src/platinum/utils/claude.py`:

```python
import os


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
```

**Step 4: Run.** Expected: `+2 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/claude.py tests/unit/test_claude_util.py
git commit -m "feat(claude): resolve_api_key helper

Reads ANTHROPIC_API_KEY (loaded from secrets/.env by Config at process
start). Raises a clear RuntimeError before any HTTP call rather than
letting Anthropic's generic 401 bubble."
```

---

## Task 7: `claude.call` -- core async wrapper (no SDK yet)

This task wires the parts together but parameterizes the SDK boundary so we can test it with a synthetic recorder before touching `anthropic`.

**Files:**
- Modify: `src/platinum/utils/claude.py`
- Modify: `tests/unit/test_claude_util.py`

**Step 1: Write failing tests.** Append to `tests/unit/test_claude_util.py`:

```python
@pytest.mark.asyncio
async def test_call_uses_recorder_response_and_returns_claude_result(tmp_path) -> None:
    from platinum.utils.claude import call

    async def synthetic_recorder(request: dict) -> dict:
        # Anthropic's tool_use response shape
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

    # Need a Story row in place because story_id has a FK to stories.
    db_path = tmp_path / "p.db"
    create_all(db_path)
    from sqlalchemy import insert
    from platinum.models.db import StoryRow
    from datetime import datetime
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
```

**Step 2: Run.** Expected: 4 ImportErrors.

**Step 3: Implement.** Add to `src/platinum/utils/claude.py`:

```python
from datetime import datetime
from pathlib import Path
import logging

from platinum.models.db import ApiUsageRow, sync_session

logger = logging.getLogger(__name__)


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
) -> ClaudeResult:
    """One Claude call in tool-use forced mode.

    With `recorder=None` (production), talks to the real SDK [Task 8].
    With a recorder injected (tests / fixture-replay), the recorder owns
    the request -> response transformation.
    """
    request = {
        "model": model,
        "max_tokens": 8192,
        "system": _attach_cache_control(system),
        "messages": messages,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
    }

    if recorder is None:
        # Real SDK path is added in Task 8; this stub keeps the test surface
        # honest until then.
        raise NotImplementedError(
            "Live SDK path not wired yet; pass a recorder for tests."
        )
    response = await recorder(request)

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
```

**Step 4: Run.**

```
pytest tests/unit/test_claude_util.py -q
```
Expected: `13 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/claude.py tests/unit/test_claude_util.py
git commit -m "feat(claude): async call() with tool-use, recorder seam, cost tracking

Forces tool-use via tool_choice={'type':'tool','name':...}; cache_control on
the last system block. Best-effort ApiUsageRow write (logged warning on
failure, never fails the call). SDK path stubbed (recorder=None raises);
live wiring in next task. ClaudeProtocolError when response lacks tool_use."
```

---

## Task 8: Live SDK path with retry

**Files:**
- Modify: `src/platinum/utils/claude.py`
- Modify: `tests/unit/test_claude_util.py`

**Step 1: Write failing tests.** Append:

```python
@pytest.mark.asyncio
async def test_default_live_recorder_calls_async_anthropic(monkeypatch, tmp_path) -> None:
    """When recorder=None and a fake AsyncAnthropic is injected via
    client_factory, call() should hit it."""
    from platinum.models.db import create_all
    from platinum.utils.claude import call

    captured = {}

    class FakeMessage:
        def __init__(self):
            self.id = "msg_live"
            self.content = [
                type("Block", (), {"type": "tool_use", "name": "t",
                                    "input": {"ok": True}, "model_dump": lambda self: {
                                        "type": "tool_use", "name": "t", "input": {"ok": True}}})()
            ]
            self.stop_reason = "tool_use"
            self.usage = type("U", (), {
                "input_tokens": 10, "output_tokens": 5,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            })()

        def model_dump(self):
            return {
                "id": self.id,
                "content": [{"type": "tool_use", "name": "t", "input": {"ok": True}}],
                "stop_reason": self.stop_reason,
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

    # Speed up the retry sleeps to keep the test fast.
    monkeypatch.setattr("platinum.utils.retry.asyncio.sleep",
                        lambda *_a, **_k: __import__("asyncio").sleep(0))
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
```

**Step 2: Run.** Expected: live path raises NotImplementedError + retry test fails.

**Step 3: Implement.** Replace the `if recorder is None: raise NotImplementedError(...)` block in `claude.call`. Add `client_factory` parameter and a retry-decorated inner function:

```python
# At the top of claude.py, add:
from typing import Callable

import anthropic

from platinum.utils.retry import retry


# Replace the call() signature and body:
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
    client_factory: Callable[[], "anthropic.AsyncAnthropic"] | None = None,
) -> ClaudeResult:
    request = {
        "model": model,
        "max_tokens": 8192,
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
    client_factory: Callable[[], "anthropic.AsyncAnthropic"] | None,
) -> dict:
    """Hit AsyncAnthropic; raise on auth/4xx, retry on 429/5xx."""
    api_key = resolve_api_key()
    factory = client_factory or (lambda: anthropic.AsyncAnthropic(api_key=api_key))
    client = factory()
    msg = await client.messages.create(**request)
    return msg.model_dump()
```

**Step 4: Run.**

```
pytest tests/unit/test_claude_util.py -q
```
Expected: `16 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/claude.py tests/unit/test_claude_util.py
git commit -m "feat(claude): live AsyncAnthropic path with retry

Default path uses anthropic.AsyncAnthropic (key resolved fresh per call so
test monkeypatching of env works). Retry decorator wraps RateLimitError +
APIStatusError; tests stub the client_factory to avoid real network."
```

---

## Task 9: Jinja prompt-loading helper

**Files:**
- Create: `src/platinum/utils/prompts.py`
- Test: `tests/unit/test_prompts.py`

**Step 1: Write failing test.**

```python
# tests/unit/test_prompts.py
from __future__ import annotations
from pathlib import Path


def test_render_template_substitutes(tmp_path: Path) -> None:
    from platinum.utils.prompts import render_template
    (tmp_path / "atmospheric_horror").mkdir()
    (tmp_path / "atmospheric_horror" / "hello.j2").write_text(
        "Hello, {{ name }}!", encoding="utf-8"
    )
    out = render_template(
        prompts_dir=tmp_path,
        track="atmospheric_horror",
        name="hello.j2",
        context={"name": "world"},
    )
    assert out == "Hello, world!"


def test_render_template_missing_raises_clear(tmp_path: Path) -> None:
    from platinum.utils.prompts import render_template
    import pytest
    with pytest.raises(FileNotFoundError, match="missing.j2"):
        render_template(
            prompts_dir=tmp_path, track="atmospheric_horror",
            name="missing.j2", context={},
        )
```

**Step 2: Run.** Expected: ImportError.

**Step 3: Implement.** Create `src/platinum/utils/prompts.py`:

```python
"""Jinja2 prompt-template rendering used by the three Session-4 stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2


def render_template(
    *,
    prompts_dir: Path,
    track: str,
    name: str,
    context: dict[str, Any],
) -> str:
    """Render `<prompts_dir>/<track>/<name>` with `context`.

    Raises FileNotFoundError with the path on miss.
    """
    template_path = Path(prompts_dir) / track / name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    return env.get_template(name).render(**context)
```

**Step 4: Run.** Expected: `2 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/utils/prompts.py tests/unit/test_prompts.py
git commit -m "feat(prompts): render_template helper

Loads Jinja2 templates from config/prompts/<track>/<name>. StrictUndefined
fails loud on missing context keys (rather than producing silently broken
prompts). Used by the three Session-4 pipeline stages."
```

---

## Task 10: System prompt template (`system.j2`)

**Files:**
- Create: `config/prompts/atmospheric_horror/system.j2`
- Test: `tests/unit/test_prompts.py` (extend)

**Step 1: Write failing test.** Append:

```python
def test_system_template_includes_voice_and_aesthetic() -> None:
    """The shipped system.j2 must render the track config's voice direction
    and visual aesthetic."""
    import yaml
    from pathlib import Path
    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]
    track_yaml = repo_root / "config" / "tracks" / "atmospheric_horror.yaml"
    track_cfg = yaml.safe_load(track_yaml.read_text(encoding="utf-8"))["track"]

    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="system.j2",
        context={"track": track_cfg},
    )
    assert track_cfg["voice"]["direction"] in out
    assert track_cfg["visual"]["aesthetic"] in out
    assert "[whisper]" in out
    assert "[pause]" in out
```

**Step 2: Run.** Expected: FileNotFoundError.

**Step 3: Implement.** Create `config/prompts/atmospheric_horror/system.j2`. ASCII only. Use the track config dict.

```jinja
You are the head editor and showrunner for a cinematic short-film series in the "Atmospheric Horror & Dark Tales" track.

Your job is to adapt public-domain literary horror into 8-12 minute narrated short films that feel like the work of a real animation studio -- not "good for AI." Quality is non-negotiable.

VOICE DIRECTION
{{ track.voice.direction }}

The narrator speaks at roughly {{ track.voice.pace_wpm }} words per minute. You may use these emotion tags inline in narration to direct the voice actor:
{% for tag in track.voice.emotion_tags_supported %}  - {{ tag }}
{% endfor %}
Use them sparingly. Restraint sells dread.

VISUAL AESTHETIC
{{ track.visual.aesthetic }}

Palette: {{ track.visual.palette }}
Lighting: {{ track.visual.lighting }}
Influences: {{ track.visual.influences | join(", ") }}

DEFAULT NEGATIVE PROMPT (for Flux): {{ track.visual.negative_prompt }}

GENERAL PRINCIPLES
- Prefer sensory specificity over abstract dread ("the smell of cold iron" beats "an unsettling presence").
- Trust silence. A pause carries more weight than another adjective.
- Keep narration second-person sparingly; first-person and close third are the workhorses.
- Avoid telegraphing the ending in the synopsis or tone notes.
- All structured outputs must populate every required field. Schema enforcement is strict.
```

**Step 4: Run.** Expected: `3 passed`.

**Step 5: Commit.**

```bash
git add config/prompts/atmospheric_horror/system.j2 tests/unit/test_prompts.py
git commit -m "feat(prompts): atmospheric_horror system template

Cached system block: voice direction, pace_wpm, supported emotion tags,
visual aesthetic, palette, lighting, influences, default negative prompt.
ASCII-only for Windows console safety."
```

---

## Task 11: `adapt.j2` template

**Files:**
- Create: `config/prompts/atmospheric_horror/adapt.j2`
- Test: `tests/unit/test_prompts.py` (extend)

**Step 1: Write failing test.** Append:

```python
def test_adapt_template_renders_with_source_and_target() -> None:
    from pathlib import Path
    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]
    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="adapt.j2",
        context={
            "title": "The Cask of Amontillado",
            "author": "Edgar Allan Poe",
            "raw_text": "The thousand injuries of Fortunato I had borne...",
            "target_seconds": 600,
            "pace_wpm": 130,
        },
    )
    assert "Cask of Amontillado" in out
    assert "Edgar Allan Poe" in out
    assert "600" in out
    assert "130" in out
    assert "thousand injuries" in out
```

**Step 2: Run.** Expected: FileNotFoundError.

**Step 3: Implement.** Create `config/prompts/atmospheric_horror/adapt.j2`:

```jinja
Adapt the following public-domain horror short story into a polished narrated script for a {{ target_seconds }}-second short film (target pace: {{ pace_wpm }} words per minute).

TITLE: {{ title }}
{% if author %}AUTHOR: {{ author }}{% endif %}

SOURCE TEXT
---
{{ raw_text }}
---

REQUIREMENTS
1. Produce a narration_script of approximately {{ (target_seconds * pace_wpm / 60) | int }} words, give or take 5 percent.
2. Preserve the source's plot, voice, and ending. Do not invent new events.
3. Modernize phrasing where archaic syntax would slow the listener; keep period-appropriate vocabulary where it carries atmosphere.
4. Insert emotion tags (whisper / breath / pause) sparingly, only where a real voice actor would benefit.
5. Provide a synopsis of <= 400 characters that does NOT spoil the climax.
6. Provide tone_notes (a few sentences) describing pacing, register, and any tricky moments.
7. Provide an arc breakdown (setup / rising / climax / resolution) -- one paragraph each.

Submit your adaptation via the submit_adapted_story tool. The schema is strict; every required field must be populated.
```

**Step 4: Run.** Expected: `4 passed`.

**Step 5: Commit.**

```bash
git add config/prompts/atmospheric_horror/adapt.j2 tests/unit/test_prompts.py
git commit -m "feat(prompts): adapt.j2 user template

Wraps source raw_text + title/author and pulls target_seconds + pace_wpm
into the prompt body so Claude knows exactly how long to write. Asks for
arc structure, tone notes, spoiler-safe synopsis."
```

---

## Task 12: `story_adapter.py` -- pure `adapt()` function

**Files:**
- Create: `src/platinum/pipeline/story_adapter.py`
- Test: `tests/unit/test_story_adapter.py`

**Step 1: Write failing tests.** Create `tests/unit/test_story_adapter.py`:

```python
"""Unit tests for pipeline/story_adapter.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from platinum.models.story import Source, Story


def _make_story() -> Story:
    return Story(
        id="story_test",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="https://example/cask",
            title="The Cask of Amontillado",
            author="Edgar Allan Poe",
            raw_text="The thousand injuries of Fortunato I had borne as I best could...",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
    )


def _track_cfg() -> dict:
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    return yaml.safe_load(
        (repo_root / "config" / "tracks" / "atmospheric_horror.yaml").read_text(encoding="utf-8")
    )["track"]


def _synth_adapter_response(*, words: int = 1300) -> dict:
    text = " ".join(["word"] * words)
    return {
        "id": "msg_synth",
        "content": [{"type": "tool_use", "name": "submit_adapted_story", "input": {
            "title": "The Cask of Amontillado",
            "synopsis": "A man lures his rival into the catacombs.",
            "narration_script": text,
            "tone_notes": "Restrained, slow build.",
            "arc": {
                "setup": "...", "rising": "...", "climax": "...", "resolution": "...",
            },
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


@pytest.mark.asyncio
async def test_adapt_returns_adapted_with_arc(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.story_adapter import adapt

    db_path = tmp_path / "p.db"
    create_all(db_path)

    async def synth(req):
        return _synth_adapter_response(words=1300)

    result = await adapt(
        story=_make_story(),
        track_cfg=_track_cfg(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path,
        recorder=synth,
    )
    assert result.title == "The Cask of Amontillado"
    assert result.arc["climax"] == "..."
    # estimated_duration_seconds = words / pace_wpm * 60 = 1300 / 130 * 60 = 600
    assert result.estimated_duration_seconds == pytest.approx(600.0, rel=0.01)


@pytest.mark.asyncio
async def test_adapt_sends_track_voice_in_system_blocks(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.story_adapter import adapt

    captured = {}

    async def capture(req):
        captured.update(req)
        return _synth_adapter_response()

    db_path = tmp_path / "p.db"
    create_all(db_path)
    track = _track_cfg()
    await adapt(
        story=_make_story(), track_cfg=track,
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=capture,
    )
    system_text = " ".join(b["text"] for b in captured["system"])
    assert track["voice"]["direction"] in system_text


@pytest.mark.asyncio
async def test_adapt_truncates_long_source(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.story_adapter import adapt

    captured = {}

    async def capture(req):
        captured.update(req)
        return _synth_adapter_response()

    db_path = tmp_path / "p.db"
    create_all(db_path)

    long_story = _make_story()
    long_story.source.raw_text = "abc " * 30_000  # 120k chars

    await adapt(
        story=long_story, track_cfg=_track_cfg(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=capture,
    )
    user_msg = captured["messages"][0]["content"]
    assert "[...]" in user_msg
    assert len(user_msg) < 90_000


@pytest.mark.asyncio
async def test_adapt_rejects_response_missing_arc_keys(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.story_adapter import adapt
    from platinum.utils.claude import ClaudeProtocolError

    async def bad(req):
        r = _synth_adapter_response()
        r["content"][0]["input"]["arc"] = {"setup": "x"}  # missing rising/climax/resolution
        return r

    db_path = tmp_path / "p.db"
    create_all(db_path)
    with pytest.raises(ClaudeProtocolError, match="arc"):
        await adapt(
            story=_make_story(), track_cfg=_track_cfg(),
            prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
            db_path=db_path, recorder=bad,
        )
```

**Step 2: Run.** Expected: ImportError.

**Step 3: Implement.** Create `src/platinum/pipeline/story_adapter.py`:

```python
"""story_adapter pipeline -- one Claude call: source text -> polished narration.

Pure function: takes (story, track_cfg, prompts_dir, claude_call, db_path)
and returns an Adapted dataclass. No I/O outside the injected claude_call.

The Stage subclass that wires claude.call into the orchestrator lives below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from platinum.models.story import Adapted, Story
from platinum.utils.claude import ClaudeProtocolError, ClaudeResult, Recorder, call as claude_call
from platinum.utils.prompts import render_template

logger = logging.getLogger(__name__)


MODEL = "claude-opus-4-7"
MAX_SOURCE_CHARS = 80_000


ADAPT_TOOL: dict[str, Any] = {
    "name": "submit_adapted_story",
    "description": "Submit the polished cinematic adaptation.",
    "input_schema": {
        "type": "object",
        "required": ["title", "synopsis", "narration_script", "tone_notes", "arc"],
        "properties": {
            "title": {"type": "string"},
            "synopsis": {"type": "string", "maxLength": 400},
            "narration_script": {"type": "string"},
            "tone_notes": {"type": "string"},
            "arc": {
                "type": "object",
                "required": ["setup", "rising", "climax", "resolution"],
                "properties": {
                    "setup": {"type": "string"},
                    "rising": {"type": "string"},
                    "climax": {"type": "string"},
                    "resolution": {"type": "string"},
                },
            },
        },
    },
}


def _truncate_source(text: str, *, limit: int = MAX_SOURCE_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 8]
    return head + "\n[...]\n"


def _build_request(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
) -> tuple[list[dict], list[dict]]:
    system = [
        {
            "type": "text",
            "text": render_template(
                prompts_dir=prompts_dir, track=story.track,
                name="system.j2", context={"track": track_cfg},
            ),
        }
    ]
    user = [
        {
            "role": "user",
            "content": render_template(
                prompts_dir=prompts_dir, track=story.track, name="adapt.j2",
                context={
                    "title": story.source.title,
                    "author": story.source.author or "",
                    "raw_text": _truncate_source(story.source.raw_text),
                    "target_seconds": int(track_cfg["length"]["target_seconds"]),
                    "pace_wpm": int(track_cfg["voice"]["pace_wpm"]),
                },
            ),
        }
    ]
    return system, user


def _adapted_from_tool_input(tool_input: dict, *, pace_wpm: int) -> Adapted:
    arc = tool_input.get("arc", {})
    required_arc = {"setup", "rising", "climax", "resolution"}
    if not isinstance(arc, dict) or not required_arc.issubset(arc.keys()):
        raise ClaudeProtocolError(
            f"adapter response missing arc keys: got {sorted(arc.keys()) if isinstance(arc, dict) else arc!r}"
        )
    script = tool_input["narration_script"]
    word_count = len(script.split())
    return Adapted(
        title=tool_input["title"],
        synopsis=tool_input["synopsis"],
        narration_script=script,
        estimated_duration_seconds=round(word_count / pace_wpm * 60, 2),
        tone_notes=tool_input["tone_notes"],
        arc={k: arc[k] for k in ("setup", "rising", "climax", "resolution")},
    )


async def adapt(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    db_path: Path,
    recorder: Recorder | None = None,
) -> Adapted:
    """Run the adapter Claude call. Mutates nothing; returns Adapted."""
    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
    )
    pace_wpm = int(track_cfg["voice"]["pace_wpm"])

    if len(story.source.raw_text) > MAX_SOURCE_CHARS:
        logger.warning(
            "Source raw_text length %d exceeds %d; truncating in adapter prompt.",
            len(story.source.raw_text), MAX_SOURCE_CHARS,
        )

    result: ClaudeResult = await claude_call(
        model=MODEL,
        system=system, messages=messages, tool=ADAPT_TOOL,
        story_id=story.id, stage="story_adapter",
        db_path=db_path, recorder=recorder,
    )
    return _adapted_from_tool_input(result.tool_input, pace_wpm=pace_wpm)
```

**Step 4: Run.**

```
pytest tests/unit/test_story_adapter.py -q
```
Expected: `4 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/pipeline/story_adapter.py tests/unit/test_story_adapter.py
git commit -m "feat(pipeline): story_adapter -- pure adapt() with strict arc schema

One Claude call (Opus 4.7, tool-use forced). Truncates raw_text > 80k
chars with [...] marker. Computes estimated_duration_seconds from word
count and pace_wpm. ClaudeProtocolError if arc keys missing."
```

---

## Task 13: `StoryAdapterStage` -- orchestrator wrapper

**Files:**
- Modify: `src/platinum/pipeline/story_adapter.py` (append Stage class)
- Test: `tests/integration/test_adapt_stages.py`

**Step 1: Write failing test.** Create `tests/integration/test_adapt_stages.py`:

```python
"""Integration tests: each Session-4 Stage subclass runs end-to-end with a
synthetic recorder injected via PipelineContext."""

from __future__ import annotations

import yaml
from datetime import datetime
from pathlib import Path

import pytest

from platinum.models.db import create_all
from platinum.models.story import StageStatus, Source, Story
from platinum.pipeline.context import PipelineContext


def _seeded_story() -> Story:
    return Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="https://example/poe", title="The Cask",
            author="Edgar Allan Poe",
            raw_text="The thousand injuries of Fortunato I had borne...",
            fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
    )


def _track_yaml(repo_root: Path) -> dict:
    return yaml.safe_load(
        (repo_root / "config" / "tracks" / "atmospheric_horror.yaml").read_text(encoding="utf-8")
    )["track"]


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _make_context(tmp_path: Path, repo_root: Path, *, recorder) -> PipelineContext:
    """Build a PipelineContext that mirrors the real layout for stages.

    The real Stage code reads track YAML and prompts from the project root,
    so we copy them under tmp_path's config/.
    """
    import logging
    import shutil
    from platinum.config import Config

    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copytree(repo_root / "config" / "prompts", tmp_path / "config" / "prompts")
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    (tmp_path / "config" / "settings.yaml").write_text("app:\n  log_level: INFO\n", encoding="utf-8")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data").mkdir()

    cfg = Config(root=tmp_path)
    create_all(cfg.data_dir / "platinum.db")
    ctx = PipelineContext(config=cfg, logger=logging.getLogger("test"))
    # Stash the recorder where the Stage can find it.
    ctx.config.settings.setdefault("test", {})["claude_recorder"] = recorder
    return ctx


@pytest.mark.asyncio
async def test_story_adapter_stage_run_populates_adapted(tmp_path, repo_root) -> None:
    from platinum.pipeline.story_adapter import StoryAdapterStage

    async def synth(req):
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "submit_adapted_story", "input": {
                "title": "The Cask",
                "synopsis": "...",
                "narration_script": "word " * 1300,
                "tone_notes": "...",
                "arc": {"setup":"a","rising":"b","climax":"c","resolution":"d"},
            }}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 50,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    stage = StoryAdapterStage()

    artifacts = await stage.run(story, ctx)

    assert story.adapted is not None
    assert story.adapted.title == "The Cask"
    assert story.adapted.arc["climax"] == "c"
    assert artifacts["model"] == "claude-opus-4-7"
    assert artifacts["cost_usd"] > 0
```

**Step 2: Run.** Expected: ImportError on `StoryAdapterStage`.

**Step 3: Implement.** Append to `src/platinum/pipeline/story_adapter.py`:

```python
from typing import ClassVar

from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage


class StoryAdapterStage(Stage):
    """Orchestrator wrapper for the adapter Claude call."""

    name: ClassVar[str] = "story_adapter"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        track_cfg = ctx.config.track(story.track)
        # Tests inject a recorder via ctx.config.settings["test"]["claude_recorder"];
        # production leaves it None.
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        adapted = await adapt(
            story=story,
            track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path,
            recorder=recorder,
        )
        story.adapted = adapted
        return {
            "model": MODEL,
            "input_tokens": int(round(len(adapted.narration_script.split()))),
            "output_tokens": int(round(len(adapted.narration_script.split()))),
            "cost_usd": 0.0,  # the real value is in ApiUsageRow; this is a placeholder summary
        }
```

Then refine the artifacts dict to surface the actual usage. Replace the `return {...}` block with capture from the call result. Refactor: extract a private helper that returns `(adapted, ClaudeResult)` so the Stage can read the real numbers:

Replace the `adapt(...)` function body's `return _adapted_from_tool_input(...)` with:

```python
    return _adapted_from_tool_input(result.tool_input, pace_wpm=pace_wpm), result
```

Update its return annotation to `tuple[Adapted, ClaudeResult]`. Update the unit tests in `tests/unit/test_story_adapter.py` that called `result = await adapt(...)` -- they need to unpack: `result, _ = await adapt(...)`.

Update `StoryAdapterStage.run`:

```python
    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        adapted, claude_result = await adapt(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
        )
        story.adapted = adapted
        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
        }
```

**Step 4: Run.**

```
pytest tests/unit/test_story_adapter.py tests/integration/test_adapt_stages.py -q
```
Expected: `5 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/pipeline/story_adapter.py tests/unit/test_story_adapter.py tests/integration/test_adapt_stages.py
git commit -m "feat(pipeline): StoryAdapterStage orchestrator wrapper

Stage subclass writes story.adapted, returns artifacts dict including
model, token counts, cost. Test injection via ctx.config.settings['test']
['claude_recorder'] keeps production code path untouched."
```

---

## Task 14: `breakdown.j2` template

**Files:**
- Create: `config/prompts/atmospheric_horror/breakdown.j2`
- Test: `tests/unit/test_prompts.py` (extend)

**Step 1: Write failing test.** Append:

```python
def test_breakdown_template_includes_target_and_optional_feedback() -> None:
    from pathlib import Path
    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]

    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="breakdown.j2",
        context={
            "narration_script": "It was a dark and stormy night.",
            "target_seconds": 600,
            "pace_wpm": 130,
            "tolerance_seconds": 30,
            "deviation_feedback": "",
            "music_moods": ["ambient_drone", "slow_strings_dread"],
        },
    )
    assert "600" in out
    assert "ambient_drone" in out
    assert "deviation_feedback" not in out  # only the rendered string

    out2 = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror", name="breakdown.j2",
        context={
            "narration_script": "S",
            "target_seconds": 600, "pace_wpm": 130, "tolerance_seconds": 30,
            "deviation_feedback": "Previous breakdown totalled 540s; lengthen.",
            "music_moods": ["ambient_drone"],
        },
    )
    assert "Previous breakdown totalled 540s" in out2
```

**Step 2: Run.** Expected: FileNotFoundError.

**Step 3: Implement.** Create `config/prompts/atmospheric_horror/breakdown.j2`:

```jinja
Break the following narration script into a sequence of cinematic scenes, each suitable for a single 5-8 second video clip.

NARRATION SCRIPT
---
{{ narration_script }}
---

TARGET TOTAL DURATION: {{ target_seconds }} seconds (plus or minus {{ tolerance_seconds }} seconds).
NARRATION PACE: {{ pace_wpm }} words per minute.

For each scene, provide:
- A 1-based index.
- The narration_text spoken during that scene (the segment of the script for this scene only).
- A mood tag from this set: {{ music_moods | join(", ") }}.
- A list of sfx_cues (zero or more from the track's library: clock_ticking_distant, wind_through_window, footstep_wood, door_creak, candle_crackle, heartbeat_slow, etc.).

REQUIREMENTS
- Aim for 8 to 16 scenes. A scene is roughly 30-90 seconds of narration.
- Concatenated narration_text across all scenes must equal (or very nearly equal) the full input narration_script -- no editorial cuts.
- Total estimated duration (sum of words / {{ pace_wpm }} * 60) must land within {{ target_seconds - tolerance_seconds }}-{{ target_seconds + tolerance_seconds }} seconds.
- Mood tags should follow the emotional arc: do not start with the most intense mood.
{% if deviation_feedback %}
PREVIOUS ATTEMPT FEEDBACK: {{ deviation_feedback }}
{% endif %}
Submit your breakdown via the submit_scene_breakdown tool.
```

**Step 4: Run.** Expected: `5 passed`.

**Step 5: Commit.**

```bash
git add config/prompts/atmospheric_horror/breakdown.j2 tests/unit/test_prompts.py
git commit -m "feat(prompts): breakdown.j2 with optional deviation feedback

Conditionally injects 'previous attempt feedback' block when regenerating
after a tolerance miss. Mood tags drawn from the track's music.moods list
to keep selector + breakdown vocabulary aligned."
```

---

## Task 15: `scene_breakdown.py` -- helpers + `BreakdownReport`

**Files:**
- Create: `src/platinum/pipeline/scene_breakdown.py`
- Test: `tests/unit/test_scene_breakdown.py`

**Step 1: Write failing tests.** Create `tests/unit/test_scene_breakdown.py`:

```python
"""Unit tests for pipeline/scene_breakdown.py."""

from __future__ import annotations

from platinum.models.story import Scene


def test_estimate_total_seconds_pure_helper() -> None:
    from platinum.pipeline.scene_breakdown import estimate_total_seconds

    scenes = [
        Scene(id="scene_001", index=1, narration_text=" ".join(["w"] * 130)),
        Scene(id="scene_002", index=2, narration_text=" ".join(["w"] * 130)),
    ]
    # 260 words / 130 wpm = 2 minutes = 120s
    assert estimate_total_seconds(scenes, pace_wpm=130) == 120.0


def test_breakdown_report_in_tolerance_range() -> None:
    from platinum.pipeline.scene_breakdown import BreakdownReport

    r = BreakdownReport(attempts=1, final_seconds=605.0, in_tolerance=True)
    assert r.attempts == 1
    assert r.in_tolerance is True


def test_scenes_from_tool_input_assigns_ids() -> None:
    from platinum.pipeline.scene_breakdown import scenes_from_tool_input

    tool_input = {"scenes": [
        {"index": 1, "narration_text": "It begins.", "mood": "ambient_drone", "sfx_cues": ["clock_ticking_distant"]},
        {"index": 2, "narration_text": "It builds.", "mood": "slow_strings_dread", "sfx_cues": []},
    ]}
    scenes = scenes_from_tool_input(tool_input)
    assert [s.id for s in scenes] == ["scene_001", "scene_002"]
    assert [s.index for s in scenes] == [1, 2]
    assert scenes[0].music_cue == "ambient_drone"
    assert scenes[0].sfx_cues == ["clock_ticking_distant"]


def test_scenes_from_tool_input_rejects_too_few() -> None:
    from platinum.pipeline.scene_breakdown import scenes_from_tool_input
    from platinum.utils.claude import ClaudeProtocolError
    import pytest
    with pytest.raises(ClaudeProtocolError, match="minItems"):
        scenes_from_tool_input({"scenes": [
            {"index": 1, "narration_text": "x", "mood": "ambient_drone", "sfx_cues": []},
        ]})
```

**Step 2: Run.** Expected: ImportError.

**Step 3: Implement.** Create `src/platinum/pipeline/scene_breakdown.py`:

```python
"""scene_breakdown pipeline: adapter narration -> Scene list with mood/sfx.

Pure validator + regen-once flow. The pure parts -- estimate, parse,
report -- are unit-tested without any Claude call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage
from platinum.utils.claude import ClaudeProtocolError, ClaudeResult, Recorder, call as claude_call
from platinum.utils.prompts import render_template

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
MIN_SCENES = 4
MAX_SCENES = 20


BREAKDOWN_TOOL: dict[str, Any] = {
    "name": "submit_scene_breakdown",
    "description": "Submit the scene-by-scene breakdown of the narration.",
    "input_schema": {
        "type": "object",
        "required": ["scenes"],
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": MIN_SCENES,
                "maxItems": MAX_SCENES,
                "items": {
                    "type": "object",
                    "required": ["index", "narration_text", "mood", "sfx_cues"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 1},
                        "narration_text": {"type": "string"},
                        "mood": {"type": "string"},
                        "sfx_cues": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True)
class BreakdownReport:
    """Result summary returned alongside the scene list."""

    attempts: int
    final_seconds: float
    in_tolerance: bool


def estimate_total_seconds(scenes: list[Scene], *, pace_wpm: int) -> float:
    total_words = sum(len(s.narration_text.split()) for s in scenes)
    return round(total_words / pace_wpm * 60, 2)


def scenes_from_tool_input(tool_input: dict) -> list[Scene]:
    raw = tool_input.get("scenes", [])
    if not isinstance(raw, list) or len(raw) < MIN_SCENES:
        raise ClaudeProtocolError(
            f"breakdown response failed minItems={MIN_SCENES}: got {len(raw) if isinstance(raw, list) else type(raw)}"
        )
    out: list[Scene] = []
    for i, item in enumerate(raw, start=1):
        out.append(
            Scene(
                id=f"scene_{i:03d}",
                index=item.get("index", i),
                narration_text=item["narration_text"],
                music_cue=item.get("mood"),
                sfx_cues=list(item.get("sfx_cues", [])),
            )
        )
    return out
```

**Step 4: Run.**

```
pytest tests/unit/test_scene_breakdown.py -q
```
Expected: `4 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/pipeline/scene_breakdown.py tests/unit/test_scene_breakdown.py
git commit -m "feat(pipeline): scene_breakdown helpers + tool schema

estimate_total_seconds (pure), scenes_from_tool_input (parses tool result
into Scene objects with assigned ids), BREAKDOWN_TOOL schema with
minItems=4 / maxItems=20. Pure units, no Claude call yet."
```

---

## Task 16: `scene_breakdown.breakdown()` -- happy path (in tolerance on first try)

**Files:**
- Modify: `src/platinum/pipeline/scene_breakdown.py`
- Modify: `tests/unit/test_scene_breakdown.py`

**Step 1: Write failing test.** Append to `tests/unit/test_scene_breakdown.py`:

```python
import pytest
import yaml
from datetime import datetime
from pathlib import Path

from platinum.models.story import Source, Story


def _fixture_story(narration: str = "word " * 1300) -> Story:
    s = Story(
        id="story_brk",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="x", title="t", author="a",
            raw_text="raw", fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
    )
    from platinum.models.story import Adapted
    s.adapted = Adapted(
        title="t", synopsis="s",
        narration_script=narration,
        estimated_duration_seconds=600.0,
        tone_notes="n",
        arc={"setup":"a","rising":"b","climax":"c","resolution":"d"},
    )
    return s


def _track() -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "tracks" / "atmospheric_horror.yaml").read_text(encoding="utf-8")
    )["track"]


def _scene_dict(idx: int, words: int) -> dict:
    return {
        "index": idx,
        "narration_text": " ".join(["w"] * words),
        "mood": "ambient_drone",
        "sfx_cues": ["wind_through_window"] if idx % 2 == 0 else [],
    }


@pytest.mark.asyncio
async def test_breakdown_first_pass_in_tolerance(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.scene_breakdown import breakdown

    # 8 scenes, 130 words each = 1040 words / 130 wpm = 480s -- below tol
    # We want the synth response to total ~600s so first pass succeeds.
    # 600s * 130wpm / 60s = 1300 words. 8 scenes of 162-163 words each.
    async def synth(req):
        return {
            "id": "ok",
            "content": [{"type": "tool_use", "name": "submit_scene_breakdown", "input": {
                "scenes": [_scene_dict(i, 162) for i in range(1, 9)],
            }}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 50,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    db_path = tmp_path / "p.db"
    create_all(db_path)
    scenes, report, _result = await breakdown(
        story=_fixture_story(), track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=synth,
    )
    assert len(scenes) == 8
    assert report.attempts == 1
    assert report.in_tolerance is True
    # 8 * 162 / 130 * 60 = 597.69s -- in [570, 630]
    assert 570 <= report.final_seconds <= 630
```

**Step 2: Run.** Expected: ImportError on `breakdown`.

**Step 3: Implement.** Append to `src/platinum/pipeline/scene_breakdown.py`:

```python
def _build_request(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    deviation_feedback: str,
) -> tuple[list[dict], list[dict]]:
    assert story.adapted is not None, "scene_breakdown requires story.adapted set"
    target_seconds = int(track_cfg["length"]["target_seconds"])
    min_s = int(track_cfg["length"]["min_seconds"])
    max_s = int(track_cfg["length"]["max_seconds"])
    # tolerance derived from min/max so it tracks the track config exactly
    tolerance = max(target_seconds - min_s, max_s - target_seconds)

    system = [
        {"type": "text", "text": render_template(
            prompts_dir=prompts_dir, track=story.track,
            name="system.j2", context={"track": track_cfg},
        )}
    ]
    messages = [
        {"role": "user", "content": render_template(
            prompts_dir=prompts_dir, track=story.track, name="breakdown.j2",
            context={
                "narration_script": story.adapted.narration_script,
                "target_seconds": target_seconds,
                "pace_wpm": int(track_cfg["voice"]["pace_wpm"]),
                "tolerance_seconds": tolerance,
                "deviation_feedback": deviation_feedback,
                "music_moods": list(track_cfg["music"]["moods"]),
            },
        )}
    ]
    return system, messages


async def breakdown(
    *,
    story: Story,
    track_cfg: dict,
    prompts_dir: Path,
    db_path: Path,
    recorder: Recorder | None = None,
) -> tuple[list[Scene], BreakdownReport, ClaudeResult]:
    """Run the breakdown call. Regen once on tolerance miss; accept second."""
    pace_wpm = int(track_cfg["voice"]["pace_wpm"])
    target = int(track_cfg["length"]["target_seconds"])
    min_s = int(track_cfg["length"]["min_seconds"])
    max_s = int(track_cfg["length"]["max_seconds"])

    deviation_feedback = ""
    last_scenes: list[Scene] = []
    last_total: float = 0.0
    last_result: ClaudeResult | None = None

    for attempt in (1, 2):
        system, messages = _build_request(
            story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
            deviation_feedback=deviation_feedback,
        )
        result = await claude_call(
            model=MODEL, system=system, messages=messages, tool=BREAKDOWN_TOOL,
            story_id=story.id, stage="scene_breakdown",
            db_path=db_path, recorder=recorder,
        )
        scenes = scenes_from_tool_input(result.tool_input)
        total = estimate_total_seconds(scenes, pace_wpm=pace_wpm)
        last_scenes, last_total, last_result = scenes, total, result
        if min_s <= total <= max_s:
            return scenes, BreakdownReport(attempts=attempt, final_seconds=total, in_tolerance=True), result
        direction = "Lengthen" if total < min_s else "Shorten"
        deviation_feedback = (
            f"Previous breakdown totalled {total:.0f}s; target is {target}s with min={min_s}s and max={max_s}s. "
            f"{direction} scenes to land in range."
        )
        logger.info(
            "scene_breakdown attempt %d off-tolerance: %.1fs vs target %ds. Regenerating.",
            attempt, total, target,
        )

    # second attempt also off
    return last_scenes, BreakdownReport(attempts=2, final_seconds=last_total, in_tolerance=False), last_result  # type: ignore[return-value]
```

**Step 4: Run.**

```
pytest tests/unit/test_scene_breakdown.py -q
```
Expected: `5 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/pipeline/scene_breakdown.py tests/unit/test_scene_breakdown.py
git commit -m "feat(pipeline): scene_breakdown happy path (in tolerance first try)

breakdown() returns (scenes, BreakdownReport, ClaudeResult). Tolerance
derived from track YAML's min_seconds / max_seconds so it tracks config
exactly. First-pass success short-circuits the regen loop."
```

---

## Task 17: `scene_breakdown` regen-once flow + accept-on-second-miss

**Files:**
- Modify: `tests/unit/test_scene_breakdown.py` (extend; the impl already supports it)

**Step 1: Write failing tests.** Append:

```python
@pytest.mark.asyncio
async def test_breakdown_low_first_pass_triggers_regen(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.scene_breakdown import breakdown

    captured_messages: list[str] = []
    call_count = {"n": 0}

    async def two_pass(req):
        call_count["n"] += 1
        captured_messages.append(req["messages"][0]["content"])
        if call_count["n"] == 1:
            # 8 * 100 words / 130 wpm * 60 = 369s -- well below 480 min
            scenes = [_scene_dict(i, 100) for i in range(1, 9)]
        else:
            # 8 * 162 = 1296 words / 130 = 597s -- in tolerance
            scenes = [_scene_dict(i, 162) for i in range(1, 9)]
        return {"id": "x", "content": [{"type": "tool_use", "name": "submit_scene_breakdown",
                                         "input": {"scenes": scenes}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}

    db_path = tmp_path / "p.db"
    create_all(db_path)
    scenes, report, _result = await breakdown(
        story=_fixture_story(), track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=two_pass,
    )
    assert call_count["n"] == 2
    assert report.attempts == 2
    assert report.in_tolerance is True
    # Second-pass user message should contain deviation feedback
    assert "Lengthen" in captured_messages[1]
    assert "Lengthen" not in captured_messages[0]


@pytest.mark.asyncio
async def test_breakdown_high_first_pass_triggers_shorten_feedback(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.scene_breakdown import breakdown

    captured: list[str] = []
    call_count = {"n": 0}

    async def two_pass(req):
        call_count["n"] += 1
        captured.append(req["messages"][0]["content"])
        scenes = (
            [_scene_dict(i, 250) for i in range(1, 9)] if call_count["n"] == 1
            else [_scene_dict(i, 162) for i in range(1, 9)]
        )
        return {"id": "x", "content": [{"type": "tool_use", "name": "submit_scene_breakdown",
                                         "input": {"scenes": scenes}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}

    db_path = tmp_path / "p.db"
    create_all(db_path)
    _, report, _ = await breakdown(
        story=_fixture_story(), track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=two_pass,
    )
    assert report.attempts == 2
    assert "Shorten" in captured[1]


@pytest.mark.asyncio
async def test_breakdown_second_pass_still_off_returns_in_tolerance_false(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.scene_breakdown import breakdown

    call_count = {"n": 0}

    async def always_low(req):
        call_count["n"] += 1
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "submit_scene_breakdown",
                         "input": {"scenes": [_scene_dict(i, 80) for i in range(1, 9)]}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    db_path = tmp_path / "p.db"
    create_all(db_path)
    scenes, report, _ = await breakdown(
        story=_fixture_story(), track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=always_low,
    )
    assert call_count["n"] == 2  # only TWO attempts, not three
    assert report.in_tolerance is False
    assert len(scenes) == 8  # we still got the second-pass scenes back
```

**Step 2: Run.** Expected: `8 passed` (impl already supports these branches).

**Step 3 / 4 / 5:** No new code. Commit the additional tests:

```bash
git add tests/unit/test_scene_breakdown.py
git commit -m "test(scene_breakdown): regen-once flow coverage

Three new cases: low first pass + 'Lengthen' feedback, high first pass +
'Shorten' feedback, second-pass still off -> in_tolerance=False with no
third attempt."
```

---

## Task 18: `SceneBreakdownStage` orchestrator wrapper

**Files:**
- Modify: `src/platinum/pipeline/scene_breakdown.py`
- Modify: `tests/integration/test_adapt_stages.py`

**Step 1: Write failing test.** Append to `tests/integration/test_adapt_stages.py`:

```python
@pytest.mark.asyncio
async def test_scene_breakdown_stage_run_populates_scenes(tmp_path, repo_root) -> None:
    from platinum.models.story import Adapted
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage

    async def synth(req):
        scenes = [{
            "index": i, "narration_text": " ".join(["w"] * 162),
            "mood": "ambient_drone", "sfx_cues": [],
        } for i in range(1, 9)]
        return {
            "id": "x", "content": [{"type": "tool_use",
                                     "name": "submit_scene_breakdown",
                                     "input": {"scenes": scenes}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    story.adapted = Adapted(
        title="t", synopsis="s", narration_script="word " * 1300,
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup":"a","rising":"b","climax":"c","resolution":"d"},
    )
    stage = SceneBreakdownStage()
    artifacts = await stage.run(story, ctx)

    assert len(story.scenes) == 8
    assert artifacts["attempts"] == 1
    assert artifacts["in_tolerance"] is True
    assert artifacts["final_seconds"] > 0
```

**Step 2: Run.** Expected: ImportError on `SceneBreakdownStage`.

**Step 3: Implement.** Append to `src/platinum/pipeline/scene_breakdown.py`:

```python
class SceneBreakdownStage(Stage):
    name: ClassVar[str] = "scene_breakdown"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        if story.adapted is None:
            raise RuntimeError(
                f"scene_breakdown requires story_adapter to have populated story.adapted "
                f"first (story={story.id})."
            )
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        scenes, report, claude_result = await breakdown(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
        )
        story.scenes = scenes
        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
            "attempts": report.attempts,
            "final_seconds": report.final_seconds,
            "in_tolerance": report.in_tolerance,
        }
```

**Step 4: Run.**

```
pytest tests/integration/test_adapt_stages.py -q
```
Expected: `2 passed` (the two integration tests we have so far).

**Step 5: Commit.**

```bash
git add src/platinum/pipeline/scene_breakdown.py tests/integration/test_adapt_stages.py
git commit -m "feat(pipeline): SceneBreakdownStage orchestrator wrapper

Stage subclass writes story.scenes from the breakdown call; surfaces
attempts / final_seconds / in_tolerance in the StageRun artifacts so
they're visible via 'platinum status --story <id>'. Hard-fails fast if
story.adapted is missing (orchestrator ordering invariant)."
```

---

## Task 19: `visual_prompts.j2` template + parsing helper

**Files:**
- Create: `config/prompts/atmospheric_horror/visual_prompts.j2`
- Create: `src/platinum/pipeline/visual_prompts.py`
- Test: `tests/unit/test_prompts.py`, `tests/unit/test_visual_prompts.py`

**Step 1: Write failing tests.**

Append to `tests/unit/test_prompts.py`:

```python
def test_visual_prompts_template_includes_aesthetic_and_scenes() -> None:
    from pathlib import Path
    from platinum.utils.prompts import render_template
    repo_root = Path(__file__).resolve().parents[2]

    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="visual_prompts.j2",
        context={
            "aesthetic": "cinematic dark",
            "default_negative": "bright, anime",
            "palette": "deep shadow",
            "scenes": [
                {"index": 1, "narration_text": "It was a dark night."},
                {"index": 2, "narration_text": "He went into the cellar."},
            ],
        },
    )
    assert "cinematic dark" in out
    assert "bright, anime" in out
    assert "1." in out
    assert "dark night" in out
    assert "into the cellar" in out
```

Create `tests/unit/test_visual_prompts.py`:

```python
"""Unit tests for pipeline/visual_prompts.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from platinum.models.story import Adapted, Scene, Source, Story


def _track() -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "tracks" / "atmospheric_horror.yaml").read_text(encoding="utf-8")
    )["track"]


def _story_with_scenes(n: int = 4) -> Story:
    s = Story(
        id="story_vp", track="atmospheric_horror",
        source=Source(type="g", url="x", title="t", author="a",
                      raw_text="r", fetched_at=datetime(2026, 4, 25), license="PD-US"),
    )
    s.adapted = Adapted(title="t", synopsis="s", narration_script="x",
                         estimated_duration_seconds=600.0, tone_notes="n",
                         arc={"setup":"","rising":"","climax":"","resolution":""})
    s.scenes = [Scene(id=f"scene_{i:03d}", index=i, narration_text=f"text {i}")
                for i in range(1, n + 1)]
    return s


def _synth_response(n: int) -> dict:
    return {
        "id": "x",
        "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
            "scenes": [
                {"index": i, "visual_prompt": f"vp{i}", "negative_prompt": f"np{i}"}
                for i in range(1, n + 1)
            ],
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


@pytest.mark.asyncio
async def test_visual_prompts_zips_into_scenes_by_index(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(4)

    async def synth(req):
        # Return out-of-order to exercise zip-by-index
        r = _synth_response(4)
        r["content"][0]["input"]["scenes"].reverse()
        return r

    scenes, _ = await visual_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=synth,
    )
    assert [(s.index, s.visual_prompt, s.negative_prompt) for s in scenes] == [
        (1, "vp1", "np1"), (2, "vp2", "np2"), (3, "vp3", "np3"), (4, "vp4", "np4"),
    ]


@pytest.mark.asyncio
async def test_visual_prompts_count_must_match_scene_count(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(4)

    async def synth(req):
        return _synth_response(3)  # only 3 prompts for 4 scenes

    with pytest.raises(ClaudeProtocolError, match="count"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
            db_path=db_path, recorder=synth,
        )
```

**Step 2: Run.** Expected: FileNotFoundError + ImportError.

**Step 3: Implement.** Create `config/prompts/atmospheric_horror/visual_prompts.j2`:

```jinja
For each of the following scenes, write a Flux-style visual prompt and a negative prompt.

TRACK AESTHETIC
{{ aesthetic }}

PALETTE: {{ palette }}
DEFAULT NEGATIVE PROMPT: {{ default_negative }}

CONVENTIONS
- Visual prompts target Flux 1.0 Dev. Roughly 60-140 tokens, comma-separated descriptors.
- Lead with the subject and setting, then lighting, then mood and texture, then a quality clause (e.g. "oil painting quality, fine film grain").
- Include the palette terms only when they fit -- do not force.
- Negative prompt may extend the default with scene-specific exclusions (e.g. "no clear daylight" if the narration places the action in a sunlit field that the track aesthetic forbids).
- Output exactly one entry per scene; preserve the input scene index.

SCENES
{% for scene in scenes %}{{ scene.index }}. {{ scene.narration_text }}

{% endfor %}
Submit your prompts via the submit_visual_prompts tool.
```

Create `src/platinum/pipeline/visual_prompts.py`:

```python
"""visual_prompts pipeline: per-scene Flux visual + negative prompt strings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage
from platinum.utils.claude import ClaudeProtocolError, ClaudeResult, Recorder, call as claude_call
from platinum.utils.prompts import render_template

logger = logging.getLogger(__name__)
MODEL = "claude-opus-4-7"


VISUAL_PROMPTS_TOOL: dict[str, Any] = {
    "name": "submit_visual_prompts",
    "description": "Submit per-scene Flux visual + negative prompts.",
    "input_schema": {
        "type": "object",
        "required": ["scenes"],
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "visual_prompt", "negative_prompt"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 1},
                        "visual_prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                    },
                },
            },
        },
    },
}


def _build_request(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
) -> tuple[list[dict], list[dict]]:
    system = [
        {"type": "text", "text": render_template(
            prompts_dir=prompts_dir, track=story.track,
            name="system.j2", context={"track": track_cfg},
        )}
    ]
    messages = [
        {"role": "user", "content": render_template(
            prompts_dir=prompts_dir, track=story.track, name="visual_prompts.j2",
            context={
                "aesthetic": track_cfg["visual"]["aesthetic"],
                "palette": track_cfg["visual"]["palette"],
                "default_negative": track_cfg["visual"]["negative_prompt"],
                "scenes": [{"index": s.index, "narration_text": s.narration_text} for s in story.scenes],
            },
        )}
    ]
    return system, messages


def _zip_into_scenes(story_scenes: list[Scene], tool_input: dict) -> list[Scene]:
    raw = tool_input.get("scenes", [])
    if len(raw) != len(story_scenes):
        raise ClaudeProtocolError(
            f"visual_prompts count {len(raw)} != scene count {len(story_scenes)}"
        )
    by_index = {item["index"]: item for item in raw}
    out: list[Scene] = []
    for scene in story_scenes:
        item = by_index.get(scene.index)
        if item is None:
            raise ClaudeProtocolError(
                f"visual_prompts response missing scene index {scene.index}"
            )
        scene.visual_prompt = item["visual_prompt"]
        scene.negative_prompt = item["negative_prompt"]
        out.append(scene)
    return out


async def visual_prompts(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    db_path: Path, recorder: Recorder | None = None,
) -> tuple[list[Scene], ClaudeResult]:
    if not story.scenes:
        raise RuntimeError(
            f"visual_prompts requires scene_breakdown to have populated story.scenes "
            f"first (story={story.id})."
        )
    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
    )
    result = await claude_call(
        model=MODEL, system=system, messages=messages, tool=VISUAL_PROMPTS_TOOL,
        story_id=story.id, stage="visual_prompts",
        db_path=db_path, recorder=recorder,
    )
    scenes = _zip_into_scenes(story.scenes, result.tool_input)
    return scenes, result


class VisualPromptsStage(Stage):
    name: ClassVar[str] = "visual_prompts"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        if not story.scenes:
            raise RuntimeError(
                f"visual_prompts requires scene_breakdown completed first (story={story.id})."
            )
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        _, claude_result = await visual_prompts(
            story=story, track_cfg=track_cfg,
            prompts_dir=ctx.config.prompts_dir,
            db_path=ctx.db_path, recorder=recorder,
        )
        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
        }
```

**Step 4: Run.**

```
pytest tests/unit/test_visual_prompts.py tests/unit/test_prompts.py -q
```
Expected: `+3 passed`.

**Step 5: Commit.**

```bash
git add config/prompts/atmospheric_horror/visual_prompts.j2 src/platinum/pipeline/visual_prompts.py tests/unit/test_visual_prompts.py tests/unit/test_prompts.py
git commit -m "feat(pipeline): visual_prompts stage + Flux-convention template

One Claude call returns per-scene visual + negative prompts; zip into
existing Scene objects by index. ClaudeProtocolError on count mismatch
or missing index. Stage subclass + tool schema follow the same pattern
as story_adapter and scene_breakdown."
```

---

## Task 20: `VisualPromptsStage` integration test

**Files:**
- Modify: `tests/integration/test_adapt_stages.py`

**Step 1: Write failing test.** Append:

```python
@pytest.mark.asyncio
async def test_visual_prompts_stage_run_populates_prompts(tmp_path, repo_root) -> None:
    from platinum.models.story import Adapted, Scene
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    async def synth(req):
        return {
            "id": "x",
            "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
                "scenes": [
                    {"index": 1, "visual_prompt": "vault, candle", "negative_prompt": "bright"},
                    {"index": 2, "visual_prompt": "fog, stones", "negative_prompt": "neon"},
                ],
            }}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    story.adapted = Adapted(
        title="t", synopsis="s", narration_script="word " * 1300,
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup":"a","rising":"b","climax":"c","resolution":"d"},
    )
    story.scenes = [
        Scene(id="scene_001", index=1, narration_text="It was night."),
        Scene(id="scene_002", index=2, narration_text="The vault opened."),
    ]
    stage = VisualPromptsStage()
    artifacts = await stage.run(story, ctx)

    assert story.scenes[0].visual_prompt == "vault, candle"
    assert story.scenes[1].negative_prompt == "neon"
    assert artifacts["model"] == "claude-opus-4-7"
```

**Step 2: Run.** Expected: PASS (impl already exists).

**Step 3 / 4 / 5:** Commit:

```bash
git add tests/integration/test_adapt_stages.py
git commit -m "test(visual_prompts): integration test for the Stage subclass"
```

---

## Task 21: Three-stage end-to-end and resume tests

**Files:**
- Modify: `tests/integration/test_adapt_stages.py`

**Step 1: Write failing tests.** Append:

```python
@pytest.mark.asyncio
async def test_three_stages_in_sequence_produces_complete_story(tmp_path, repo_root) -> None:
    """Run StoryAdapterStage -> SceneBreakdownStage -> VisualPromptsStage
    via the orchestrator and verify the final Story has all three populated."""
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.story_adapter import StoryAdapterStage
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    async def router(req):
        # Distinguish by tool_choice name to dispatch the right canned response.
        tool_name = req["tool_choice"]["name"]
        if tool_name == "submit_adapted_story":
            return {
                "id": "ad", "content": [{"type": "tool_use", "name": tool_name, "input": {
                    "title": "T", "synopsis": "S", "narration_script": "word " * 1300,
                    "tone_notes": "N",
                    "arc": {"setup":"a","rising":"b","climax":"c","resolution":"d"},
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if tool_name == "submit_scene_breakdown":
            return {
                "id": "br", "content": [{"type": "tool_use", "name": tool_name, "input": {
                    "scenes": [
                        {"index": i, "narration_text": " ".join(["w"] * 162),
                         "mood": "ambient_drone", "sfx_cues": []}
                        for i in range(1, 9)
                    ],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if tool_name == "submit_visual_prompts":
            return {
                "id": "vp", "content": [{"type": "tool_use", "name": tool_name, "input": {
                    "scenes": [{"index": i, "visual_prompt": f"vp{i}",
                                 "negative_prompt": f"np{i}"} for i in range(1, 9)],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        raise AssertionError(f"unexpected tool_choice: {tool_name}")

    ctx = _make_context(tmp_path, repo_root, recorder=router)
    story = _seeded_story()

    orch = Orchestrator(stages=[
        StoryAdapterStage(), SceneBreakdownStage(), VisualPromptsStage(),
    ])
    final = await orch.run(story, ctx)

    assert final.adapted is not None
    assert len(final.scenes) == 8
    assert all(s.visual_prompt for s in final.scenes)
    completed = [r for r in final.stages if r.status == StageStatus.COMPLETE]
    assert {r.stage for r in completed} >= {"story_adapter", "scene_breakdown", "visual_prompts"}


@pytest.mark.asyncio
async def test_resume_skips_completed_stage(tmp_path, repo_root) -> None:
    """A pre-existing COMPLETE story_adapter run should make that stage skip."""
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.story_adapter import StoryAdapterStage

    calls = {"n": 0}

    async def synth(req):
        calls["n"] += 1
        return {
            "id": "x", "content": [{"type": "tool_use", "name": "submit_adapted_story",
                                     "input": {"title": "T", "synopsis": "S",
                                                "narration_script": "x", "tone_notes": "n",
                                                "arc": {"setup":"a","rising":"b",
                                                         "climax":"c","resolution":"d"}}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }

    ctx = _make_context(tmp_path, repo_root, recorder=synth)
    story = _seeded_story()
    # Mark adapter already complete.
    from platinum.models.story import StageRun
    story.stages.append(StageRun(
        stage="story_adapter", status=StageStatus.COMPLETE,
        started_at=datetime(2026, 4, 25), completed_at=datetime(2026, 4, 25),
    ))

    orch = Orchestrator(stages=[StoryAdapterStage()])
    await orch.run(story, ctx)
    assert calls["n"] == 0  # adapter was skipped
```

**Step 2: Run.** Expected: both pass with the existing implementations.

**Step 3 / 4 / 5:** Commit:

```bash
git add tests/integration/test_adapt_stages.py
git commit -m "test(integration): three-stage end-to-end + resume coverage

Router-style synthetic recorder dispatches by tool name; full pipeline
yields adapted + scenes + visuals. Resume test verifies orchestrator
skips a stage whose latest StageRun is already COMPLETE."
```

---

## Task 22: CLI `adapt` command

**Files:**
- Modify: `src/platinum/cli.py`
- Create: `tests/integration/test_adapt_command.py`

**Step 1: Write failing tests.** Create `tests/integration/test_adapt_command.py`:

```python
"""Integration tests for `platinum adapt`."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_project(tmp_path: Path) -> Iterator[Path]:
    """Mirror the real project layout under tmp_path for CLI tests."""
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    shutil.copytree(repo_root / "config" / "prompts", tmp_path / "config" / "prompts")
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    (tmp_path / "config" / "settings.yaml").write_text(
        "app:\n  log_level: INFO\n", encoding="utf-8"
    )
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data" / "stories").mkdir(parents=True)
    yield tmp_path


def _seed_curated_story(project: Path, story_id: str) -> Path:
    from platinum.models.story import (
        Source, StageRun, StageStatus, Story,
    )
    s = Story(
        id=story_id, track="atmospheric_horror",
        source=Source(
            type="gutenberg", url=f"https://example/{story_id}",
            title=f"Title {story_id}", author="A", raw_text="raw...",
            fetched_at=datetime(2026, 4, 25), license="PD-US",
        ),
        stages=[
            StageRun(stage="source_fetcher", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25)),
            StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                      started_at=datetime(2026, 4, 25),
                      completed_at=datetime(2026, 4, 25),
                      artifacts={"decision": "approved"}),
        ],
    )
    story_dir = project / "data" / "stories" / story_id
    story_dir.mkdir(parents=True)
    s.save(story_dir / "story.json")
    return story_dir / "story.json"


def _router_factory():
    """Return a recorder that dispatches by tool name."""
    async def router(req):
        name = req["tool_choice"]["name"]
        if name == "submit_adapted_story":
            return {
                "id": "ad", "content": [{"type": "tool_use", "name": name, "input": {
                    "title": "T", "synopsis": "S", "narration_script": "word " * 1300,
                    "tone_notes": "n",
                    "arc": {"setup":"a","rising":"b","climax":"c","resolution":"d"},
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        if name == "submit_scene_breakdown":
            return {
                "id": "br", "content": [{"type": "tool_use", "name": name, "input": {
                    "scenes": [{"index": i, "narration_text": " ".join(["w"] * 162),
                                 "mood": "ambient_drone", "sfx_cues": []}
                                for i in range(1, 9)],
                }}], "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            }
        return {
            "id": "vp", "content": [{"type": "tool_use", "name": name, "input": {
                "scenes": [{"index": i, "visual_prompt": f"vp{i}", "negative_prompt": f"np{i}"}
                            for i in range(1, 9)],
            }}], "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        }
    return router


def test_cli_adapt_walks_eligible_stories(cli_project, monkeypatch) -> None:
    """`platinum adapt` adapts every curator-approved-but-not-yet-adapted story."""
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr(
        "platinum.config._ROOT", cli_project, raising=False,
    )
    # Wire the recorder into Config.settings via a monkeypatched factory.
    router = _router_factory()

    original_init = cli_mod.Config.__init__

    def init_with_recorder(self, root=None):
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = router

    monkeypatch.setattr(cli_mod, "Config", type("C", (cli_mod.Config,), {"__init__": init_with_recorder}))

    _seed_curated_story(cli_project, "story_a")
    _seed_curated_story(cli_project, "story_b")

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt"])
    assert result.exit_code == 0, result.output

    for sid in ("story_a", "story_b"):
        data = json.loads((cli_project / "data" / "stories" / sid / "story.json").read_text(encoding="utf-8"))
        assert data["adapted"] is not None
        assert len(data["scenes"]) == 8
        assert all(s.get("visual_prompt") for s in data["scenes"])


def test_cli_adapt_no_eligible_stories_exits_zero(cli_project, monkeypatch) -> None:
    from platinum import cli as cli_mod

    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt"])
    assert result.exit_code == 0
    assert "no eligible stories" in result.output.lower()


def test_cli_adapt_story_filter_targets_one(cli_project, monkeypatch) -> None:
    from platinum import cli as cli_mod
    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    router = _router_factory()
    original_init = cli_mod.Config.__init__

    def init_with_recorder(self, root=None):
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = router

    monkeypatch.setattr(cli_mod, "Config", type("C", (cli_mod.Config,), {"__init__": init_with_recorder}))

    _seed_curated_story(cli_project, "story_a")
    _seed_curated_story(cli_project, "story_b")

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["adapt", "--story", "story_a"])
    assert result.exit_code == 0

    a = json.loads((cli_project / "data" / "stories" / "story_a" / "story.json").read_text(encoding="utf-8"))
    b = json.loads((cli_project / "data" / "stories" / "story_b" / "story.json").read_text(encoding="utf-8"))
    assert a["adapted"] is not None
    assert b["adapted"] is None
```

**Step 2: Run.** Expected: stub error or wrong exit code.

**Step 3: Implement.** Replace the `adapt` command body in `src/platinum/cli.py`. First, update the `from platinum.pipeline.story_curator` import block to also import the new stages and orchestrator. Replace the existing `adapt` function:

```python
@app.command()
def adapt(
    story: Optional[str] = typer.Option(
        None, "--story", "-s", help="Adapt only this story id."
    ),
    track: Optional[str] = typer.Option(
        None, "--track", "-t", help="Restrict to one track id."
    ),
) -> None:
    """Adapt curator-approved stories: narration -> scenes -> visual prompts.

    Walks `data/stories/*/story.json` and runs the three Session-4 Stages
    (story_adapter, scene_breakdown, visual_prompts) on every story whose
    curator decision was approve and whose visual_prompts stage is not
    yet COMPLETE. Resume-safe: stages already COMPLETE are skipped.
    """
    import asyncio
    import logging

    from platinum.models.story import StageStatus, Story
    from platinum.pipeline.context import PipelineContext
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage
    from platinum.pipeline.story_adapter import StoryAdapterStage
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    cfg = Config()
    eligible: list[Story] = []
    for story_dir in sorted(p for p in cfg.stories_dir.iterdir() if p.is_dir()):
        story_json = story_dir / "story.json"
        if not story_json.exists():
            continue
        try:
            s = Story.load(story_json)
        except Exception as exc:
            console.print(f"[yellow]Skipping unreadable story at {story_dir.name}: {exc}[/yellow]")
            continue

        if story is not None and s.id != story:
            continue
        if track is not None and s.track != track:
            continue

        curator = s.latest_stage_run("story_curator")
        if curator is None or curator.status != StageStatus.COMPLETE:
            continue

        vp = s.latest_stage_run("visual_prompts")
        if vp is not None and vp.status == StageStatus.COMPLETE:
            continue

        eligible.append(s)

    if not eligible:
        console.print("[yellow]No eligible stories to adapt.[/yellow]")
        return

    ctx = PipelineContext(config=cfg, logger=logging.getLogger("platinum.adapt"))
    orchestrator = Orchestrator(stages=[
        StoryAdapterStage(), SceneBreakdownStage(), VisualPromptsStage(),
    ])

    for s in eligible:
        console.print(f"[cyan]Adapting {s.id} (track={s.track})...[/cyan]")
        try:
            asyncio.run(orchestrator.run(s, ctx))
        except Exception as exc:
            console.print(f"[red]{s.id} failed: {exc}[/red]")
            raise

    console.print(f"[green]Adapted {len(eligible)} story candidate(s).[/green]")
```

**Step 4: Run.**

```
pytest tests/integration/test_adapt_command.py -q
```
Expected: `3 passed`.

**Step 5: Commit.**

```bash
git add src/platinum/cli.py tests/integration/test_adapt_command.py
git commit -m "feat(cli): platinum adapt -- batch-walk approved stories

Replaces the stub. Walks data/stories/, runs the three Session-4 Stages
via Orchestrator on every approved-but-not-yet-adapted story. --story
and --track filters available. Reuses orchestrator skip-if-complete so
re-running after a partial failure resumes from the right stage."
```

---

## Task 23: Status command output reflects new stages

**Files:**
- Modify: `tests/integration/test_adapt_command.py` (extend)

**Step 1: Write failing test.** Append:

```python
def test_cli_adapt_then_status_reflects_complete(cli_project, monkeypatch) -> None:
    from platinum import cli as cli_mod
    monkeypatch.chdir(cli_project)
    monkeypatch.setattr("platinum.config._ROOT", cli_project, raising=False)

    router = _router_factory()
    original_init = cli_mod.Config.__init__

    def init_with_recorder(self, root=None):
        original_init(self, root=root)
        self.settings.setdefault("test", {})["claude_recorder"] = router

    monkeypatch.setattr(cli_mod, "Config", type("C", (cli_mod.Config,), {"__init__": init_with_recorder}))

    _seed_curated_story(cli_project, "story_x")

    runner = CliRunner()
    assert runner.invoke(cli_mod.app, ["adapt"]).exit_code == 0

    status = runner.invoke(cli_mod.app, ["status", "--story", "story_x"])
    assert status.exit_code == 0
    assert "story_adapter" in status.output
    # Each of the three stages should show as COMPLETE
    assert status.output.count("COMPLETE") >= 3
```

**Step 2 / 3:** Should pass already; just commit.

```bash
git add tests/integration/test_adapt_command.py
git commit -m "test(cli): status reflects three new stages as COMPLETE after adapt"
```

---

## Task 24: Full test sweep + ruff + mypy

**Steps:**

```bash
cd C:/Users/claws/OneDrive/Desktop/platinum
pytest -q
```
Expected: ~149 tests, 0 failed.

```bash
ruff check src tests
```
Expected: clean. Fix any reported issues, run `ruff check --fix src tests` if appropriate, re-run.

```bash
mypy src/platinum/utils/claude.py src/platinum/utils/prompts.py src/platinum/pipeline/story_adapter.py src/platinum/pipeline/scene_breakdown.py src/platinum/pipeline/visual_prompts.py
```
Expected: clean (annotate any leftover `dict` parameters with `dict[str, Any]`).

If any commits are needed for lint/type fixes:

```bash
git add -p
git commit -m "chore(session-4): ruff + mypy passes"
```

---

## Task 25: Live smoke + fixture recording (manual; user runs once)

**Goal:** Run the full pipeline once against the real Anthropic API to (a) verify it works end-to-end and (b) capture fixtures we can replay forever after.

**Steps (user runs interactively):**

1. Confirm `secrets/.env` has `ANTHROPIC_API_KEY=sk-ant-...`. If absent, get one at https://console.anthropic.com/.
2. Fetch a fresh candidate (Session 2 already wired):
   ```
   python -m platinum fetch --track atmospheric_horror --limit 3
   ```
3. Curate the candidates (Session 3):
   ```
   python -m platinum curate
   ```
   Approve at least one.
4. Run the new adapter against the approved story. Without recording yet:
   ```
   python -m platinum adapt --track atmospheric_horror
   ```
   Expected: ~$0.30-1.00 spent, exit 0, three Stage rows COMPLETE per approved story. Open `data/stories/<id>/story.json` and skim `adapted.narration_script` -- it should read like a polished narration.
5. Verify SQLite cost tracking:
   ```
   python -c "import sqlite3; rows = sqlite3.connect('data/platinum.db').execute('SELECT model, input_tokens, output_tokens, cost_usd FROM api_usage').fetchall(); print(rows)"
   ```
   Expect 3 rows per adapted story (one per stage) totalling ~$0.50.
6. Status reads back cleanly:
   ```
   python -m platinum status --story <id>
   ```
   The first 5 stages should show COMPLETE.

**Optional fixture recording** (defer to Task 26 if the smoke worked).

**Commit (project state, not code):**

```bash
git add -p tasks/todo.md     # if you tracked notes there
git commit -m "chore: Session 4 smoke notes (no functional changes)" --allow-empty
```

---

## Task 26: Update `tasks/todo.md` and write the Session 4 review

**Files:**
- Modify: `tasks/todo.md` (append Session 4 review section)

**Step 1.** Append a `## Review (Session 4 complete - YYYY-MM-DD)` block following the same shape as Session 3's review (see prior commit `4a27281`). Cover: tests added, files changed, deliverable verified, surprises / lessons, deferred items.

**Step 2.** Commit:

```bash
git add tasks/todo.md
git commit -m "docs: Session 4 review -- Claude integration + story adapter

149 tests passing (45 new). Three Stages, one CLI verb, recorded fixtures,
prompt caching, cost tracking, ~\$0.48/story typical end-to-end."
```

---

## Skill cross-references

- @superpowers:test-driven-development -- the strict red-green-refactor loop you should already know.
- @superpowers:verification-before-completion -- the "would a staff engineer approve this?" gate before you call it done.
- @superpowers:systematic-debugging -- if any test fails you don't immediately understand, use this; do not "fix" by changing assertions.

## Open follow-ups (deferred)

- Multi-track prompts. Only `atmospheric_horror/` is authored this session. When the second track ships its first story, copy the four templates and tune them.
- Real prompt-quality iteration (read the live narration_script and adjust system.j2 / adapt.j2). Likely 1-2 follow-up commits after the smoke.
- Per-track config loader module (`tracks/`). For now we read the YAML directly.
