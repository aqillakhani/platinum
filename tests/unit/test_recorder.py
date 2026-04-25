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
