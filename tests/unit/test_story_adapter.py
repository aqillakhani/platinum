"""Unit tests for pipeline/story_adapter.py."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

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

    adapted, claude_result = await adapt(
        story=_make_story(),
        track_cfg=_track_cfg(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path,
        recorder=synth,
    )
    assert adapted.title == "The Cask of Amontillado"
    assert adapted.arc["climax"] == "..."
    # estimated_duration_seconds = words / pace_wpm * 60 = 1300 / 130 * 60 = 600
    assert adapted.estimated_duration_seconds == pytest.approx(600.0, rel=0.01)
    assert claude_result.usage.input_tokens == 100


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
