"""Unit tests for pipeline/story_bible.py (S8.B.2)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from platinum.models.story import Adapted, Scene, Source, Story
from platinum.models.story_bible import BibleScene, StoryBible
from platinum.pipeline.story_bible import (
    STORY_BIBLE_TOOL,
    _zip_into_story,
    story_bible,
)
from platinum.utils.claude import ClaudeProtocolError

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


def _track() -> dict:
    track_path = (
        Path(__file__).resolve().parents[2]
        / "config" / "tracks" / "atmospheric_horror.yaml"
    )
    cfg = yaml.safe_load(track_path.read_text(encoding="utf-8"))["track"]
    cfg.setdefault("story_bible", {
        "enabled": True,
        "model": "claude-opus-4-7",
        "max_tokens": 16000,
    })
    cfg["story_bible"]["enabled"] = True
    return cfg


def _story_with_scenes(n: int = 3) -> Story:
    s = Story(
        id="story_bible_test", track="atmospheric_horror",
        source=Source(type="g", url="x", title="t", author="a",
                      raw_text="r", fetched_at=datetime(2026, 5, 2), license="PD-US"),
    )
    s.adapted = Adapted(title="t", synopsis="s", narration_script="x",
                         estimated_duration_seconds=600.0, tone_notes="n",
                         arc={"setup": "", "rising_action": "", "climax": "", "resolution": ""})
    s.scenes = [
        Scene(id=f"scene_{i:03d}", index=i, narration_text=f"narration {i}")
        for i in range(1, n + 1)
    ]
    return s


def _bible_scene_dict(*, index: int, **over) -> dict:
    base = {
        "index": index,
        "narrative_beat": f"beat {index}",
        "hero_shot": "close-up",
        "visible_characters": ["Montresor"],
        "gaze_map": {"Montresor": "off-camera"},
        "props_visible": ["candle"],
        "blocking": "centered",
        "light_source": "single beeswax candle",
        "color_anchors": ["black cloak"],
        "brightness_floor": "low",
    }
    base.update(over)
    return base


def _synth_bible(n: int) -> dict:
    """Hand-crafted bible tool_input matching N scenes."""
    return {
        "world_genre_atmosphere": "Italian carnival into vaulted catacombs.",
        "character_continuity": {
            "Montresor": {
                "face": "lean noble, dark goatee",
                "costume": "black wool cloak",
                "posture": "patient, calculating",
            },
        },
        "environment_continuity": {"palazzo": "ash-grey wall, oak desk"},
        "scenes": [_bible_scene_dict(index=i) for i in range(1, n + 1)],
    }


def _synth_response(n: int) -> dict:
    return {
        "id": "x",
        "content": [{
            "type": "tool_use",
            "name": "submit_story_bible",
            "input": _synth_bible(n),
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


# ---------- Tool schema --------------------------------------------------


def test_tool_schema_has_required_top_level_keys() -> None:
    schema = STORY_BIBLE_TOOL["input_schema"]
    assert set(schema["required"]) == {
        "world_genre_atmosphere",
        "character_continuity",
        "environment_continuity",
        "scenes",
    }


def test_tool_schema_brightness_floor_enum_low_medium_high() -> None:
    scene_props = STORY_BIBLE_TOOL["input_schema"]["properties"]["scenes"]
    assert scene_props["items"]["properties"]["brightness_floor"]["enum"] == [
        "low", "medium", "high",
    ]


# ---------- _zip_into_story ---------------------------------------------


def test_zip_into_story_happy_path() -> None:
    story = _story_with_scenes(3)
    bible = _zip_into_story(story, _synth_bible(3))
    assert isinstance(bible, StoryBible)
    assert story.bible is bible
    assert [s.index for s in bible.scenes] == [1, 2, 3]


def test_zip_into_story_missing_top_level_key_raises() -> None:
    story = _story_with_scenes(2)
    bad = _synth_bible(2)
    bad.pop("character_continuity")
    with pytest.raises(ClaudeProtocolError, match="missing top-level"):
        _zip_into_story(story, bad)


def test_zip_into_story_count_mismatch_raises() -> None:
    story = _story_with_scenes(3)
    with pytest.raises(ClaudeProtocolError, match="scenes count"):
        _zip_into_story(story, _synth_bible(2))  # wrong count


def test_zip_into_story_missing_index_raises() -> None:
    story = _story_with_scenes(3)
    bad = _synth_bible(3)
    bad["scenes"][1]["index"] = 99  # out-of-range index breaks the index match
    with pytest.raises(ClaudeProtocolError, match="missing scene index"):
        _zip_into_story(story, bad)


def test_zip_into_story_missing_required_scene_field_raises() -> None:
    story = _story_with_scenes(2)
    bad = _synth_bible(2)
    del bad["scenes"][0]["light_source"]
    with pytest.raises(ClaudeProtocolError, match="missing required field"):
        _zip_into_story(story, bad)


def test_zip_into_story_preserves_story_scene_order() -> None:
    """Even if the bible response is out of order, the resulting StoryBible
    should preserve the response order (zip_into_story does not re-sort).
    The downstream visual_prompts rewriter looks up by index, not position."""
    story = _story_with_scenes(3)
    bible_in = _synth_bible(3)
    bible_in["scenes"].reverse()
    bible = _zip_into_story(story, bible_in)
    assert [s.index for s in bible.scenes] == [3, 2, 1]


# ---------- story_bible() coroutine -------------------------------------


@pytest.mark.asyncio
async def test_story_bible_happy_path(tmp_path: Path) -> None:
    from platinum.models.db import create_all
    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(3)

    async def synth(req):
        # Confirm max_tokens overridden to 16000 per S8.B design.
        assert req["max_tokens"] == 16000, req["max_tokens"]
        return _synth_response(3)

    bible, result = await story_bible(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR,
        db_path=db_path, recorder=synth,
    )
    assert story.bible is bible
    assert len(bible.scenes) == 3
    assert result.usage.model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_story_bible_disabled_track_raises(tmp_path: Path) -> None:
    from platinum.models.db import create_all
    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)
    track = _track()
    track["story_bible"]["enabled"] = False

    with pytest.raises(RuntimeError, match="story_bible.enabled=false"):
        await story_bible(
            story=story, track_cfg=track,
            prompts_dir=PROMPTS_DIR,
            db_path=db_path, recorder=lambda req: _synth_response(2),
        )


@pytest.mark.asyncio
async def test_story_bible_no_scenes_raises(tmp_path: Path) -> None:
    from platinum.models.db import create_all
    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(0)

    with pytest.raises(RuntimeError, match="scene_breakdown"):
        await story_bible(
            story=story, track_cfg=_track(),
            prompts_dir=PROMPTS_DIR,
            db_path=db_path, recorder=lambda req: _synth_response(0),
        )


@pytest.mark.asyncio
async def test_story_bible_request_includes_scene_narrations(tmp_path: Path) -> None:
    """Confirm story narrations are threaded into the user message — the
    bible writer needs them to identify characters and beats."""
    from platinum.models.db import create_all
    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(3)
    story.scenes[0].narration_text = "Distinctive narration zebrafish 42"

    captured: dict = {}
    async def synth(req):
        captured["req"] = req
        return _synth_response(3)

    await story_bible(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR,
        db_path=db_path, recorder=synth,
    )
    user_text = captured["req"]["messages"][0]["content"]
    assert "Distinctive narration zebrafish 42" in user_text


# ---------- StoryBibleStage.is_complete ---------------------------------


def test_stage_is_complete_false_when_no_bible() -> None:
    from platinum.pipeline.story_bible import StoryBibleStage
    story = _story_with_scenes(3)
    assert story.bible is None
    assert StoryBibleStage().is_complete(story) is False


def test_stage_is_complete_true_when_bible_covers_all_scenes() -> None:
    from platinum.pipeline.story_bible import StoryBibleStage
    story = _story_with_scenes(3)
    _zip_into_story(story, _synth_bible(3))
    assert StoryBibleStage().is_complete(story) is True


def test_stage_is_complete_false_when_bible_partial() -> None:
    from platinum.pipeline.story_bible import StoryBibleStage
    story = _story_with_scenes(3)
    # Construct a bible that only covers scene 1 + 2 (missing 3)
    bible = StoryBible(
        world_genre_atmosphere="x",
        scenes=[
            BibleScene(**_bible_scene_dict(index=1)),
            BibleScene(**_bible_scene_dict(index=2)),
        ],
    )
    story.bible = bible
    assert StoryBibleStage().is_complete(story) is False
