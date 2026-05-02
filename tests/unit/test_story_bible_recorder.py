"""Live recorder fixture for the story_bible stage (S8.B.3).

Replay mode (default): loads the recorded fixture for the Cask story and
asserts structural invariants on the bible Opus 4.7 produces — counts, enum
ranges, character-continuity exhaustiveness, scene-level non-emptiness.

Record mode (PLATINUM_RECORD_FIXTURES=1): hits Anthropic with the actual
production code path (story_bible() → claude.call → _live_call) on the
real Cask story.json. ~$0.30 per record. The recorder writes the
(request, response) pair to disk so subsequent CI runs replay for free.

Mirrors the content_check recorder pattern at
tests/unit/test_content_check.py::test_claude_content_checker_records_and_replays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from platinum.models.story import Story

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "config" / "prompts"
TRACK_PATH = REPO_ROOT / "config" / "tracks" / "atmospheric_horror.yaml"
CASK_STORY_PATH = REPO_ROOT / "data" / "stories" / "story_2026_04_25_001" / "story.json"


def _cask_fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "anthropic"
        / "story_bible"
        / "test_records_and_replays_cask__1.json"
    )


def _load_track_cfg() -> dict:
    cfg = yaml.safe_load(TRACK_PATH.read_text(encoding="utf-8"))["track"]
    # Defensive: if the YAML hasn't been updated yet (older branches), force-enable
    # so the test exercises the same code path the production track will.
    cfg.setdefault("story_bible", {
        "enabled": True,
        "model": "claude-opus-4-7",
        "max_tokens": 16000,
    })
    cfg["story_bible"]["enabled"] = True
    return cfg


def _load_cask_story() -> Story:
    raw = json.loads(CASK_STORY_PATH.read_text(encoding="utf-8"))
    return Story.from_dict(raw)


@pytest.mark.asyncio
async def test_records_and_replays_cask(tmp_path: Path) -> None:
    """S8.B.3: live Opus 4.7 bible call against the Cask story round-trips
    through FixtureRecorder. Asserts the structural shape every downstream
    visual_prompts rewriter depends on.
    """
    from platinum.models.db import create_all
    from platinum.pipeline.story_bible import story_bible
    from platinum.utils.claude import _live_call
    from tests._fixtures import FixtureRecorder

    fixture_path = _cask_fixture_path()
    record = os.environ.get("PLATINUM_RECORD_FIXTURES") == "1"

    db_path = tmp_path / "p.db"
    create_all(db_path)

    track_cfg = _load_track_cfg()
    story = _load_cask_story()
    # Reset bible so this test exercises a fresh generation path even if
    # someone has hand-set it on the on-disk Cask story.json.
    story.bible = None

    if record:
        # Trigger secrets/.env load so ANTHROPIC_API_KEY reaches resolve_api_key().
        from platinum.config import Config
        Config()

        async def live(request: dict) -> dict:
            return await _live_call(request, client_factory=None)

        recorder = FixtureRecorder(path=fixture_path, mode="record", live=live)
    else:
        recorder = FixtureRecorder(path=fixture_path, mode="replay")

    bible, result = await story_bible(
        story=story, track_cfg=track_cfg,
        prompts_dir=PROMPTS_DIR,
        db_path=db_path, recorder=recorder,
    )

    # Cost path resolves (model present in pricing table).
    assert result.usage.model == "claude-opus-4-7"
    assert result.usage.input_tokens > 0

    # Structural invariants the visual_prompts rewriter depends on.
    assert len(bible.scenes) == len(story.scenes) == 16

    # World atmosphere is a populated paragraph.
    assert isinstance(bible.world_genre_atmosphere, str)
    assert len(bible.world_genre_atmosphere.strip()) > 30

    # character_continuity covers the two protagonists with non-empty
    # face/costume/posture signatures.
    assert "Montresor" in bible.character_continuity, bible.character_continuity
    assert "Fortunato" in bible.character_continuity, bible.character_continuity
    for name in ("Montresor", "Fortunato"):
        sig = bible.character_continuity[name]
        assert sig["face"].strip(), f"{name}.face empty"
        assert sig["costume"].strip(), f"{name}.costume empty"
        assert sig["posture"].strip(), f"{name}.posture empty"

    # Per-scene structural shape: light_source non-empty, brightness_floor in
    # the documented enum, gaze_map ⊆ visible_characters (system prompt
    # forbids gaze entries for off-stage characters). visible_characters MAY
    # be empty for pure landscape / object-focus shots — the system prompt
    # explicitly permits this for closing tableaus etc.
    landscape_scenes = 0
    for sc in bible.scenes:
        assert sc.light_source.strip(), f"scene {sc.index}: light_source empty"
        assert sc.brightness_floor in {"low", "medium", "high"}, (
            f"scene {sc.index}: brightness_floor={sc.brightness_floor!r}"
        )
        gaze_names = set(sc.gaze_map.keys())
        visible = set(sc.visible_characters)
        assert gaze_names <= visible, (
            f"scene {sc.index}: gaze_map names {gaze_names - visible} "
            f"not in visible_characters {visible}"
        )
        if not sc.visible_characters:
            landscape_scenes += 1
    # At most a small fraction of scenes can be character-less landscapes;
    # if Opus collapses too many scenes to "no characters" the bible is no
    # longer useful for grounding two-character beats.
    assert landscape_scenes <= 2, (
        f"too many empty-visible scenes ({landscape_scenes}); "
        "bible should keep characters in frame for narrative beats"
    )

    # Scene 9 is the iconic "trowel reveal" — both characters MUST be in
    # frame. The S8.B prototype's closure depended on the bible getting this
    # one right; pinning it here prevents prompt regressions.
    scene9 = next(s for s in bible.scenes if s.index == 9)
    assert "Montresor" in scene9.visible_characters, scene9.visible_characters
    assert "Fortunato" in scene9.visible_characters, scene9.visible_characters
