"""Story dataclass round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path

from platinum.models.story import (
    Adapted,
    ReviewStatus,
    Scene,
    Source,
    StageRun,
    StageStatus,
    Story,
)


def test_story_roundtrip_to_dict_from_dict(story: Story) -> None:
    d = story.to_dict()
    rebuilt = Story.from_dict(d)
    assert rebuilt.to_dict() == d


def test_story_roundtrip_through_json_file(story: Story, tmp_path: Path) -> None:
    path = tmp_path / "story.json"
    story.save(path)
    loaded = Story.load(path)
    assert loaded.to_dict() == story.to_dict()


def test_story_save_is_atomic_on_tmp_rename(story: Story, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "story.json"
    story.save(path)
    assert path.exists()
    # No leftover *.tmp files beside it.
    tmp_siblings = list(path.parent.glob("*.tmp"))
    assert tmp_siblings == [], f"leftover tmp files: {tmp_siblings}"


def test_enums_serialize_as_strings(story: Story) -> None:
    d = story.to_dict()
    assert d["scenes"][0]["review_status"] == ReviewStatus.PENDING.value
    assert d["stages"][0]["status"] == StageStatus.COMPLETE.value
    # Ensure the raw JSON blob is plain strings, not enum reprs.
    text = json.dumps(d)
    assert "ReviewStatus." not in text
    assert "StageStatus." not in text


def test_path_fields_normalize_to_posix(tmp_path: Path) -> None:
    scene = Scene(
        id="scene_001",
        index=0,
        narration_text="x",
        narration_audio_path=Path("C:\\some\\windows\\path.wav"),
        keyframe_candidates=[Path("a/b.png"), Path("c\\d.png")],
    )
    d = scene.to_dict()
    # Stored as forward-slash strings so JSON is portable cross-platform.
    assert "\\" not in d["narration_audio_path"]
    for p in d["keyframe_candidates"]:
        assert "\\" not in p


def test_latest_stage_run_returns_most_recent(story: Story) -> None:
    from datetime import datetime

    # First run failed, second succeeded — latest should be the second.
    story.stages = [
        StageRun(stage="keyframe_generator", status=StageStatus.FAILED,
                 started_at=datetime(2026, 4, 24, 10), error="boom"),
        StageRun(stage="keyframe_generator", status=StageStatus.COMPLETE,
                 started_at=datetime(2026, 4, 24, 11)),
    ]
    latest = story.latest_stage_run("keyframe_generator")
    assert latest is not None
    assert latest.status == StageStatus.COMPLETE


def test_latest_stage_run_missing_returns_none(story: Story) -> None:
    assert story.latest_stage_run("publisher") is None


def test_adapted_roundtrip() -> None:
    adapted = Adapted(
        title="Shortened",
        synopsis="One line.",
        narration_script="Once upon a time.",
        estimated_duration_seconds=180.0,
        tone_notes="Slow and measured.",
        arc={"setup": "s", "rising_action": "r", "climax": "c", "resolution": "res"},
    )
    assert Adapted.from_dict(adapted.to_dict()) == adapted


def test_source_roundtrip(source: Source) -> None:
    assert Source.from_dict(source.to_dict()) == source
