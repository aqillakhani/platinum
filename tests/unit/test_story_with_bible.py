"""Story round-trip tests with the optional bible field (S8.B.1).

A Story without a bible round-trips unchanged (back-compat). A Story with
a bible round-trips through to_dict / from_dict and through save / load.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from platinum.models.story import Story
from platinum.models.story_bible import BibleScene, StoryBible


def _build_bible() -> StoryBible:
    return StoryBible(
        world_genre_atmosphere="Italian carnival → catacombs.",
        character_continuity={
            "Montresor": {"face": "lean noble", "costume": "black cloak", "posture": "patient"},
            "Fortunato": {"face": "ruddy", "costume": "motley jester", "posture": "drunk"},
        },
        environment_continuity={"catacombs": "stone niches, niter walls"},
        scenes=[
            BibleScene(
                index=0,
                narrative_beat="Establishing brood.",
                hero_shot="close-up profile",
                visible_characters=["Montresor"],
                gaze_map={"Montresor": "into the dark"},
                props_visible=["pewter goblet", "candle"],
                blocking="Montresor mid-frame",
                light_source="single beeswax candle",
                color_anchors=["black cloak"],
                brightness_floor="low",
            ),
            BibleScene(
                index=1,
                narrative_beat="Tavern meeting.",
                hero_shot="two-shot, medium",
                visible_characters=["Montresor", "Fortunato"],
                gaze_map={"Montresor": "Fortunato", "Fortunato": "wine"},
                props_visible=["wine bottle"],
                blocking="Fortunato left, Montresor right",
                light_source="lantern overhead",
                color_anchors=["motley red+yellow"],
                brightness_floor="medium",
            ),
        ],
    )


def test_story_default_bible_is_none(story: Story) -> None:
    """Existing fixture still has bible=None — no breaking change."""
    assert story.bible is None


def test_story_round_trip_with_bible(story: Story) -> None:
    story.bible = _build_bible()
    rebuilt = Story.from_dict(story.to_dict())
    assert rebuilt.bible == story.bible


def test_story_round_trip_without_bible_back_compat(story: Story) -> None:
    """Old story.json files lack the 'bible' key entirely; from_dict must
    backfill bible=None instead of raising."""
    d = story.to_dict()
    d.pop("bible", None)  # simulate an older serialization
    rebuilt = Story.from_dict(d)
    assert rebuilt.bible is None


def test_story_save_and_load_with_bible(story: Story, tmp_path: Path) -> None:
    story.bible = _build_bible()
    path = tmp_path / "story.json"
    story.save(path)
    loaded = Story.load(path)
    assert loaded.bible == story.bible


def test_story_to_dict_with_bible_is_pure_json_types(story: Story) -> None:
    """Asserts to_dict produces a dict tree with only JSON-safe primitives —
    so json.dump succeeds without a custom encoder."""
    import json
    story.bible = _build_bible()
    text = json.dumps(story.to_dict(), ensure_ascii=False)
    assert "Montresor" in text
    assert "BibleScene(" not in text  # not the dataclass repr


def test_story_save_omits_bible_when_none(story: Story, tmp_path: Path) -> None:
    """A saved story with bible=None must round-trip with bible still None
    (key may be present-as-null or absent — both are valid for back-compat)."""
    assert story.bible is None
    path = tmp_path / "story.json"
    story.save(path)
    loaded = Story.load(path)
    assert loaded.bible is None


@pytest.mark.parametrize("brightness_level", ["low", "medium", "high"])
def test_bible_scene_brightness_levels_round_trip(
    story: Story, tmp_path: Path, brightness_level: str
) -> None:
    bible = _build_bible()
    bible.scenes[0] = BibleScene(
        **{**bible.scenes[0].__dict__, "brightness_floor": brightness_level}
    )
    story.bible = bible
    path = tmp_path / "story.json"
    story.save(path)
    loaded = Story.load(path)
    assert loaded.bible is not None
    assert loaded.bible.scenes[0].brightness_floor == brightness_level
