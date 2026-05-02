"""Round-trip + immutability tests for the StoryBible / BibleScene dataclasses.

S8.B.1 introduces:
  * BibleScene  -- per-scene narrative directive (frozen dataclass)
  * StoryBible  -- whole-story directive containing N BibleScene + global continuity
"""
from __future__ import annotations

import dataclasses

import pytest

from platinum.models.story_bible import BibleScene, StoryBible


def _scene(**over) -> BibleScene:
    base = dict(
        index=1,
        narrative_beat="Montresor seethes alone the night Fortunato wronged him.",
        hero_shot="close-up, profile, eye-level",
        visible_characters=["Montresor"],
        gaze_map={"Montresor": "into middle distance"},
        props_visible=["pewter goblet", "beeswax candle"],
        blocking="Montresor mid-frame, candle foreground-right",
        light_source="single beeswax candle, foreground right, warm amber",
        color_anchors=["black wool cloak", "dark oak desk"],
        brightness_floor="low",
    )
    base.update(over)
    return BibleScene(**base)


def _bible(**over) -> StoryBible:
    base = dict(
        world_genre_atmosphere="19th c. Italian carnival into vaulted catacombs.",
        character_continuity={
            "Montresor": {
                "face": "lean European nobleman, gaunt, dark goatee",
                "costume": "black wool cloak, fur collar",
                "posture": "patient, rigid, calculating",
            },
            "Fortunato": {
                "face": "ruddy, jovial",
                "costume": "motley red+yellow stripes, conical jester cap with bells",
                "posture": "swaying, drunk",
            },
        },
        environment_continuity={
            "palazzo_study": "ash-grey wall, oak desk, single candle",
            "carnival_street": "torchlit cobblestone, foggy",
            "catacombs": "stone niches, niter-coated walls, torchlight",
        },
        scenes=[_scene(index=1)],
    )
    base.update(over)
    return StoryBible(**base)


# ----- BibleScene basics ----------------------------------------------------


def test_bible_scene_constructs_with_required_fields() -> None:
    s = _scene()
    assert s.index == 1
    assert s.visible_characters == ["Montresor"]
    assert s.brightness_floor == "low"


def test_bible_scene_is_frozen() -> None:
    s = _scene()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.index = 2  # type: ignore[misc]


def test_bible_scene_round_trip_via_asdict() -> None:
    s = _scene(visible_characters=["Montresor", "Fortunato"], brightness_floor="medium")
    d = dataclasses.asdict(s)
    rt = BibleScene(**d)
    assert rt == s


def test_bible_scene_brightness_floor_accepts_low_medium_high() -> None:
    for level in ("low", "medium", "high"):
        s = _scene(brightness_floor=level)
        assert s.brightness_floor == level


# ----- StoryBible basics ----------------------------------------------------


def test_story_bible_constructs_with_required_fields() -> None:
    b = _bible()
    assert "Montresor" in b.character_continuity
    assert len(b.scenes) == 1


def test_story_bible_is_frozen() -> None:
    b = _bible()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.world_genre_atmosphere = "x"  # type: ignore[misc]


def test_story_bible_round_trip_via_asdict() -> None:
    b = _bible(scenes=[_scene(index=1), _scene(index=2, visible_characters=["Fortunato"])])
    d = dataclasses.asdict(b)
    rt = StoryBible.from_dict(d)
    assert rt == b


def test_story_bible_to_dict_round_trips() -> None:
    """to_dict + from_dict pair is what Story.save / Story.load uses."""
    b = _bible()
    rt = StoryBible.from_dict(b.to_dict())
    assert rt == b


# ----- Edge cases used by downstream code ----------------------------------


def test_story_bible_empty_continuity_dicts_round_trip() -> None:
    b = _bible(character_continuity={}, environment_continuity={})
    rt = StoryBible.from_dict(b.to_dict())
    assert rt.character_continuity == {}
    assert rt.environment_continuity == {}


def test_bible_scene_lists_are_independent_after_round_trip() -> None:
    """Mutating a list returned by from_dict must not poison the source dict."""
    b = _bible()
    d = b.to_dict()
    rt = StoryBible.from_dict(d)
    # rt's scene list is a different object than the source dict's
    assert rt.scenes is not d["scenes"]
    # And it must not be the same object as b.scenes either.
    assert rt.scenes is not b.scenes
