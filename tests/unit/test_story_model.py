"""Round-trip + new-field tests for the Scene dataclass.

S7 adds review_feedback (str | None), regen_count (int), and reject_count
(int) to Scene.
"""
from __future__ import annotations

from platinum.models.story import Scene


def _build_minimal_scene(**overrides) -> Scene:
    base = {
        "id": "scene_001",
        "index": 1,
        "narration_text": "x",
    }
    base.update(overrides)
    return Scene(**base)


def test_scene_review_feedback_defaults_to_none() -> None:
    scene = _build_minimal_scene()
    assert scene.review_feedback is None


def test_scene_regen_count_defaults_to_zero() -> None:
    scene = _build_minimal_scene()
    assert scene.regen_count == 0


def test_scene_round_trip_preserves_review_feedback() -> None:
    scene = _build_minimal_scene(review_feedback="needs more candle light")
    rt = Scene.from_dict(scene.to_dict())
    assert rt.review_feedback == "needs more candle light"


def test_scene_round_trip_preserves_regen_count() -> None:
    scene = _build_minimal_scene(regen_count=3)
    rt = Scene.from_dict(scene.to_dict())
    assert rt.regen_count == 3


def test_scene_from_dict_backfills_missing_review_feedback() -> None:
    """Old story.json files without these fields must still load."""
    raw = {"id": "s1", "index": 1, "narration_text": "x"}
    scene = Scene.from_dict(raw)
    assert scene.review_feedback is None
    assert scene.regen_count == 0
    assert scene.reject_count == 0


def test_scene_reject_count_defaults_to_zero() -> None:
    scene = _build_minimal_scene()
    assert scene.reject_count == 0


def test_scene_round_trip_preserves_reject_count() -> None:
    scene = _build_minimal_scene(reject_count=2)
    rt = Scene.from_dict(scene.to_dict())
    assert rt.reject_count == 2


def test_scene_b2_1_fields_default_to_empty_or_none() -> None:
    """S7.1.B2.1: Scene gains four reference-conditioning fields with
    benign defaults so older Stories work without changes:

      character_refs: list[str]   default []     (no recurring characters)
      pose_ref_path: str | None   default None   (no pose preprocessor output)
      depth_ref_path: str | None  default None   (no depth preprocessor output)
      composition_notes: str | None default None (no scene blocking notes)
    """
    scene = _build_minimal_scene()
    assert scene.character_refs == []
    assert scene.pose_ref_path is None
    assert scene.depth_ref_path is None
    assert scene.composition_notes is None


def test_scene_round_trip_preserves_b2_1_fields() -> None:
    """S7.1.B2.1: to_dict / from_dict preserves all four new fields."""
    scene = _build_minimal_scene(
        character_refs=["Fortunato", "Montresor"],
        pose_ref_path="data/stories/x/keyframes/scene_007/_pose.png",
        depth_ref_path="data/stories/x/keyframes/scene_007/_depth.png",
        composition_notes=(
            "Medium shot. Two men face each other across a vault arch. "
            "Foreground: Montresor in dark cloak holding a torch."
        ),
    )
    rt = Scene.from_dict(scene.to_dict())
    assert rt.character_refs == ["Fortunato", "Montresor"]
    assert rt.pose_ref_path == "data/stories/x/keyframes/scene_007/_pose.png"
    assert rt.depth_ref_path == "data/stories/x/keyframes/scene_007/_depth.png"
    assert rt.composition_notes is not None
    assert "vault arch" in rt.composition_notes


def test_scene_from_dict_backfills_missing_b2_1_fields() -> None:
    """S7.1.B2.1: pre-S7.1 story.json files (no character_refs / pose_ref_path /
    depth_ref_path / composition_notes keys) must still load with defaults."""
    raw = {"id": "s1", "index": 1, "narration_text": "x"}
    scene = Scene.from_dict(raw)
    assert scene.character_refs == []
    assert scene.pose_ref_path is None
    assert scene.depth_ref_path is None
    assert scene.composition_notes is None


# ---- S7.1.B2.2: Story.characters dict --------------------------------------


def _build_minimal_story():
    """Tiny Story with one Scene; B2.2 tests the new characters dict only."""
    from datetime import UTC, datetime

    from platinum.models.story import Source, Story
    return Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="https://example.test/source",
            title="t",
            author="a",
            raw_text="rt",
            fetched_at=datetime(2026, 4, 29, tzinfo=UTC),
            license="PD-US",
        ),
        scenes=[_build_minimal_scene()],
    )


def test_story_characters_defaults_to_empty_dict() -> None:
    """S7.1.B2.2: Story.characters defaults to {} so older Stories work."""
    story = _build_minimal_story()
    assert story.characters == {}


def test_story_round_trip_preserves_characters() -> None:
    """S7.1.B2.2: characters dict round-trips through to_dict / from_dict."""
    from platinum.models.story import Story

    story = _build_minimal_story()
    story.characters = {
        "Fortunato": "data/stories/x/references/Fortunato/candidate_2.png",
        "Montresor": "data/stories/x/references/Montresor/candidate_0.png",
    }
    rt = Story.from_dict(story.to_dict())
    assert rt.characters == {
        "Fortunato": "data/stories/x/references/Fortunato/candidate_2.png",
        "Montresor": "data/stories/x/references/Montresor/candidate_0.png",
    }


def test_story_from_dict_backfills_missing_characters_field() -> None:
    """S7.1.B2.2: pre-S7.1 story.json files (no `characters` key) load with {}."""
    from datetime import UTC, datetime

    from platinum.models.story import Source, Story

    raw = {
        "id": "story_x",
        "track": "atmospheric_horror",
        "source": Source(
            type="gutenberg", url="u", title="t", author="a", raw_text="rt",
            fetched_at=datetime(2026, 4, 29, tzinfo=UTC), license="PD-US",
        ).to_dict(),
        "scenes": [],
        # NB: no `characters` key -- pre-S7.1 layout
    }
    story = Story.from_dict(raw)
    assert story.characters == {}
