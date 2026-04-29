"""Tests for pipeline/character_references.py -- CharacterReferenceStage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from platinum.models.story import Scene, Source, Story


def _scene(idx: int, *, character_refs: list[str] | None = None) -> Scene:
    return Scene(
        id=f"scene_{idx:03d}",
        index=idx,
        narration_text=f"text {idx}",
        character_refs=list(character_refs or []),
    )


def _story(scenes: list[Scene], *, characters: dict[str, str] | None = None) -> Story:
    s = Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="u", title="t", author="a", raw_text="rt",
            fetched_at=datetime(2026, 4, 29, tzinfo=UTC), license="PD-US",
        ),
    )
    s.scenes = scenes
    if characters is not None:
        s.characters = dict(characters)
    return s


def test_stage_name_is_character_references() -> None:
    """S7.1.B4.1: stage registers under 'character_references'."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    assert CharacterReferenceStage.name == "character_references"


def test_is_complete_when_no_characters_in_any_scene() -> None:
    """S7.1.B4.1: stories without recurring characters need no refs;
    is_complete is True so the stage is a no-op."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    story = _story([_scene(1), _scene(2)])  # no character_refs anywhere
    stage = CharacterReferenceStage()
    assert stage.is_complete(story) is True


def test_is_complete_false_when_character_discovered_but_not_picked() -> None:
    """S7.1.B4.1: a character mentioned in scene.character_refs but absent
    from story.characters means the user has not picked a ref yet ->
    stage is not complete."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    story = _story(
        [_scene(1, character_refs=["Fortunato"])],
        characters={},
    )
    stage = CharacterReferenceStage()
    assert stage.is_complete(story) is False


def test_is_complete_false_when_picked_path_missing_on_disk(tmp_path: Path) -> None:
    """S7.1.B4.1: if Story.characters[name] points at a path that doesn't
    exist on disk, the ref is broken -- stage is not complete."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    missing = tmp_path / "ghost.png"  # never created
    story = _story(
        [_scene(1, character_refs=["Fortunato"])],
        characters={"Fortunato": str(missing)},
    )
    stage = CharacterReferenceStage()
    assert stage.is_complete(story) is False


def test_is_complete_true_when_all_characters_picked_and_files_exist(
    tmp_path: Path,
) -> None:
    """S7.1.B4.1: every discovered character has a picked path AND that
    path exists on disk -> is_complete=True."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    fortunato = tmp_path / "Fortunato.png"
    montresor = tmp_path / "Montresor.png"
    fortunato.write_bytes(b"\x89PNG_f")
    montresor.write_bytes(b"\x89PNG_m")
    story = _story(
        [
            _scene(1, character_refs=["Fortunato", "Montresor"]),
            _scene(2, character_refs=["Fortunato"]),
        ],
        characters={
            "Fortunato": str(fortunato),
            "Montresor": str(montresor),
        },
    )
    stage = CharacterReferenceStage()
    assert stage.is_complete(story) is True


# ---- B4.2: _discover_characters --------------------------------------------


def test_discover_characters_empty_when_no_scene_has_character_refs() -> None:
    """S7.1.B4.2: no character_refs anywhere -> empty set."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    story = _story([_scene(1), _scene(2)])
    stage = CharacterReferenceStage()
    assert stage._discover_characters(story) == set()


def test_discover_characters_collects_partial_set() -> None:
    """S7.1.B4.2: only some scenes have character_refs -> union of those."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    story = _story([
        _scene(1, character_refs=["Fortunato"]),
        _scene(2),  # no character_refs
        _scene(3, character_refs=["Montresor"]),
    ])
    stage = CharacterReferenceStage()
    assert stage._discover_characters(story) == {"Fortunato", "Montresor"}


def test_discover_characters_dedupes_across_scenes() -> None:
    """S7.1.B4.2: same character across multiple scenes is collapsed to one entry."""
    from platinum.pipeline.character_references import CharacterReferenceStage

    story = _story([
        _scene(1, character_refs=["Fortunato", "Montresor"]),
        _scene(2, character_refs=["Fortunato"]),
        _scene(3, character_refs=["Montresor", "Fortunato"]),
    ])
    stage = CharacterReferenceStage()
    assert stage._discover_characters(story) == {"Fortunato", "Montresor"}
