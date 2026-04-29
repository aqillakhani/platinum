"""Tests for pipeline/character_extraction.py -- proper-noun heuristic."""

from __future__ import annotations

from platinum.models.story import Scene


def _scene(idx: int, text: str) -> Scene:
    return Scene(id=f"scene_{idx:03d}", index=idx, narration_text=text)


def test_extract_character_names_finds_recurring_proper_nouns() -> None:
    """S7.1.B3.1: names appearing in 2+ scenes are kept."""
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [
        _scene(1, "Fortunato laughed at Montresor."),
        _scene(2, "Montresor smiled. Fortunato did not."),
    ]
    assert set(extract_character_names(scenes)) == {"Fortunato", "Montresor"}


def test_extract_character_names_ignores_one_off_capitalized_words() -> None:
    """S7.1.B3.1: names appearing in only one scene are filtered out.

    Single-scene names are likely place-of-the-moment proper nouns
    (Venice) or one-off mentions, not recurring characters that need an
    IP-Adapter face reference.
    """
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [_scene(1, "In Venice, Fortunato met him.")]
    assert extract_character_names(scenes) == []


def test_extract_character_names_drops_common_sentence_starts() -> None:
    """S7.1.B3.1: capitalized articles / pronouns / prepositions that
    happen to start sentences are not treated as character names."""
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [
        _scene(1, "The traveler entered. He drew a coin."),
        _scene(2, "The traveler turned. He drew his blade."),
    ]
    # "The" and "He" appear in 2 scenes but they aren't names.
    # "traveler" is lowercase so it isn't a candidate.
    assert extract_character_names(scenes) == []


def test_extract_character_names_handles_multi_word_names() -> None:
    """S7.1.B3.1: multi-word names ("Lord Montresor", "Don Quixote") are
    captured as a single phrase when both tokens are capitalized."""
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [
        _scene(1, "Lord Montresor descended into the vault."),
        _scene(2, "Lord Montresor lifted his torch."),
    ]
    assert "Lord Montresor" in extract_character_names(scenes)


def test_extract_character_names_strips_leading_stopword_from_multi_word_match() -> None:
    """S7.1.B3.1: 'In Venice' matches the multi-word regex but the leading
    'In' is an English preposition, not part of the name. The leading
    token is stripped, leaving 'Venice' as the candidate (which then
    gets filtered as one-off because Venice only appears in one scene).
    """
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [
        _scene(1, "In Venice, the carnival raged."),
        _scene(2, "In Venice, the canals reflected torchlight."),
    ]
    # 'Venice' (after stripping 'In') appears in 2 scenes, so it survives
    # the recurrence filter. We accept this corner case: place names
    # repeated across scenes will look like characters to the heuristic;
    # users override via review UI / characters.json.
    assert "Venice" in extract_character_names(scenes)


def test_extract_character_names_returns_sorted_list() -> None:
    """S7.1.B3.1: deterministic ordering for stable test fixtures."""
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [
        _scene(1, "Zara faced Aria."),
        _scene(2, "Aria turned to Zara."),
    ]
    assert extract_character_names(scenes) == ["Aria", "Zara"]


def test_extract_character_names_handles_empty_or_missing_text() -> None:
    """S7.1.B3.1: scenes with empty narration_text are skipped without errors."""
    from platinum.pipeline.character_extraction import extract_character_names

    scenes = [_scene(1, ""), _scene(2, "")]
    assert extract_character_names(scenes) == []
