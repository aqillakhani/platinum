"""Heuristic discovery of recurring character names from scene narration.

The visual_prompts stage threads a `characters` dict (name -> short
description) through visual_prompts.j2 so Claude has an explicit list to
draw from when assigning per-scene character_refs. When the user has not
written an explicit characters.json on disk, we fall back to this
heuristic: walk each scene's narration_text, collect capitalized
multi-word phrases that look like proper nouns, drop common
sentence-start English words, and keep names that appear in two or more
distinct scenes.

The heuristic is best-effort. Place names repeated across scenes will
look like characters; one-off character mentions (a single appearance)
are filtered as transient. The user is the source of truth -- review UI
+ characters.json override anything this module produces.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from platinum.models.story import Scene

# English single-word capitalisations that frequently start sentences but
# are not proper nouns. When a multi-word match begins with one of these,
# we strip the leading token and re-evaluate the rest.
_NON_NAME_CAPITALS: frozenset[str] = frozenset({
    "The", "A", "An", "He", "She", "It", "They", "We", "I", "You",
    "There", "His", "Her", "My", "Your", "Their", "Our",
    "This", "That", "These", "Those", "But", "And", "Or", "If", "When",
    "While", "After", "Before", "Then", "Now", "Once", "Yet", "So",
    "In", "On", "At", "To", "For", "Of", "With", "From", "By", "As",
    "Up", "Down", "Out", "Over", "Under", "Through", "Into", "Onto",
    "Be", "Been", "Being",
})

# Word boundaries on both sides; capitalised single word optionally
# followed by additional capitalised words separated by single spaces.
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def extract_character_names(scenes: Iterable[Scene]) -> list[str]:
    """Return sorted names that appear in 2+ distinct scenes.

    Rules:
      - Capitalised words (and runs of capitalised words separated by
        single spaces) are candidate names.
      - If the leading token of a multi-word match is a common English
        sentence-starter (article / pronoun / preposition), strip it;
        the remainder, if still capitalised, becomes the candidate.
      - A name must appear in at least two distinct scene indices to
        survive the recurrence filter.
    """
    name_to_scene_indices: dict[str, set[int]] = {}
    for scene in scenes:
        text = scene.narration_text or ""
        names_in_scene: set[str] = set()
        for match in _NAME_RE.finditer(text):
            phrase = match.group(0)
            first = phrase.split(None, 1)[0]
            if first in _NON_NAME_CAPITALS:
                rest = phrase.split(None, 1)
                if len(rest) <= 1:
                    continue
                phrase = rest[1]
            names_in_scene.add(phrase)
        for name in names_in_scene:
            name_to_scene_indices.setdefault(name, set()).add(scene.index)
    return sorted(
        name for name, idxs in name_to_scene_indices.items() if len(idxs) >= 2
    )
