"""Story bible -- whole-story narrative directive consumed by visual_prompts.

S8.B.1 introduces an optional pre-pass stage that reads the entire story
once and produces a structured directive. The visual_prompts stage then
rewrites each scene's prompt grounded in that directive — the goal is to
fix the S8.A content-fidelity gap (right characters, right props, right
beat) without re-running visual_prompts in isolation per scene.

This module owns the data model only. The pipeline stage that produces
this object lives at ``src/platinum/pipeline/story_bible.py`` (added in
S8.B.2 / S8.B.3). It is round-tripped through ``Story.bible`` via
``Story.to_dict`` / ``Story.from_dict``.

Design ref: docs/plans/2026-05-02-session-8.B-story-bible-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BibleScene:
    """Per-scene narrative directive.

    All fields are required: a downstream rewriter relies on every key
    being present. The bible producer (Opus 4.7 via tool-use) is told to
    fill every slot.

    Field semantics:
      * narrative_beat: one short sentence naming the moment (NOT the
        narration text).
      * hero_shot: cinematic frame description (lens / angle / distance).
      * visible_characters: every named character in frame. The
        downstream visual_prompts rewriter MUST mention each by name; a
        post-condition check enforces this.
      * gaze_map: per-character gaze direction (where the character is
        looking). Driving signal for the "no eye contact" complaint.
      * props_visible: explicit objects in frame (prevents Flux dropping
        story-critical props like the bells, trowel, mask).
      * blocking: spatial layout ("foreground left", "mid-ground").
      * light_source: named light source(s). At least one must be
        non-empty — the brightness guardrail checks the rewritten prompt
        does not ban this source.
      * color_anchors: signature colors that must persist (motley red,
        Montresor's black cloak).
      * brightness_floor: "low" | "medium" | "high". Drives the
        rewriter's exposure guardrail. "low" = chiaroscuro permitted;
        "medium" / "high" = strict no-banned-light.
    """
    index: int
    narrative_beat: str
    hero_shot: str
    visible_characters: list[str]
    gaze_map: dict[str, str]
    props_visible: list[str]
    blocking: str
    light_source: str
    color_anchors: list[str]
    brightness_floor: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "narrative_beat": self.narrative_beat,
            "hero_shot": self.hero_shot,
            "visible_characters": list(self.visible_characters),
            "gaze_map": dict(self.gaze_map),
            "props_visible": list(self.props_visible),
            "blocking": self.blocking,
            "light_source": self.light_source,
            "color_anchors": list(self.color_anchors),
            "brightness_floor": self.brightness_floor,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BibleScene:
        return cls(
            index=int(d["index"]),
            narrative_beat=d["narrative_beat"],
            hero_shot=d["hero_shot"],
            visible_characters=list(d.get("visible_characters", [])),
            gaze_map=dict(d.get("gaze_map", {})),
            props_visible=list(d.get("props_visible", [])),
            blocking=d.get("blocking", ""),
            light_source=d["light_source"],
            color_anchors=list(d.get("color_anchors", [])),
            brightness_floor=d["brightness_floor"],
        )


@dataclass(frozen=True)
class StoryBible:
    """Whole-story narrative directive.

    * world_genre_atmosphere: short paragraph framing the whole arc.
    * character_continuity: per-named-character persistent visual
      signature. Maps name -> {face, costume, posture}. The visual_prompts
      rewriter consults this so a character looks the same across scenes.
    * environment_continuity: per-named-locale persistent visual anchors
      (shared lighting, recurring props, wall textures). Maps locale ->
      one-paragraph description.
    * scenes: per-scene directives, indexed to align 1:1 with
      Story.scenes by ``BibleScene.index``.
    """
    world_genre_atmosphere: str
    character_continuity: dict[str, dict[str, str]] = field(default_factory=dict)
    environment_continuity: dict[str, str] = field(default_factory=dict)
    scenes: list[BibleScene] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_genre_atmosphere": self.world_genre_atmosphere,
            "character_continuity": {
                name: dict(traits) for name, traits in self.character_continuity.items()
            },
            "environment_continuity": dict(self.environment_continuity),
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoryBible:
        return cls(
            world_genre_atmosphere=d.get("world_genre_atmosphere", ""),
            character_continuity={
                name: dict(traits)
                for name, traits in d.get("character_continuity", {}).items()
            },
            environment_continuity=dict(d.get("environment_continuity", {})),
            scenes=[BibleScene.from_dict(s) for s in d.get("scenes", [])],
        )
