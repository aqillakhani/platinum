"""visual_prompts pipeline: per-scene Flux visual + negative prompt strings."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import ReviewStatus, Scene, Story
from platinum.models.story_bible import StoryBible
from platinum.pipeline.character_extraction import extract_character_names
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage
from platinum.utils.claude import (
    ClaudeProtocolError,
    ClaudeResult,
    Recorder,
)
from platinum.utils.claude import (
    call as claude_call,
)
from platinum.utils.prompts import (
    render_template,
)

logger = logging.getLogger(__name__)
MODEL = "claude-opus-4-7"


# S8.B.6 exposure guardrail. The prototype's scene-1 regression had Sonnet
# bake "blackness consumes the room... only void" into the visual_prompt
# while banning "candle, torch, flame, lantern, light source" in the
# negative_prompt — Flux complied too well and mean RGB collapsed to ~2.0.
# Two regexes catch this pattern:
#   * BANNED: lit-anchor words must NOT appear in negative_prompt.
#   * REQUIRED: at least one positive light word MUST appear in visual_prompt.
# Word boundaries prevent false matches ("delight" → no, "candlelight" → yes).
_BANNED_NEGATIVE_RE = re.compile(
    r"\b(candle\w*|torch\w*|flame\w*|lantern\w*|lamp\w*|fire\w*|light source)\b",
    re.IGNORECASE,
)
_REQUIRED_LIGHT_RE = re.compile(
    r"\b("
    r"candle\w*|"
    r"torch\w*|"
    r"flame\w*|"
    r"lantern\w*|"
    r"lamp\w*|"
    r"fire\w*|"
    r"sunlit|sunlight|daylight|moonlit|moonlight|"
    r"lit|lights?|"
    r"aglow|glow(?:ing)?|"
    r"illuminat\w*"
    r")\b",
    re.IGNORECASE,
)


class VisualPromptsRewriteViolation(ClaudeProtocolError):
    """A per-scene post-condition violation that the Stage retry layer
    can recover from by re-prompting Opus with the violation as
    ``deviation_feedback`` (S8.B.10).

    Distinct from base ``ClaudeProtocolError``: protocol errors (count
    mismatch, missing scene index, malformed tool_use) signal a deeper
    failure that retry can't fix and continue to raise the base class.

    Attributes:
        scene_index: 1-based scene index that violated.
        emitted_prompt: The visual_prompt Opus produced for that scene
            (used as ``current_prompt`` in the retry's deviation_feedback).
        feedback: Operator-style instruction telling Opus how to correct
            the violation (rendered into the j2 template's
            ``DEVIATION FEEDBACK`` block).
        additional_violations: Other per-scene violations from the same
            Opus call (S8.C: when Opus drifts on multiple scenes in one
            response, single-retry only fixed the first detected one
            and the next call introduced a new violation elsewhere).
            ``_zip_into_scenes`` packs every violation into one exception
            so ``VisualPromptsStage.run`` can feed all of them into
            ``deviation_feedback`` on the single retry.
    """

    def __init__(
        self,
        message: str,
        *,
        scene_index: int,
        emitted_prompt: str,
        feedback: str,
        additional_violations: list["VisualPromptsRewriteViolation"] | None = None,
    ) -> None:
        super().__init__(message)
        self.scene_index = scene_index
        self.emitted_prompt = emitted_prompt
        self.feedback = feedback
        self.additional_violations: list[VisualPromptsRewriteViolation] = (
            list(additional_violations) if additional_violations else []
        )

    def all_violations(self) -> list["VisualPromptsRewriteViolation"]:
        """Return every violation from the same Opus call, primary first.

        Convenience for the Stage retry layer so it doesn't have to
        special-case "primary vs. extras"."""
        return [self, *self.additional_violations]


VISUAL_PROMPTS_TOOL: dict[str, Any] = {
    "name": "submit_visual_prompts",
    "description": "Submit per-scene Flux visual + negative prompts.",
    "input_schema": {
        "type": "object",
        "required": ["scenes"],
        "properties": {
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "visual_prompt", "negative_prompt"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 1},
                        "visual_prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                        # S7.1.B2.4 -- optional structural fields. Declaring
                        # them here gives Claude an explicit contract to emit
                        # the keys; keeping them out of `required` preserves
                        # backwards compat with old recorded fixtures.
                        "composition_notes": {"type": "string"},
                        "character_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


def _load_characters(story_id: str, stories_dir: Path) -> dict[str, str]:
    """Load per-story characters dict from stories_dir/<story_id>/characters.json.

    Returns {} when the file does not exist. Phase A ships this file as
    optional and gitignored; Phase B B3.3 auto-extracts characters from
    the breakdown stage. The returned dict is fed to visual_prompts.j2's
    TRACK CHARACTERS section.
    """
    path = stories_dir / story_id / "characters.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_request(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    deviation_feedback: list | None = None,
    characters: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    bible = story.bible
    bible_by_index = {bs.index: bs for bs in bible.scenes} if bible else {}
    scenes_ctx: list[dict] = []
    for s in story.scenes:
        bs = bible_by_index.get(s.index)
        scene_bible: dict | None = None
        if bs is not None:
            scene_bible = {
                "narrative_beat": bs.narrative_beat,
                "hero_shot": bs.hero_shot,
                "visible_characters": list(bs.visible_characters),
                "gaze_map": dict(bs.gaze_map),
                "props_visible": list(bs.props_visible),
                "blocking": bs.blocking,
                "light_source": bs.light_source,
                "color_anchors": list(bs.color_anchors),
                "brightness_floor": bs.brightness_floor,
            }
        scenes_ctx.append({
            "index": s.index,
            "narration_text": s.narration_text,
            "bible": scene_bible,  # Always present (None when no bible) so
                                    # the StrictUndefined Jinja env's
                                    # ``{% if scene.bible %}`` resolves cleanly.
        })
    bible_ctx: dict | None = None
    if bible is not None:
        bible_ctx = {
            "world_genre_atmosphere": bible.world_genre_atmosphere,
            "character_continuity": {
                name: dict(sig) for name, sig in bible.character_continuity.items()
            },
            "environment_continuity": dict(bible.environment_continuity),
        }

    system = [
        {"type": "text", "text": render_template(
            prompts_dir=prompts_dir, track=story.track,
            name="system.j2", context={"track": track_cfg},
        )}
    ]
    messages = [
        {"role": "user", "content": render_template(
            prompts_dir=prompts_dir, track=story.track, name="visual_prompts.j2",
            context={
                "aesthetic": track_cfg["visual"]["aesthetic"],
                "palette": track_cfg["visual"]["palette"],
                "default_negative": track_cfg["visual"]["negative_prompt"],
                # S8.C period anchor + styling tail. Empty-string defaults keep
                # tracks that haven't opted in (and pre-S8.C test fixtures)
                # rendering cleanly under StrictUndefined -- the j2's period
                # block is `{% if period %}`-guarded.
                "period": track_cfg["visual"].get("period", ""),
                "period_styling": track_cfg["visual"].get("period_styling", ""),
                "scenes": scenes_ctx,
                "characters": characters or {},
                "deviation_feedback": deviation_feedback,
                "bible": bible_ctx,
            },
        )}
    ]
    return system, messages


def _zip_into_scenes(
    story_scenes: list[Scene], tool_input: dict,
    *, scene_filter: set[int] | None = None,
    bible: StoryBible | None = None,
) -> list[Scene]:
    """Mutate scenes with new visual_prompt + negative_prompt by index match.

    When scene_filter is set, only apply to scenes whose index is in the
    filter. For REJECTED scenes that get applied, also clear review_feedback
    and keyframe_path and flip status to REGENERATE.

    When ``bible`` is provided (S8.B.5), enforces a post-condition: every
    name in ``bible.scenes[i].visible_characters`` must appear (case
    insensitive) in the rewritten ``visual_prompt`` for scene i. Mismatch
    raises ``ClaudeProtocolError`` so the orchestrator's single-retry path
    can take a second swing.

    Raises ClaudeProtocolError on count mismatch or missing scene index.
    """
    raw = tool_input.get("scenes", [])
    if len(raw) != len(story_scenes):
        raise ClaudeProtocolError(
            f"visual_prompts count {len(raw)} != scene count {len(story_scenes)}"
        )
    bible_by_index = (
        {bs.index: bs for bs in bible.scenes} if bible is not None else {}
    )
    by_index = {item["index"]: item for item in raw}

    # Pass 1: validate every in-scope scene; collect all violations from the
    # same Opus response so the retry layer can feed them all back at once.
    # S8.C verify discovered that single-violation raise-on-first masked
    # multi-scene drift -- the Stage's single-retry path fixed scene 10's
    # banned-neg, then scene 7's missing-light surfaced on the retry and
    # propagated as a hard failure. Collect-all-then-raise lets one retry
    # round address every violation Opus emitted.
    violations: list[VisualPromptsRewriteViolation] = []
    for scene in story_scenes:
        item = by_index.get(scene.index)
        if item is None:
            raise ClaudeProtocolError(
                f"visual_prompts response missing scene index {scene.index}"
            )
        if scene_filter is not None and scene.index not in scene_filter:
            continue
        bs = bible_by_index.get(scene.index)
        # Bible post-condition: every visible character name (case insensitive)
        # must appear in the rewritten visual_prompt. Catches the prototype's
        # scene-2 character-drop regression before it corrupts the story.
        if bs is not None and bs.visible_characters:
            vp_lower = item["visual_prompt"].lower()
            missing = [
                name for name in bs.visible_characters
                if name.lower() not in vp_lower
            ]
            if missing:
                violations.append(VisualPromptsRewriteViolation(
                    f"visual_prompt for scene {scene.index} missing required "
                    f"character(s) {missing}; expected from bible "
                    f"visible_characters={bs.visible_characters}",
                    scene_index=scene.index,
                    emitted_prompt=item["visual_prompt"],
                    feedback=(
                        f"Your previous visual_prompt was missing required "
                        f"character(s) {missing}. The story bible's "
                        f"visible_characters declares {list(bs.visible_characters)} "
                        f"as required for scene {scene.index}. Re-emit the "
                        f"visual_prompt naming each character explicitly "
                        f"(use the character_continuity description from the bible)."
                    ),
                ))
                continue  # one violation per scene; skip the exposure checks
        # S8.B.6 exposure guardrail. Only enforced when bible is present --
        # the bible-required path is the one that ships the directive
        # "negative_prompt MUST NOT exclude any of: candle, torch, flame,
        # lantern, lamp, fire, light source" (S8.B.9 expanded). Prevents the
        # prototype's scene-1 regression from corrupting the story.
        if bs is not None:
            banned = _BANNED_NEGATIVE_RE.findall(item["negative_prompt"])
            if banned:
                violations.append(VisualPromptsRewriteViolation(
                    f"visual_prompts scene {scene.index}: negative_prompt bans "
                    f"lit anchor(s) {banned}; Flux needs these as light sources. "
                    f"Remove them from negative_prompt.",
                    scene_index=scene.index,
                    emitted_prompt=item["visual_prompt"],
                    feedback=(
                        f"Your previous negative_prompt was: "
                        f"'{item['negative_prompt']}'. It contained banned "
                        f"light-anchor word(s) {banned}, which Flux needs as "
                        f"light sources. Remove ALL forms of {banned} from the "
                        f"negative_prompt -- these are case-insensitive word "
                        f"stems, so 'lamps', 'lamplight', 'firelight' etc. all "
                        f"trip the gate. If you need to forbid a related "
                        f"concept, phrase it without these stems (e.g. 'bright "
                        f"daylight' instead of 'no overhead lamp glow')."
                    ),
                ))
                continue
            if not _REQUIRED_LIGHT_RE.search(item["visual_prompt"]):
                violations.append(VisualPromptsRewriteViolation(
                    f"visual_prompts scene {scene.index}: visual_prompt has no "
                    f"named light source. Add at least one of: candle, torch, "
                    f"lantern, lamp, fire, sun, daylight, moonlight, lit, glow.",
                    scene_index=scene.index,
                    emitted_prompt=item["visual_prompt"],
                    feedback=(
                        f"Your previous visual_prompt had no positive light "
                        f"word. Add at least one named lit anchor: candle, "
                        f"torch, lantern, lamp, fire, daylight, moonlight, "
                        f"glow, lit, illuminated. The bible declares scene "
                        f"{scene.index}'s light_source as: '{bs.light_source}'; "
                        f"lead with that."
                    ),
                ))
                continue

    if violations:
        primary = violations[0]
        primary.additional_violations = violations[1:]
        raise primary

    # Pass 2: all clean -- mutate scenes in place. This is now safe because
    # validation has already inspected every in-scope scene; we never leave
    # the story in a half-applied state.
    out: list[Scene] = []
    for scene in story_scenes:
        item = by_index[scene.index]
        if scene_filter is not None and scene.index not in scene_filter:
            continue
        scene.visual_prompt = item["visual_prompt"]
        scene.negative_prompt = item["negative_prompt"]
        # S7.1.B2.3 -- optional composition_notes + character_refs from the new
        # template. Default to leaving Scene fields untouched when the keys are
        # absent so old recorded fixtures (and tracks that don't ask for these)
        # keep round-tripping.
        if "composition_notes" in item:
            scene.composition_notes = item["composition_notes"]
        if "character_refs" in item:
            scene.character_refs = list(item["character_refs"])
        if scene.review_status == ReviewStatus.REJECTED:
            scene.review_status = ReviewStatus.REGENERATE
            scene.review_feedback = None
            scene.keyframe_path = None
        out.append(scene)
    return out


async def visual_prompts(
    *, story: Story, track_cfg: dict, prompts_dir: Path,
    db_path: Path, recorder: Recorder | None = None,
    scene_filter: set[int] | None = None,
    deviation_feedback: list | None = None,
    stories_dir: Path | None = None,
    characters: dict[str, str] | None = None,
) -> tuple[list[Scene], ClaudeResult]:
    """Run the visual_prompts call. Mutates story.scenes in place
    (sets visual_prompt and negative_prompt per scene) and returns it
    alongside the ClaudeResult. Raises if story.scenes is empty or
    if the response count/indexes don't match story.scenes.

    Characters resolution (S7.1.B3.3) -- precedence:
      1. Explicit `characters` kwarg (highest). Tests and CLI overrides
         use this to pin a specific cast.
      2. stories_dir/<story.id>/characters.json on disk. User-authored
         overrides; empty dict if file missing.
      3. Auto-extracted from scene narration via
         extract_character_names(); names paired with placeholder
         description "(reference pending)".

    The resulting dict is threaded into visual_prompts.j2's TRACK
    CHARACTERS section so Claude has a closed vocabulary when assigning
    each scene's character_refs list.
    """
    if not story.scenes:
        raise RuntimeError(
            f"visual_prompts requires scene_breakdown to have populated story.scenes "
            f"first (story={story.id})."
        )
    # S8.B.5: when the track opts into the story_bible pre-pass, the rewriter
    # is no longer allowed to run on narration alone. Fail fast with a clear
    # pointer at the bible command rather than silently producing prompts that
    # drift from the bible's directives.
    if track_cfg.get("story_bible", {}).get("enabled", False) and story.bible is None:
        raise RuntimeError(
            f"visual_prompts requires story.bible for track {story.track!r} "
            f"(story_bible.enabled=true). Run "
            f"`platinum bible {story.id}` first."
        )
    if characters is None:
        from_disk = (
            _load_characters(story.id, stories_dir) if stories_dir else {}
        )
        if from_disk:
            characters = from_disk
        else:
            names = extract_character_names(story.scenes)
            characters = {name: "(reference pending)" for name in names}
    system, messages = _build_request(
        story=story, track_cfg=track_cfg, prompts_dir=prompts_dir,
        deviation_feedback=deviation_feedback,
        characters=characters,
    )
    # S8.B.8: bible-aware prompts on full-length stories (16+ scenes) blew past
    # the 8192 default ceiling on the Cask verify — Opus returned an empty
    # scenes array because the tool call was truncated mid-flight. Mirror the
    # story_bible.max_tokens pattern: per-track override with 8192 fallback so
    # legacy tracks/short stories keep their prior request size.
    max_tokens = int(
        track_cfg.get("visual_prompts", {}).get("max_tokens", 8192)
    )
    result = await claude_call(
        model=MODEL, system=system, messages=messages, tool=VISUAL_PROMPTS_TOOL,
        story_id=story.id, stage="visual_prompts",
        db_path=db_path, recorder=recorder,
        max_tokens=max_tokens,
    )
    scenes = _zip_into_scenes(
        story.scenes, result.tool_input,
        scene_filter=scene_filter, bible=story.bible,
    )
    return scenes, result


class VisualPromptsStage(Stage):
    name: ClassVar[str] = "visual_prompts"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        if not story.scenes:
            raise RuntimeError(
                f"visual_prompts requires scene_breakdown completed first (story={story.id})."
            )
        track_cfg = ctx.config.track(story.track)
        recorder = ctx.config.settings.get("test", {}).get("claude_recorder")
        runtime = ctx.config.settings.get("runtime", {})
        scene_filter_raw = runtime.get("scene_filter")
        scene_filter = set(scene_filter_raw) if scene_filter_raw is not None else None
        runtime_feedback = runtime.get("deviation_feedback")
        # B3.3: tests / CLI can pin a specific cast via runtime config; None
        # falls through to disk -> auto-extract precedence in visual_prompts().
        characters_override = runtime.get("characters")

        async def _call(deviation_feedback: list | None) -> ClaudeResult:
            _, result = await visual_prompts(
                story=story, track_cfg=track_cfg,
                prompts_dir=ctx.config.prompts_dir,
                db_path=ctx.db_path, recorder=recorder,
                scene_filter=scene_filter,
                deviation_feedback=deviation_feedback,
                stories_dir=ctx.config.stories_dir,
                characters=characters_override,
            )
            return result

        # S8.B.10 single-retry-with-feedback (S8.C extended to multi-violation).
        # Bible-required tracks have post-condition guardrails on the rewriter
        # response (visible_chars present, no banned light tokens in
        # negative_prompt, at least one positive light word in visual_prompt).
        # Opus drifts despite the j2 directive: S8.B verify saw "lamp" in
        # scene 5, then "flame" in scene 6 on a clean second attempt; S8.C
        # verify (cask, 16 scenes) saw two simultaneous violations -- scene 10
        # banned "torches" in negative_prompt + scene 7 emitted no light word
        # in visual_prompt -- and the prior single-violation retry path fixed
        # only scene 10 before scene 7 surfaced and propagated.
        # ``_zip_into_scenes`` now packs every per-scene violation from one
        # Opus call into a single exception via ``additional_violations``.
        # Catch it, build a deviation_feedback entry per violation, retry
        # once. Single retry only -- a second response with any violation
        # propagates so the operator can intervene. Plain
        # ``ClaudeProtocolError`` (count mismatch, missing scene index) is
        # NOT retried -- those signal a deeper protocol issue retry can't fix.
        try:
            claude_result = await _call(runtime_feedback)
        except VisualPromptsRewriteViolation as exc:
            retry_feedback = list(runtime_feedback or [])
            for v in exc.all_violations():
                retry_feedback.append({
                    "index": v.scene_index,
                    "current_prompt": v.emitted_prompt,
                    "feedback": v.feedback,
                })
            claude_result = await _call(retry_feedback)

        return {
            "model": claude_result.usage.model,
            "input_tokens": claude_result.usage.input_tokens,
            "output_tokens": claude_result.usage.output_tokens,
            "cache_read_input_tokens": claude_result.usage.cache_read_input_tokens,
            "cost_usd": claude_result.usage.cost_usd,
        }
