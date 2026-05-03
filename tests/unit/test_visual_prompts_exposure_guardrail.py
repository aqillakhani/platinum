"""Exposure guardrail post-condition for visual_prompts (S8.B.6).

The S8.B prototype's scene-1 regression banned every light element from
the negative prompt — Flux drove mean RGB to ~2.0 because the rewriter
took "atmospheric darkness" too literally and excluded its own anchors.

Two linter rules fire when a story bible is in play:

1. The rewritten ``negative_prompt`` MUST NOT contain any of the banned
   light tokens (case insensitive): candle, torch, flame, lantern,
   "light source", lamp, fire. These are the lit anchors Flux needs.

2. The rewritten ``visual_prompt`` MUST contain at least one positive
   light vocabulary word (case insensitive, word-boundary matched) so
   Flux has at least one explicit lit anchor to render.

When ``bible`` is None (track has not opted into the pre-pass), the
guardrail is a no-op — back-compat with the pre-S8.B rewriter path.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from platinum.models.story import Adapted, Scene, Source, Story
from platinum.models.story_bible import BibleScene, StoryBible

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


def _track() -> dict:
    track_path = (
        Path(__file__).resolve().parents[2]
        / "config" / "tracks" / "atmospheric_horror.yaml"
    )
    return yaml.safe_load(track_path.read_text(encoding="utf-8"))["track"]


def _story_with_scenes(n: int = 2) -> Story:
    s = Story(
        id="story_exposure", track="atmospheric_horror",
        source=Source(type="g", url="x", title="t", author="a",
                      raw_text="r", fetched_at=datetime(2026, 5, 2),
                      license="PD-US"),
    )
    s.adapted = Adapted(
        title="t", synopsis="s", narration_script="x",
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup": "", "rising": "", "climax": "", "resolution": ""},
    )
    s.scenes = [
        Scene(id=f"scene_{i:03d}", index=i, narration_text=f"text {i}")
        for i in range(1, n + 1)
    ]
    return s


def _bible_for(n_scenes: int) -> StoryBible:
    return StoryBible(
        world_genre_atmosphere="x",
        character_continuity={},
        environment_continuity={},
        scenes=[
            BibleScene(
                index=i, narrative_beat="b", hero_shot="m",
                visible_characters=[], gaze_map={},
                props_visible=[], blocking="c",
                light_source="single beeswax candle",
                color_anchors=[], brightness_floor="low",
            )
            for i in range(1, n_scenes + 1)
        ],
    )


def _synth_response(n: int, *, visual_prompts: list[str], negative_prompts: list[str]) -> dict:
    return {
        "id": "x",
        "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
            "scenes": [
                {"index": i,
                 "visual_prompt": visual_prompts[i - 1],
                 "negative_prompt": negative_prompts[i - 1]}
                for i in range(1, n + 1)
            ],
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


# ---------- Banned tokens in negative_prompt ----------------------------


@pytest.mark.parametrize("banned", [
    "candle", "torch", "flame", "lantern", "light source", "lamp", "fire",
    "Candle",  # case insensitive
    "TORCH",
])
@pytest.mark.asyncio
async def test_negative_prompt_banning_light_anchor_raises(
    tmp_path: Path, banned: str,
) -> None:
    """If the rewriter bans a lit anchor in negative_prompt, the linter
    raises ClaudeProtocolError — Flux needs those anchors."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(1)
    story.bible = _bible_for(1)

    async def synth(req):
        return _synth_response(
            1,
            visual_prompts=["candle in a dark room"],
            negative_prompts=[f"bright daylight, {banned}, neon"],
        )

    with pytest.raises(ClaudeProtocolError, match=r"negative_prompt|exposure"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
        )


# ---------- Required positive light word in visual_prompt --------------


@pytest.mark.asyncio
async def test_visual_prompt_with_no_light_word_raises(tmp_path: Path) -> None:
    """The visual_prompt must include at least one positive light
    vocabulary word so Flux has a lit anchor to render. A prompt that
    uses only darkness language (the prototype's scene-1 failure mode)
    raises so the orchestrator can retry."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(1)
    story.bible = _bible_for(1)

    async def synth(req):
        # Prototype scene-1 regression: "blackness consumes the room... only void"
        return _synth_response(
            1,
            visual_prompts=["blackness consumes the room, only void, deep shadow"],
            negative_prompts=["bright daylight"],
        )

    with pytest.raises(ClaudeProtocolError, match=r"visual_prompt|light"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
        )


# ---------- Happy path -------------------------------------------------


@pytest.mark.asyncio
async def test_clean_rewrite_passes_exposure_guardrail(tmp_path: Path) -> None:
    """Visual prompt with a named light source + clean negative passes."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)
    story.bible = _bible_for(2)

    async def synth(req):
        return _synth_response(
            2,
            visual_prompts=[
                "single beeswax candle catching the rim of the goblet",
                "amber torchlight from upper-left grazing the stone wall",
            ],
            negative_prompts=[
                "bright daylight, neon, modern technology",
                "bright daylight, anime, plastic",
            ],
        )

    scenes, _ = await visual_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert "candle" in scenes[0].visual_prompt.lower()
    assert "torchlight" in scenes[1].visual_prompt.lower()


# ---------- Word-boundary semantics ------------------------------------


@pytest.mark.asyncio
async def test_visual_prompt_word_match_uses_boundaries(tmp_path: Path) -> None:
    """The required-light-word check uses word boundaries so partial
    matches like "delight" or "highlights" don't satisfy the requirement
    (they're not lit anchors). A prompt that contains only "delight"
    should still raise."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(1)
    story.bible = _bible_for(1)

    async def synth(req):
        # "delight" contains "light" as a substring but not as a word.
        return _synth_response(
            1,
            visual_prompts=["a man's quiet delight, deep shadows everywhere"],
            negative_prompts=["bright daylight"],
        )

    with pytest.raises(ClaudeProtocolError, match=r"visual_prompt|light"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
        )


# ---------- Spec-vs-impl drift: directive must list every banned stem -----


def test_j2_directive_lists_all_banned_negative_stems() -> None:
    """The visual_prompts.j2 directive listing tokens forbidden in
    negative_prompt MUST mention every stem the post-condition regex
    bans. Otherwise Opus is judged on rules it wasn't given.

    Spec drift discovered S8.B verify (S8.B.9): the directive listed
    candle/torch/flame/lantern/'light source' but the regex also bans
    lamp\\w* and fire\\w*. Opus respected the directive and emitted
    "lamp" in scene 5's negative_prompt; the regex rejected the response;
    the verify run hit ClaudeProtocolError after a $0.87 Opus call.
    """
    from platinum.pipeline.visual_prompts import _BANNED_NEGATIVE_RE  # noqa: F401
    from platinum.utils.prompts import render_template

    bible_ctx = {
        "world_genre_atmosphere": "x",
        "character_continuity": {},
        "environment_continuity": {},
    }
    rendered = render_template(
        prompts_dir=PROMPTS_DIR,
        track="atmospheric_horror",
        name="visual_prompts.j2",
        context={
            "aesthetic": "a", "palette": "p", "default_negative": "n",
            "scenes": [], "characters": {}, "deviation_feedback": None,
            "bible": bible_ctx,
        },
    )
    directive_idx = rendered.find("negative_prompt MUST NOT exclude")
    assert directive_idx >= 0, (
        "Directive 'negative_prompt MUST NOT exclude' not found in rendered j2."
    )
    directive_line = rendered[directive_idx:directive_idx + 300]

    # Each stem in the regex post-condition must appear in the directive.
    # Regex bans: candle\\w*|torch\\w*|flame\\w*|lantern\\w*|lamp\\w*|fire\\w*|light source
    stems = ["candle", "torch", "flame", "lantern", "lamp", "fire", "light source"]
    missing = [s for s in stems if s not in directive_line.lower()]
    assert not missing, (
        f"j2 directive missing banned stems {missing}. Directive: "
        f"{directive_line!r}. Each stem in _BANNED_NEGATIVE_RE must be "
        f"explicitly listed so Opus knows what to avoid."
    )


# ---------- Back-compat: skipped when no bible -------------------------


@pytest.mark.asyncio
async def test_exposure_guardrail_skipped_when_bible_absent(tmp_path: Path) -> None:
    """Pre-S8.B path: if the track has no bible enabled and story.bible
    is None, the rewriter runs without the exposure guardrail."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(1)
    assert story.bible is None

    track_cfg = _track()
    track_cfg["story_bible"]["enabled"] = False  # disable for this test

    async def synth(req):
        # This would fail the guardrail if it ran (no light word, banned
        # candle in negative_prompt) — but the guardrail is gated on bible.
        return _synth_response(
            1,
            visual_prompts=["only void, deep shadow"],
            negative_prompts=["bright daylight, candle"],
        )

    scenes, _ = await visual_prompts(
        story=story, track_cfg=track_cfg,
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert scenes[0].visual_prompt == "only void, deep shadow"
