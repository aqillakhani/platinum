"""Bible-aware visual_prompts rendering + post-condition (S8.B.5).

When the story has a populated ``StoryBible``, the visual_prompts request
threads the bible into the rewriter's user message — world atmosphere,
character_continuity signatures, per-scene hero_shot/visible_characters/
props_visible/brightness_floor — and the lead instruction tells Sonnet
to bake the bible's directives into every prompt.

After the rewriter responds, ``_zip_into_scenes`` enforces a post-condition:
every name in ``bible.scenes[i].visible_characters`` MUST appear (case
insensitive) in the rewritten ``visual_prompt`` for scene i. Mismatch
raises ``ClaudeProtocolError`` so the orchestrator's existing single-retry
path can take a second swing.

Back-compat: ``story.bible`` is None (or absent) → template renders
unchanged, post-condition skipped, every previously-passing test still
passes.
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


def _story_with_scenes(n: int = 4) -> Story:
    s = Story(
        id="story_vp_bible", track="atmospheric_horror",
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


def _bible(
    scene_indices: list[int],
    *, visible_per_scene: dict[int, list[str]] | None = None,
) -> StoryBible:
    """Build a StoryBible covering the given scene indices."""
    visible_per_scene = visible_per_scene or {}
    return StoryBible(
        world_genre_atmosphere=(
            "Late 18th-century Italian carnival night bleeding into "
            "subterranean catacombs."
        ),
        character_continuity={
            "Montresor": {
                "face": "lean noble, dark goatee, sharp cheekbones",
                "costume": "floor-length black wool cloak",
                "posture": "patient, calculating, hands clasped",
            },
            "Fortunato": {
                "face": "florid, ruddy cheeks, glassy drunken eyes",
                "costume": "tight parti-striped motley with conical bell cap",
                "posture": "swaying, leaning, gesturing broadly",
            },
        },
        environment_continuity={
            "palazzo_study": "ash-grey stone walls, oak desk, beeswax candle",
            "catacombs": "low barrel-vault tunnels, nitre-glittering walls",
        },
        scenes=[
            BibleScene(
                index=i,
                narrative_beat=f"beat {i}",
                hero_shot=f"two-shot, medium, eye-level for scene {i}",
                visible_characters=visible_per_scene.get(i, ["Montresor"]),
                gaze_map={"Montresor": "off-camera"},
                props_visible=["candle", "trowel"],
                blocking="Montresor foreground-left, still",
                light_source="single beeswax candle, foreground right",
                color_anchors=["black cloak", "amber candle flame"],
                brightness_floor="low",
            )
            for i in scene_indices
        ],
    )


def _synth_response(n: int, *, visual_prompts: list[str] | None = None) -> dict:
    """Synthetic submit_visual_prompts response with N scenes.

    visual_prompts overrides the per-scene prompt strings; default is
    "vp{i} mentioning Montresor and Fortunato" so the post-condition
    finds both protagonists by default.
    """
    if visual_prompts is None:
        visual_prompts = [
            f"vp{i} mentioning Montresor and Fortunato in candlelit study"
            for i in range(1, n + 1)
        ]
    return {
        "id": "x",
        "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
            "scenes": [
                {"index": i, "visual_prompt": visual_prompts[i - 1],
                 "negative_prompt": f"np{i}"}
                for i in range(1, n + 1)
            ],
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


# ---------- Template renders bible context ------------------------------


def test_template_renders_bible_context_when_bible_present() -> None:
    """When _build_request is called with a populated story.bible, the
    rendered prompt contains the world atmosphere, character continuity
    signatures, per-scene hero_shot, visible_characters, props_visible,
    and brightness_floor."""
    from platinum.pipeline.visual_prompts import _build_request

    story = _story_with_scenes(2)
    story.bible = _bible(
        [1, 2],
        visible_per_scene={1: ["Montresor"], 2: ["Montresor", "Fortunato"]},
    )

    _, messages = _build_request(
        story=story, track_cfg=_track(), prompts_dir=PROMPTS_DIR,
    )
    user_text = messages[0]["content"]

    # World atmosphere appears.
    assert "carnival night" in user_text.lower()

    # Character continuity signatures threaded in.
    assert "Montresor" in user_text
    assert "lean noble" in user_text
    assert "Fortunato" in user_text
    assert "parti-striped motley" in user_text

    # Per-scene hero_shot rendered.
    assert "two-shot, medium, eye-level for scene 1" in user_text
    assert "two-shot, medium, eye-level for scene 2" in user_text

    # Per-scene visible_characters and props.
    assert "candle" in user_text  # prop
    assert "trowel" in user_text  # prop
    assert "single beeswax candle" in user_text  # light_source

    # Brightness floor rendered.
    assert "brightness floor" in user_text.lower() or "low" in user_text


def test_template_omits_bible_context_when_bible_absent() -> None:
    """Back-compat: when story.bible is None, the template renders the
    pre-S8.B prompt unchanged — no STORY BIBLE CONTEXT block."""
    from platinum.pipeline.visual_prompts import _build_request

    story = _story_with_scenes(2)
    assert story.bible is None

    _, messages = _build_request(
        story=story, track_cfg=_track(), prompts_dir=PROMPTS_DIR,
    )
    user_text = messages[0]["content"]
    assert "STORY BIBLE CONTEXT" not in user_text
    # The pre-existing scenes block still renders narration.
    assert "text 1" in user_text


def test_template_lead_instruction_when_bible_present() -> None:
    """The bible-aware lead instruction asks Sonnet to (1) lead with
    hero_shot, (2) name every visible character, (3) include every prop,
    (4) honor brightness_floor, (5) keep negative prompts free of the
    bible's named light anchors."""
    from platinum.pipeline.visual_prompts import _build_request

    story = _story_with_scenes(2)
    story.bible = _bible([1, 2])
    _, messages = _build_request(
        story=story, track_cfg=_track(), prompts_dir=PROMPTS_DIR,
    )
    user_text = messages[0]["content"]
    lower = user_text.lower()
    assert "hero_shot" in lower or "hero shot" in lower
    assert "visible_characters" in lower or "visible characters" in lower
    assert "brightness_floor" in lower or "brightness floor" in lower


# ---------- _zip_into_scenes visible-characters post-condition ----------


@pytest.mark.asyncio
async def test_zip_post_condition_passes_when_all_visible_characters_present(
    tmp_path: Path,
) -> None:
    """Post-condition holds when every name in bible.scenes[i].visible_characters
    appears (case insensitive) in the rewritten visual_prompt."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)
    story.bible = _bible([1, 2], visible_per_scene={
        1: ["Montresor"],
        2: ["Montresor", "Fortunato"],
    })

    async def synth(req):
        return _synth_response(2, visual_prompts=[
            "Montresor in candlelit study",  # scene 1 — has Montresor
            "Montresor and Fortunato meeting in carnival",  # scene 2 — both
        ])

    scenes, _ = await visual_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert scenes[0].visual_prompt == "Montresor in candlelit study"
    assert scenes[1].visual_prompt == "Montresor and Fortunato meeting in carnival"


@pytest.mark.asyncio
async def test_zip_post_condition_raises_when_visible_character_missing(
    tmp_path: Path,
) -> None:
    """Sonnet drops a character → ClaudeProtocolError citing the missing
    name and the scene index. This is the prototype's scene-2 regression
    path; the post-condition prevents it from corrupting the story."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)
    story.bible = _bible([1, 2], visible_per_scene={
        1: ["Montresor"],
        2: ["Montresor", "Fortunato"],
    })

    async def synth(req):
        # Scene 2 drops Fortunato — the rewriter regression we're guarding.
        return _synth_response(2, visual_prompts=[
            "Montresor in candlelit study",
            "Montresor alone with a wine bottle",
        ])

    with pytest.raises(ClaudeProtocolError, match=r"scene 2.*Fortunato|Fortunato.*scene 2"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
        )


@pytest.mark.asyncio
async def test_zip_post_condition_is_case_insensitive(tmp_path: Path) -> None:
    """Sonnet sometimes emits "MONTRESOR" or "fortunato" — the post-condition
    must match case-insensitively, otherwise it'll false-positive on
    legitimate output."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(1)
    story.bible = _bible([1], visible_per_scene={1: ["Montresor", "Fortunato"]})

    async def synth(req):
        return _synth_response(1, visual_prompts=[
            "MONTRESOR and FORTUNATO at the carnival",  # all caps
        ])

    scenes, _ = await visual_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert "MONTRESOR" in scenes[0].visual_prompt


@pytest.mark.asyncio
async def test_zip_post_condition_skipped_when_bible_absent(
    tmp_path: Path,
) -> None:
    """Back-compat: stories whose track does NOT enable story_bible run
    the rewriter on narration alone (pre-S8.B path) with no post-condition
    — every test that worked before S8.B still works on tracks that
    haven't opted into the bible pre-pass."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)
    assert story.bible is None

    track_cfg = _track()
    # Simulate a track that hasn't opted into the bible pre-pass.
    track_cfg["story_bible"]["enabled"] = False

    async def synth(req):
        # No bible → no character requirement; arbitrary prompts ok.
        return _synth_response(2, visual_prompts=["foo", "bar"])

    scenes, _ = await visual_prompts(
        story=story, track_cfg=track_cfg,
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert scenes[0].visual_prompt == "foo"
    assert scenes[1].visual_prompt == "bar"


# ---------- Bible-required precondition --------------------------------


@pytest.mark.asyncio
async def test_visual_prompts_raises_when_bible_required_but_missing(
    tmp_path: Path,
) -> None:
    """If track_cfg.story_bible.enabled is true but story.bible is None,
    visual_prompts() fails fast with a message pointing at `platinum bible`.

    Edge case: a user manually clears story.bible and re-runs visual_prompts
    via --rerun-rejected. The orchestrator's normal flow (adapt) prepends
    the bible stage, so this guard fires only on a direct call after
    user intervention."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)
    assert story.bible is None

    track_cfg = _track()
    # Force the bible-required mode.
    track_cfg.setdefault("story_bible", {})["enabled"] = True

    async def synth(req):  # would never be called — guard fires first
        raise AssertionError("recorder invoked despite missing bible")

    with pytest.raises(RuntimeError, match=r"bible|platinum bible"):
        await visual_prompts(
            story=story, track_cfg=track_cfg,
            prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
        )
