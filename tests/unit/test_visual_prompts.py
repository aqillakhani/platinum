"""Unit tests for pipeline/visual_prompts.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from platinum.models.story import Adapted, Scene, Source, Story


def _track() -> dict:
    track_path = (
        Path(__file__).resolve().parents[2]
        / "config" / "tracks" / "atmospheric_horror.yaml"
    )
    return yaml.safe_load(track_path.read_text(encoding="utf-8"))["track"]


def _story_with_scenes(n: int = 4) -> Story:
    s = Story(
        id="story_vp", track="atmospheric_horror",
        source=Source(type="g", url="x", title="t", author="a",
                      raw_text="r", fetched_at=datetime(2026, 4, 25), license="PD-US"),
    )
    s.adapted = Adapted(title="t", synopsis="s", narration_script="x",
                         estimated_duration_seconds=600.0, tone_notes="n",
                         arc={"setup":"","rising":"","climax":"","resolution":""})
    s.scenes = [Scene(id=f"scene_{i:03d}", index=i, narration_text=f"text {i}")
                for i in range(1, n + 1)]
    return s


def _synth_response(n: int) -> dict:
    return {
        "id": "x",
        "content": [{"type": "tool_use", "name": "submit_visual_prompts", "input": {
            "scenes": [
                {"index": i, "visual_prompt": f"vp{i}", "negative_prompt": f"np{i}"}
                for i in range(1, n + 1)
            ],
        }}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


def _render_visual_prompts(
    *,
    characters: dict[str, str] | None = None,
    scenes: list[dict] | None = None,
    deviation_feedback: list | None = None,
) -> str:
    from platinum.utils.prompts import render_template

    track = _track()
    return render_template(
        prompts_dir=PROMPTS_DIR,
        track="atmospheric_horror",
        name="visual_prompts.j2",
        context={
            "aesthetic": track["visual"]["aesthetic"],
            "palette": track["visual"]["palette"],
            "default_negative": track["visual"]["negative_prompt"],
            "scenes": scenes or [{"index": 1, "narration_text": "x"}],
            "characters": characters or {},
            "deviation_feedback": deviation_feedback,
        },
    )


def test_visual_prompts_template_includes_track_characters_section() -> None:
    rendered = _render_visual_prompts(
        characters={
            "Fortunato": "Italian gentleman in motley",
            "Montresor": "narrator in dark cloak",
        }
    )
    assert "TRACK CHARACTERS" in rendered
    assert "Fortunato: Italian gentleman in motley" in rendered
    assert "Montresor: narrator in dark cloak" in rendered


def test_visual_prompts_template_omits_track_characters_when_empty() -> None:
    rendered = _render_visual_prompts(characters={})
    # The CONVENTIONS prose mentions "TRACK CHARACTERS" by name; the actual
    # section header has a parenthetical we use as the unique marker.
    assert "TRACK CHARACTERS (recurring" not in rendered


def test_visual_prompts_template_lists_camera_framing_tokens() -> None:
    rendered = _render_visual_prompts()
    for token in [
        "extreme close-up",
        "close-up",
        "medium shot",
        "medium-wide shot",
        "wide shot",
        "over-the-shoulder shot",
        "low-angle shot",
        "high-angle shot",
    ]:
        assert token in rendered


def test_visual_prompts_template_states_required_fields() -> None:
    rendered = _render_visual_prompts()
    assert "composition_notes" in rendered
    assert "character_refs" in rendered


def test_visual_prompts_template_retains_darkness_density_caps() -> None:
    rendered = _render_visual_prompts()
    assert "DARKNESS DENSITY CAPS" in rendered
    assert "lit edge" in rendered


def test_build_request_passes_characters_dict_to_template() -> None:
    from platinum.pipeline.visual_prompts import _build_request

    story = _story_with_scenes(2)
    _, messages = _build_request(
        story=story,
        track_cfg=_track(),
        prompts_dir=PROMPTS_DIR,
        characters={"Fortunato": "Italian gentleman in motley"},
    )
    user_text = messages[0]["content"]
    assert "TRACK CHARACTERS (recurring" in user_text
    assert "Fortunato: Italian gentleman in motley" in user_text


def test_build_request_omits_characters_section_when_unset() -> None:
    from platinum.pipeline.visual_prompts import _build_request

    story = _story_with_scenes(2)
    _, messages = _build_request(
        story=story,
        track_cfg=_track(),
        prompts_dir=PROMPTS_DIR,
    )
    user_text = messages[0]["content"]
    assert "TRACK CHARACTERS (recurring" not in user_text


def test_load_characters_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    from platinum.pipeline.visual_prompts import _load_characters

    assert _load_characters("nonexistent_story", tmp_path) == {}


def test_load_characters_reads_json_when_file_exists(tmp_path: Path) -> None:
    from platinum.pipeline.visual_prompts import _load_characters

    (tmp_path / "story_x").mkdir()
    (tmp_path / "story_x" / "characters.json").write_text(
        json.dumps({"Fortunato": "Italian gentleman", "Montresor": "narrator"}),
        encoding="utf-8",
    )
    assert _load_characters("story_x", tmp_path) == {
        "Fortunato": "Italian gentleman",
        "Montresor": "narrator",
    }


@pytest.mark.asyncio
async def test_visual_prompts_uses_stories_dir_to_load_characters(tmp_path: Path) -> None:
    """When stories_dir is provided, visual_prompts() loads characters and threads them through."""
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(2)

    stories_dir = tmp_path / "stories"
    (stories_dir / story.id).mkdir(parents=True)
    (stories_dir / story.id / "characters.json").write_text(
        json.dumps({"Fortunato": "Italian gentleman in motley"}), encoding="utf-8"
    )

    captured_request: dict = {}

    async def synth(req):
        captured_request["request"] = req
        return _synth_response(2)

    await visual_prompts(
        story=story,
        track_cfg=_track(),
        prompts_dir=PROMPTS_DIR,
        db_path=db_path,
        recorder=synth,
        stories_dir=stories_dir,
    )
    user_text = captured_request["request"]["messages"][0]["content"]
    assert "TRACK CHARACTERS (recurring" in user_text
    assert "Fortunato: Italian gentleman in motley" in user_text


@pytest.mark.asyncio
async def test_visual_prompts_zips_into_scenes_by_index(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(4)

    async def synth(req):
        # Return out-of-order to exercise zip-by-index
        r = _synth_response(4)
        r["content"][0]["input"]["scenes"].reverse()
        return r

    scenes, _ = await visual_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
        db_path=db_path, recorder=synth,
    )
    assert [(s.index, s.visual_prompt, s.negative_prompt) for s in scenes] == [
        (1, "vp1", "np1"), (2, "vp2", "np2"), (3, "vp3", "np3"), (4, "vp4", "np4"),
    ]


@pytest.mark.asyncio
async def test_visual_prompts_count_must_match_scene_count(tmp_path) -> None:
    from platinum.models.db import create_all
    from platinum.pipeline.visual_prompts import visual_prompts
    from platinum.utils.claude import ClaudeProtocolError

    db_path = tmp_path / "p.db"
    create_all(db_path)
    story = _story_with_scenes(4)

    async def synth(req):
        return _synth_response(3)  # only 3 prompts for 4 scenes

    with pytest.raises(ClaudeProtocolError, match="count"):
        await visual_prompts(
            story=story, track_cfg=_track(),
            prompts_dir=Path(__file__).resolve().parents[2] / "config" / "prompts",
            db_path=db_path, recorder=synth,
        )
