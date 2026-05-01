"""Unit tests for pipeline/motion_prompts.py (S8.A.4).

Mirrors test_content_check.py patterns: one Claude vision call per scene,
mock recorder returns shaped tool_use dict, stage iterates approved scenes
and writes scene.motion_prompt.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from platinum.models.story import Adapted, Scene, Source, Story

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "config" / "prompts"


def _track() -> dict:
    import yaml

    track_path = REPO_ROOT / "config" / "tracks" / "atmospheric_horror.yaml"
    return yaml.safe_load(track_path.read_text(encoding="utf-8"))["track"]


def _story_with_keyframes(tmp_path: Path, n: int = 2) -> Story:
    """Build a Story with n scenes, each with a real on-disk keyframe PNG.

    Uses tests._fixtures.make_synthetic_png to keep this hermetic — no
    repo image fixture committed.
    """
    from tests._fixtures import make_synthetic_png

    story = Story(
        id="story_mp_test",
        track="atmospheric_horror",
        source=Source(
            type="g", url="x", title="t", author="a", raw_text="r",
            fetched_at=datetime(2026, 5, 1), license="PD-US",
        ),
    )
    story.adapted = Adapted(
        title="t", synopsis="s", narration_script="x",
        estimated_duration_seconds=600.0, tone_notes="n",
        arc={"setup": "", "rising": "", "climax": "", "resolution": ""},
    )
    story.scenes = []
    for i in range(1, n + 1):
        kf = tmp_path / f"scene_{i:03d}.png"
        make_synthetic_png(kf, kind="grey", value=64 + i * 16)
        sc = Scene(
            id=f"scene_{i:03d}",
            index=i,
            narration_text=f"narration line {i}",
            visual_prompt=(
                f"medium shot, subject {i} performing action {i}, "
                "candlelit corridor, painterly cinematography, dim mood"
            ),
            keyframe_path=kf,
        )
        story.scenes.append(sc)
    return story


def _synth_motion_prompt_response(prompt: str = "slow dolly forward") -> dict:
    """Anthropic-API-shaped dict for a single submit_motion_prompt tool call."""
    return {
        "id": "msg_mp_synth",
        "content": [
            {
                "type": "tool_use",
                "name": "submit_motion_prompt",
                "input": {
                    "motion_prompt": prompt,
                    "rationale": "keyframe shows X; this motion continues from there",
                },
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


def test_motion_prompt_tool_has_required_motion_prompt_field() -> None:
    """The submit_motion_prompt tool requires motion_prompt; rationale optional."""
    from platinum.pipeline.motion_prompts import MOTION_PROMPT_TOOL

    assert MOTION_PROMPT_TOOL["name"] == "submit_motion_prompt"
    schema = MOTION_PROMPT_TOOL["input_schema"]
    assert "motion_prompt" in schema["required"]
    assert "rationale" not in schema.get("required", [])
    assert schema["properties"]["motion_prompt"]["type"] == "string"


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def test_build_request_includes_keyframe_image_block(tmp_path: Path) -> None:
    """The Claude request's user message must include a vision content
    block carrying the keyframe image (base64 PNG)."""
    from platinum.pipeline.motion_prompts import _build_request
    from tests._fixtures import make_synthetic_png

    kf = tmp_path / "keyframe.png"
    make_synthetic_png(kf, kind="grey", value=80)

    scene = Scene(
        id="scene_001", index=1, narration_text="x",
        visual_prompt="medium shot, subject in candlelight",
        keyframe_path=kf,
    )
    track_cfg = _track()
    system, messages = _build_request(
        scene=scene, track_cfg=track_cfg, prompts_dir=PROMPTS_DIR,
        story_track="atmospheric_horror",
    )
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["type"] == "text"
    assert isinstance(messages, list) and len(messages) == 1
    user_content = messages[0]["content"]
    assert isinstance(user_content, list)
    image_blocks = [b for b in user_content if b.get("type") == "image"]
    assert len(image_blocks) == 1, "expected exactly one image block (the keyframe)"
    assert image_blocks[0]["source"]["type"] == "base64"
    assert image_blocks[0]["source"]["media_type"] == "image/png"
    assert image_blocks[0]["source"]["data"], "image data must be non-empty"


def test_build_request_includes_visual_prompt_context(tmp_path: Path) -> None:
    """The user message must include the original visual_prompt as context
    (so Claude knows what action was intended) AND the scene narration."""
    from platinum.pipeline.motion_prompts import _build_request
    from tests._fixtures import make_synthetic_png

    kf = tmp_path / "k.png"
    make_synthetic_png(kf, kind="grey", value=64)

    scene = Scene(
        id="scene_005", index=5,
        narration_text="They descended into the catacombs.",
        visual_prompt="wide shot, two figures with torches on stone steps",
        keyframe_path=kf,
    )
    _, messages = _build_request(
        scene=scene, track_cfg=_track(), prompts_dir=PROMPTS_DIR,
        story_track="atmospheric_horror",
    )
    text_blocks = [
        b for b in messages[0]["content"] if b.get("type") == "text"
    ]
    rendered = "\n".join(b["text"] for b in text_blocks)
    assert "two figures with torches" in rendered, "visual_prompt context missing"
    assert "catacombs" in rendered, "narration_text missing"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_response_returns_motion_prompt_string() -> None:
    """_parse_response extracts motion_prompt from tool_input."""
    from platinum.pipeline.motion_prompts import _parse_response

    tool_input = {
        "motion_prompt": "slow dolly forward as candles flicker",
        "rationale": "subject already in pose",
    }
    out = _parse_response(tool_input, scene_index=5)
    assert out == "slow dolly forward as candles flicker"


def test_parse_response_raises_protocol_error_on_missing_field() -> None:
    """Missing motion_prompt = ClaudeProtocolError (raises loud, not silent)."""
    from platinum.pipeline.motion_prompts import _parse_response
    from platinum.utils.claude import ClaudeProtocolError

    with pytest.raises(ClaudeProtocolError):
        _parse_response({"rationale": "no prompt here"}, scene_index=3)


# ---------------------------------------------------------------------------
# Stage iteration
# ---------------------------------------------------------------------------


async def test_motion_prompts_skips_scenes_without_keyframe_path(
    tmp_path: Path,
) -> None:
    """A scene with no keyframe_path is left untouched (no Claude call)."""
    from platinum.models.db import create_all
    from platinum.pipeline.motion_prompts import motion_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)

    story = _story_with_keyframes(tmp_path, n=2)
    # Wipe the keyframe_path for scene 1; scene 2 keeps its keyframe.
    story.scenes[0].keyframe_path = None

    call_count = 0

    async def synth(_: dict) -> dict:
        nonlocal call_count
        call_count += 1
        return _synth_motion_prompt_response("a generic motion")

    await motion_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert call_count == 1, "expected Claude call only for the scene with a keyframe"
    assert story.scenes[0].motion_prompt is None
    assert story.scenes[1].motion_prompt == "a generic motion"


async def test_motion_prompts_skips_scenes_with_existing_motion_prompt(
    tmp_path: Path,
) -> None:
    """Re-running the stage is a no-op for scenes already populated.

    Cost-control + resume semantics: scene.motion_prompt set => skip.
    """
    from platinum.models.db import create_all
    from platinum.pipeline.motion_prompts import motion_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)

    story = _story_with_keyframes(tmp_path, n=2)
    story.scenes[0].motion_prompt = "already set"

    call_count = 0

    async def synth(_: dict) -> dict:
        nonlocal call_count
        call_count += 1
        return _synth_motion_prompt_response("fresh prompt")

    await motion_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert call_count == 1, "should call Claude once (for scene 2 only)"
    assert story.scenes[0].motion_prompt == "already set", "must not overwrite"
    assert story.scenes[1].motion_prompt == "fresh prompt"


async def test_motion_prompts_writes_motion_prompt_per_scene(
    tmp_path: Path,
) -> None:
    """Each scene gets its own Claude call; result lands on scene.motion_prompt."""
    from platinum.models.db import create_all
    from platinum.pipeline.motion_prompts import motion_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)

    story = _story_with_keyframes(tmp_path, n=3)

    seen: list[int] = []

    async def synth(request: dict) -> dict:
        # The user message should mention exactly one scene's narration.
        # We use this to spy on which scene this call is for.
        msg_text = ""
        for block in request["messages"][0]["content"]:
            if block.get("type") == "text":
                msg_text += block["text"]
        for s in story.scenes:
            if s.narration_text in msg_text:
                seen.append(s.index)
                return _synth_motion_prompt_response(f"motion for scene {s.index}")
        raise AssertionError(f"no scene matched in message: {msg_text!r}")

    await motion_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert seen == [1, 2, 3]
    assert story.scenes[0].motion_prompt == "motion for scene 1"
    assert story.scenes[1].motion_prompt == "motion for scene 2"
    assert story.scenes[2].motion_prompt == "motion for scene 3"


async def test_motion_prompts_returns_per_scene_results(tmp_path: Path) -> None:
    """motion_prompts() returns a list of (scene, ClaudeResult) for callers
    that want to record per-scene cost into stage artifacts."""
    from platinum.models.db import create_all
    from platinum.pipeline.motion_prompts import motion_prompts

    db_path = tmp_path / "p.db"
    create_all(db_path)

    story = _story_with_keyframes(tmp_path, n=2)

    async def synth(_: dict) -> dict:
        return _synth_motion_prompt_response("x")

    results = await motion_prompts(
        story=story, track_cfg=_track(),
        prompts_dir=PROMPTS_DIR, db_path=db_path, recorder=synth,
    )
    assert len(results) == 2
    for scene, claude_result in results:
        assert scene.motion_prompt == "x"
        assert claude_result.usage.cost_usd >= 0  # cost is non-negative


# ---------------------------------------------------------------------------
# S8.A.6 -- video_generator integration (prompt selection)
# ---------------------------------------------------------------------------


def test_select_video_prompt_prefers_motion_prompt() -> None:
    """When scene.motion_prompt is set, the I2V prompt is motion_prompt."""
    from platinum.pipeline.video_generator import _select_video_prompt

    scene = Scene(
        id="scene_001", index=1, narration_text="x",
        visual_prompt="V", motion_prompt="M",
    )
    assert _select_video_prompt(scene) == "M"


def test_select_video_prompt_falls_back_to_visual_prompt() -> None:
    """When motion_prompt is None, visual_prompt is used (pre-S8.A path)."""
    from platinum.pipeline.video_generator import _select_video_prompt

    scene = Scene(
        id="scene_001", index=1, narration_text="x",
        visual_prompt="V", motion_prompt=None,
    )
    assert _select_video_prompt(scene) == "V"


def test_select_video_prompt_returns_empty_when_both_unset() -> None:
    from platinum.pipeline.video_generator import _select_video_prompt

    scene = Scene(id="scene_001", index=1, narration_text="x")
    assert _select_video_prompt(scene) == ""


def test_select_video_prompt_handles_simplenamespace_without_motion_prompt() -> None:
    """SimpleNamespace test mocks lacking motion_prompt should fall back to
    visual_prompt, not raise AttributeError."""
    from types import SimpleNamespace

    from platinum.pipeline.video_generator import _select_video_prompt

    legacy = SimpleNamespace(visual_prompt="legacy V")
    assert _select_video_prompt(legacy) == "legacy V"


# ---------------------------------------------------------------------------
# S8.A.8 -- canonical stage order
# ---------------------------------------------------------------------------


def test_motion_prompts_inserted_between_review_and_video() -> None:
    """The orchestrator's canonical stage list must place motion_prompts
    after keyframe_review and before video_generator. If a future PR
    moves it, this test fails so the rationale (keyframe must exist
    before vision call; prompt must exist before video) gets a fresh
    review."""
    from platinum.pipeline.orchestrator import CANONICAL_STAGE_NAMES

    assert "motion_prompts" in CANONICAL_STAGE_NAMES
    review_idx = CANONICAL_STAGE_NAMES.index("keyframe_review")
    motion_idx = CANONICAL_STAGE_NAMES.index("motion_prompts")
    video_idx = CANONICAL_STAGE_NAMES.index("video_generator")
    assert review_idx < motion_idx < video_idx


