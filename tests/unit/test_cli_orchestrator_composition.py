"""Unit tests for CLI orchestrator stage composition (S8.B.4).

The adapt and keyframes commands compose their orchestrator stage lists
based on track configuration. When ``track_cfg.story_bible.enabled`` is
true, the bible stage is inserted at the right place; when false, the
list is unchanged. These tests pin that behavior independently of the
heavier integration tests (which require recorder fixtures and disk I/O).
"""
from __future__ import annotations


def _track_cfg(*, bible_enabled: bool) -> dict:
    return {
        "id": "atmospheric_horror",
        "story_bible": {
            "enabled": bible_enabled,
            "model": "claude-opus-4-7",
            "max_tokens": 16000,
        },
    }


def test_adapt_stages_omits_bible_when_disabled() -> None:
    from platinum.cli import _adapt_stages

    stages = _adapt_stages(_track_cfg(bible_enabled=False))
    names = [s.name for s in stages]
    assert names == ["story_adapter", "scene_breakdown", "visual_prompts"]


def test_adapt_stages_inserts_bible_between_breakdown_and_visual_prompts() -> None:
    from platinum.cli import _adapt_stages

    stages = _adapt_stages(_track_cfg(bible_enabled=True))
    names = [s.name for s in stages]
    assert names == [
        "story_adapter",
        "scene_breakdown",
        "story_bible",
        "visual_prompts",
    ]


def test_adapt_stages_handles_missing_story_bible_block() -> None:
    """Tracks that haven't yet declared a story_bible block (older YAMLs,
    or stub tracks pre-S8.B) must not crash — the helper treats an absent
    block as disabled."""
    from platinum.cli import _adapt_stages

    track_no_bible = {"id": "legacy_track"}  # no story_bible key at all
    stages = _adapt_stages(track_no_bible)
    names = [s.name for s in stages]
    assert "story_bible" not in names
    assert names == ["story_adapter", "scene_breakdown", "visual_prompts"]


def test_keyframes_phase2_stages_omits_bible_when_disabled() -> None:
    from platinum.cli import _keyframes_phase2_stages

    stages = _keyframes_phase2_stages(_track_cfg(bible_enabled=False))
    names = [s.name for s in stages]
    assert names == ["pose_depth_maps", "keyframe_generator"]


def test_keyframes_phase2_stages_prepends_bible_when_enabled() -> None:
    """Per S8.B plan: prepend StoryBibleStage to keyframes phase-2 stages
    when the track enables it. Acts as a safety net — if the user runs
    keyframes directly without an adapt-time bible, the stage runs here
    and produces the bible on the fly. If the bible is already populated,
    StoryBibleStage.is_complete returns True and the orchestrator skips it."""
    from platinum.cli import _keyframes_phase2_stages

    stages = _keyframes_phase2_stages(_track_cfg(bible_enabled=True))
    names = [s.name for s in stages]
    assert names == ["story_bible", "pose_depth_maps", "keyframe_generator"]


def test_keyframes_phase2_stages_threads_scene_filter_to_keyframe_generator() -> None:
    """S8.B.7: --scenes / --rerun-rejected feed a scene_filter into the
    helper, which forwards it to KeyframeGeneratorStage so its per-scene
    is_complete inspects only the active subset. Pin this so the
    plumbing isn't accidentally severed by future refactors."""
    from platinum.cli import _keyframes_phase2_stages

    stages = _keyframes_phase2_stages(
        _track_cfg(bible_enabled=False), scene_filter={1, 3, 5},
    )
    keyframe_stage = next(s for s in stages if s.name == "keyframe_generator")
    assert keyframe_stage.scene_filter == {1, 3, 5}


def test_keyframes_phase2_stages_default_scene_filter_is_none() -> None:
    """When the CLI doesn't pass --scenes, the keyframe_generator stage's
    scene_filter is None and the per-scene is_complete inspects every scene."""
    from platinum.cli import _keyframes_phase2_stages

    stages = _keyframes_phase2_stages(_track_cfg(bible_enabled=False))
    keyframe_stage = next(s for s in stages if s.name == "keyframe_generator")
    assert keyframe_stage.scene_filter is None
