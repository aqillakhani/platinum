"""Tests for pipeline/pose_depth_maps.py -- PoseDepthMapStage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from platinum.models.story import Scene, Source, Story


def _scene(
    idx: int,
    *,
    composition_notes: str | None = None,
    pose_ref_path: str | None = None,
    depth_ref_path: str | None = None,
) -> Scene:
    return Scene(
        id=f"scene_{idx:03d}",
        index=idx,
        narration_text=f"narration {idx}",
        composition_notes=composition_notes,
        pose_ref_path=pose_ref_path,
        depth_ref_path=depth_ref_path,
    )


def _story(scenes: list[Scene]) -> Story:
    s = Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg", url="u", title="t", author="a", raw_text="rt",
            fetched_at=datetime(2026, 4, 29, tzinfo=UTC), license="PD-US",
        ),
    )
    s.scenes = scenes
    return s


def test_stage_name_is_pose_depth_maps() -> None:
    """S7.1.B5.2: stage registers under 'pose_depth_maps'."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    assert PoseDepthMapStage.name == "pose_depth_maps"


def test_is_complete_when_no_scene_has_composition_notes() -> None:
    """S7.1.B5.2: scenes without composition_notes don't need preprocessor
    outputs (transitional shots, dialogue-only). is_complete=True so
    the stage is a no-op for those stories."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    story = _story([_scene(1), _scene(2)])  # no composition_notes
    stage = PoseDepthMapStage()
    assert stage.is_complete(story) is True


def test_is_complete_false_when_composition_notes_present_but_no_paths(
    tmp_path: Path,
) -> None:
    """S7.1.B5.2: a scene with composition_notes but missing pose_ref_path
    or depth_ref_path means PoseDepthMapStage hasn't run yet.
    is_complete=False so the orchestrator will execute it."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    story = _story([
        _scene(
            1,
            composition_notes="Medium shot. Two men face off.",
            pose_ref_path=None,
            depth_ref_path=None,
        )
    ])
    stage = PoseDepthMapStage()
    assert stage.is_complete(story) is False


def test_is_complete_false_when_paths_set_but_files_missing(tmp_path: Path) -> None:
    """S7.1.B5.2: paths point at files that no longer exist on disk -> not complete."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    pose_ghost = tmp_path / "ghost_pose.png"  # never created
    depth_ghost = tmp_path / "ghost_depth.png"
    story = _story([
        _scene(
            1,
            composition_notes="Medium shot. Two men.",
            pose_ref_path=str(pose_ghost),
            depth_ref_path=str(depth_ghost),
        )
    ])
    stage = PoseDepthMapStage()
    assert stage.is_complete(story) is False


def test_is_complete_true_when_all_scenes_have_valid_refs(tmp_path: Path) -> None:
    """S7.1.B5.2: every composition_notes scene has BOTH pose and depth
    paths set to existing files -> is_complete=True (resume can skip)."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    pose1 = tmp_path / "scene1_pose.png"
    depth1 = tmp_path / "scene1_depth.png"
    pose2 = tmp_path / "scene2_pose.png"
    depth2 = tmp_path / "scene2_depth.png"
    for p in (pose1, depth1, pose2, depth2):
        p.write_bytes(b"\x89PNG_x")
    story = _story([
        _scene(
            1,
            composition_notes="Medium shot. Two men.",
            pose_ref_path=str(pose1),
            depth_ref_path=str(depth1),
        ),
        _scene(
            2,
            composition_notes="Close-up of a torch.",
            pose_ref_path=str(pose2),
            depth_ref_path=str(depth2),
        ),
        _scene(3),  # transitional, no composition_notes -- skipped by check
    ])
    stage = PoseDepthMapStage()
    assert stage.is_complete(story) is True


def test_is_complete_false_when_one_scene_has_only_pose_no_depth(tmp_path: Path) -> None:
    """S7.1.B5.2: BOTH pose AND depth must be present per scene; partial
    state means the preprocessor pass was interrupted -> rerun."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    pose = tmp_path / "scene1_pose.png"
    pose.write_bytes(b"\x89PNG_x")
    story = _story([
        _scene(
            1,
            composition_notes="Medium shot.",
            pose_ref_path=str(pose),
            depth_ref_path=None,  # missing
        )
    ])
    stage = PoseDepthMapStage()
    assert stage.is_complete(story) is False
