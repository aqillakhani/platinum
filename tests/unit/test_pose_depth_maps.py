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


# ---- B5.3: prerender + preprocessor pass -----------------------------------


def _build_test_ctx(tmp_path: Path):
    """Set up a tmp project + ctx wired to a FakeComfyClient (responses
    will be added by the caller)."""
    import shutil

    from platinum.config import Config
    from platinum.pipeline.context import PipelineContext
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient

    repo_root = Path(__file__).resolve().parents[2]

    (tmp_path / "config" / "tracks").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_path / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    shutil.copytree(
        repo_root / "config" / "workflows",
        tmp_path / "config" / "workflows",
        dirs_exist_ok=True,
    )

    config = Config(root=tmp_path)
    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses={}),
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
    }
    return PipelineContext(
        config=config,
        logger=__import__("logging").getLogger("test"),
    )


def _build_prerender_response(
    *, scene_index: int, repo_root: Path, fixture: Path
) -> dict[str, list[Path]]:
    """Build a {sig: [fixture]} response for the prerender call.

    Mirrors PoseDepthMapStage._prerender's workflow construction so the
    FakeComfyClient is keyed by the exact signature the Stage will compute.
    """
    from platinum.utils.comfyui import workflow_signature
    from platinum.utils.workflow import inject, load_workflow

    wf_template = load_workflow(
        "flux_dev_keyframe", config_dir=repo_root / "config"
    )
    # Default negative prompt -- mirror Stage's choice exactly.
    wf = inject(
        wf_template,
        prompt="Medium shot. Two men face each other across a vault arch.",
        negative_prompt="cartoon, anime, plastic, blurry, low quality",
        seed=scene_index * 1000 + 999,
        width=512,
        height=896,
        output_prefix=f"scene_{scene_index:03d}_prerender",
    )
    sampler_id = wf["_meta"]["role"]["sampler"]
    wf[sampler_id]["inputs"]["steps"] = 8
    return {workflow_signature(wf): [fixture]}


def _build_preprocessor_responses(
    *, prerender_path: Path, repo_root: Path,
    pose_fixture: Path, depth_fixture: Path,
) -> dict[str, list[Path]]:
    """Build responses for the pose_depth_map preprocessor calls.

    The Stage submits the same workflow twice (once for pose, once for
    depth); FakeComfyClient.responses[sig] rotates through the list so
    [pose_fixture, depth_fixture] yields each in order.
    """
    import copy

    from platinum.utils.comfyui import workflow_signature
    from platinum.utils.workflow import load_workflow

    wf_template = load_workflow(
        "pose_depth_map", config_dir=repo_root / "config"
    )
    wf = copy.deepcopy(wf_template)
    image_id = wf["_meta"]["role"]["image_input"]
    wf[image_id]["inputs"]["image"] = str(prerender_path)
    return {workflow_signature(wf): [pose_fixture, depth_fixture]}


async def test_run_generates_pose_and_depth_for_each_composition_scene(
    tmp_path: Path,
) -> None:
    """S7.1.B5.3: run() walks scenes with composition_notes, prerenders
    via Flux, runs DWPose + DepthAnythingV2 preprocessors, and writes
    scene.pose_ref_path + scene.depth_ref_path."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"
    ctx = _build_test_ctx(tmp_path)
    story = _story([
        _scene(
            1,
            composition_notes=(
                "Medium shot. Two men face each other across a vault arch."
            ),
        ),
        _scene(2),  # no composition_notes -- skipped
    ])
    ctx.config.story_dir(story.id).mkdir(parents=True, exist_ok=True)

    # Wire FakeComfyClient responses for the prerender + preprocessor pass.
    prerender_responses = _build_prerender_response(
        scene_index=1, repo_root=tmp_path,
        fixture=fixtures / "candidate_0.png",
    )
    out_dir = (
        ctx.config.story_dir(story.id) / "keyframes" / "scene_001"
    )
    expected_prerender = out_dir / "_prerender.png"
    preproc_responses = _build_preprocessor_responses(
        prerender_path=expected_prerender, repo_root=tmp_path,
        pose_fixture=fixtures / "candidate_1.png",
        depth_fixture=fixtures / "candidate_2.png",
    )
    ctx.config.settings["test"]["comfy_client"].responses = {
        **prerender_responses,
        **preproc_responses,
    }

    stage = PoseDepthMapStage()
    artifacts = await stage.run(story, ctx)

    # Only scene 1 had composition_notes -> only it appears in prepared.
    assert artifacts["prepared_scenes"] == [1]
    # Scene fields populated
    assert story.scenes[0].pose_ref_path is not None
    assert story.scenes[0].depth_ref_path is not None
    assert Path(story.scenes[0].pose_ref_path).exists()
    assert Path(story.scenes[0].depth_ref_path).exists()
    # Scene 2 (no composition_notes) untouched.
    assert story.scenes[1].pose_ref_path is None
    assert story.scenes[1].depth_ref_path is None


async def test_run_returns_empty_prepared_when_no_composition_scenes(
    tmp_path: Path,
) -> None:
    """S7.1.B5.3: stories with no composition_notes are a no-op."""
    from platinum.pipeline.pose_depth_maps import PoseDepthMapStage

    ctx = _build_test_ctx(tmp_path)
    story = _story([_scene(1), _scene(2)])  # no composition_notes
    ctx.config.story_dir(story.id).mkdir(parents=True, exist_ok=True)

    stage = PoseDepthMapStage()
    artifacts = await stage.run(story, ctx)
    assert artifacts["prepared_scenes"] == []
