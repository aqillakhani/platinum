"""Integration tests for KeyframeGeneratorStage."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from platinum.config import Config
from platinum.models.story import Scene, Source, Story
from platinum.pipeline.context import PipelineContext


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_config(tmp_project: Path, repo_root: Path) -> Config:
    """Copy track YAML and workflow configs to tmp_project; return fresh Config."""
    (tmp_project / "config" / "tracks").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_project / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    shutil.copytree(
        repo_root / "config" / "workflows",
        tmp_project / "config" / "workflows",
        dirs_exist_ok=True,
    )
    return Config(root=tmp_project)


def _build_story(n: int = 2) -> Story:
    scenes = [
        Scene(
            id=f"scene_{i:03d}",
            index=i,
            narration_text=f"Narration {i}.",
            narration_duration_seconds=5.0,
            visual_prompt=f"candle scene {i}",
            negative_prompt="bright daylight",
        )
        for i in range(n)
    ]
    return Story(
        id="story_int_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="http://example.test",
            title="Stage Test Story",
            author="Anon",
            raw_text="x",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=scenes,
    )


def _build_responses_for_story(story: Story, repo_root: Path) -> dict[str, list[Path]]:
    from platinum.config import Config
    from platinum.utils.comfyui import workflow_signature
    from platinum.utils.workflow import inject, load_workflow

    config = Config(root=repo_root)
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    track_cfg = config.track(story.track)
    track_visual = dict(track_cfg.get("visual", {}))
    fixtures = _fixture_dir()
    out: dict[str, list[Path]] = {}
    for scene in story.scenes:
        negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")
        for i, seed in enumerate(
            (scene.index * 1000, scene.index * 1000 + 1, scene.index * 1000 + 2)
        ):
            wf = inject(
                wf_template,
                prompt=scene.visual_prompt,
                negative_prompt=negative_text,
                seed=seed,
                width=768,
                height=1344,
                output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
            )
            out[workflow_signature(wf)] = [fixtures / f"candidate_{i}.png"]
    return out


async def test_keyframe_stage_runs_end_to_end(tmp_project, repo_root) -> None:  # noqa: ANN001
    """Stage.run mutates each scene, persists keyframe_path, returns artifacts dict."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=2)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    scorer = FakeAestheticScorer(fixed_score=8.0)

    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": scorer,
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert all(s.keyframe_path is not None for s in story.scenes)
    assert all(len(s.keyframe_candidates) == 3 for s in story.scenes)
    assert all(len(s.keyframe_scores) == 3 for s in story.scenes)
    assert artifacts["scenes_total"] == 2
    assert artifacts["scenes_succeeded"] == 2
    assert artifacts["scenes_via_fallback"] == 0


async def test_stage_respects_yaml_candidates_per_scene_override(tmp_project, repo_root) -> None:  # noqa: ANN001
    """yaml track.image_model.candidates_per_scene=5 -> 5 keyframe candidates per scene.

    Verifies end-to-end: Stage reads config override, forwards to generate(),
    which forwards to generate_for_scene(), producing 5 candidates instead of 3.
    """
    import shutil

    import yaml

    from platinum.config import Config
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    # Setup config directory and modify YAML BEFORE creating Config.
    (tmp_project / "config" / "tracks").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        repo_root / "config" / "tracks" / "atmospheric_horror.yaml",
        tmp_project / "config" / "tracks" / "atmospheric_horror.yaml",
    )
    shutil.copytree(
        repo_root / "config" / "workflows",
        tmp_project / "config" / "workflows",
        dirs_exist_ok=True,
    )

    # Modify track YAML to specify 5 candidates and disable quality gates.
    track_cfg_path = tmp_project / "config" / "tracks" / "atmospheric_horror.yaml"
    track_yaml_str = track_cfg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(track_yaml_str)
    track_cfg = data["track"]
    # Update image_model candidates and relax quality gates for synthetic images.
    if "image_model" not in track_cfg:
        track_cfg["image_model"] = {}
    track_cfg["image_model"]["candidates_per_scene"] = 5
    if "quality_gates" not in track_cfg:
        track_cfg["quality_gates"] = {}
    track_cfg["quality_gates"].update({
        "aesthetic_min_score": 0.0,
        "brightness_floor_mean_rgb": 0.0,
        "subject_min_edge_density": 0.0,
    })
    yaml_output = yaml.dump({"track": track_cfg}, default_flow_style=False)
    track_cfg_path.write_text(yaml_output, encoding="utf-8")

    # Now create Config with modified YAML.
    config = Config(root=tmp_project)
    story = _build_story(n=1)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)

    # Build FakeComfyClient responses for 5 candidates per scene.
    wf_template = load_workflow("flux_dev_keyframe", config_dir=tmp_project / "config")
    responses: dict[str, list[Path]] = {}
    scene = story.scenes[0]
    for i in range(5):
        seed = scene.index * 1000 + i
        wf = inject(
            wf_template,
            prompt=scene.visual_prompt,
            negative_prompt=scene.negative_prompt or "bright daylight",
            seed=seed,
            width=768,
            height=1344,
            output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
        )
        # Create distinct candidate PNG.
        candidate_path = tmp_project / "fixtures" / f"candidate_{i}_5cand.png"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_png(candidate_path, kind="grey", value=50 + i * 40)
        responses[workflow_signature(wf)] = [candidate_path]

    comfy = FakeComfyClient(responses=responses)
    scorer = FakeAestheticScorer(fixed_score=7.0)

    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": scorer,
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    # Assert: 5 candidates per scene when yaml specifies 5.
    assert all(len(s.keyframe_candidates) == 5 for s in story.scenes)
    assert all(len(s.keyframe_scores) == 5 for s in story.scenes)
    assert artifacts["scenes_total"] == 1
    assert artifacts["scenes_succeeded"] == 1


async def test_keyframe_stage_persists_story_json_after_each_scene(tmp_project, repo_root) -> None:  # noqa: ANN001
    """Per-scene checkpoint: story.json on disk reflects each scene as it lands."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=2)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story.save(story_dir / "story.json")

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    await stage.run(story, ctx)

    # Reload from disk; assert scenes round-trip with keyframe_path set.
    reloaded = Story.load(story_dir / "story.json")
    assert all(s.keyframe_path is not None for s in reloaded.scenes)


async def test_keyframe_stage_resumes_when_scene_already_has_keyframe(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """Pre-set scene 0's keyframe_path; Stage should skip it."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=2)
    pre_existing = (
        tmp_project / "data" / "stories" / story.id / "keyframes"
        / "scene_000" / "candidate_0.png"
    )
    pre_existing.parent.mkdir(parents=True, exist_ok=True)
    pre_existing.write_bytes((_fixture_dir() / "candidate_0.png").read_bytes())
    story.scenes[0].keyframe_path = pre_existing

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert story.scenes[0].keyframe_path == pre_existing  # unchanged
    assert story.scenes[1].keyframe_path is not None  # newly generated
    assert artifacts["scenes_total"] == 2
    assert artifacts["scenes_succeeded"] == 2  # both scenes have keyframes (one pre-existing)


async def test_keyframe_stage_records_failure_in_artifacts(tmp_project, repo_root) -> None:  # noqa: ANN001
    """All-fail scene -> Stage raises KeyframeGenerationError."""
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        KeyframeGeneratorStage,
    )
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=1)
    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses={}),  # all candidates KeyError
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))
    stage = KeyframeGeneratorStage()
    with pytest.raises(KeyframeGenerationError):
        await stage.run(story, ctx)


async def test_keyframe_stage_closes_stage_constructed_remote_scorer(  # noqa: ANN001
    tmp_project, repo_root, monkeypatch,
) -> None:
    """When Stage constructs RemoteAestheticScorer (no test override), it must
    call aclose() in a finally block so httpx connections don't leak across
    multi-scene runs on a live vast.ai box.
    """
    import httpx

    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import RemoteAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    config.settings.setdefault("aesthetics", {})["host"] = "http://test:8189"

    story = _build_story(n=1)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    responses = _build_responses_for_story(story, repo_root)

    async def _handler(request: httpx.Request) -> httpx.Response:
        # /score and /clip-sim share this handler; the scorer reads
        # different keys per endpoint, so return both with neutral values.
        return httpx.Response(200, json={"score": 7.5, "similarity": 0.5})

    aclose_calls: list[int] = []
    real_init = RemoteAestheticScorer.__init__
    real_aclose = RemoteAestheticScorer.aclose

    def _spy_init(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("transport", httpx.MockTransport(_handler))
        real_init(self, **kwargs)

    async def _spy_aclose(self):  # type: ignore[no-untyped-def]
        aclose_calls.append(1)
        await real_aclose(self)

    monkeypatch.setattr(RemoteAestheticScorer, "__init__", _spy_init)
    monkeypatch.setattr(RemoteAestheticScorer, "aclose", _spy_aclose)

    # Inject only comfy + mp_hands; let Stage construct the scorer itself.
    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses=responses),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    await stage.run(story, ctx)

    assert len(aclose_calls) == 1, (
        f"expected exactly one aclose() call from Stage cleanup, got {len(aclose_calls)}"
    )
    assert story.scenes[0].keyframe_path is not None


async def test_keyframe_stage_scene_filter_processes_only_selected_indices(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """Filter {0, 2} on 3-scene story -> scenes 0,2 get keyframe_path; scene 1 stays None."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=3)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    config.settings["runtime"] = {"scene_filter": {0, 2}}
    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    await stage.run(story, ctx)

    assert story.scenes[0].keyframe_path is not None
    assert story.scenes[1].keyframe_path is None  # filtered out
    assert story.scenes[2].keyframe_path is not None


async def test_keyframe_stage_no_scene_filter_processes_all_scenes(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """No scene_filter (or None) -> all 3 scenes processed (regression for default path)."""
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=3)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)

    responses = _build_responses_for_story(story, repo_root)
    comfy = FakeComfyClient(responses=responses)
    # Note: deliberately omit the runtime block; default behaviour should process all.
    config.settings["test"] = {
        "comfy_client": comfy,
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    await stage.run(story, ctx)

    for scene in story.scenes:
        assert scene.keyframe_path is not None


async def test_stage_brightness_gate_persists_correct_selection(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """End-to-end: Stage with FakeComfyClient producing one black + two bright
    PNGs. Selection persists to story.json. Brightness-failing candidate is
    NEVER the keyframe_path.
    """
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    config = _setup_config(tmp_project, repo_root)
    wf_template = load_workflow("flux_dev_keyframe", config_dir=config.config_dir)

    # 2-scene story; for each scene cand 0 = dark, cand 1 + 2 = bright.
    story = _build_story(n=2)
    fixtures_dir = tmp_project / "fixtures"
    fixtures_dir.mkdir()
    responses: dict[str, list[Path]] = {}
    track_cfg = config.track(story.track)
    track_visual = dict(track_cfg.get("visual", {}))

    for scene in story.scenes:
        negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")
        for cand_idx in range(3):
            p = fixtures_dir / f"s{scene.index}_c{cand_idx}.png"
            value = 0 if cand_idx == 0 else 200          # cand 0 dark, others bright
            make_synthetic_png(p, kind="grey", value=value)

            # Match seeds to candidates 0, 1, 2 for this scene
            seed = scene.index * 1000 + cand_idx
            wf = inject(
                wf_template,
                prompt=scene.visual_prompt,
                negative_prompt=negative_text,
                seed=seed,
                width=768,
                height=1344,
                output_prefix=f"scene_{scene.index:03d}_candidate_{cand_idx}",
            )
            responses[workflow_signature(wf)] = [p]

    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story_path = story_dir / "story.json"
    story.save(story_path)

    # Override quality_gates to disable subject gate for this brightness-only regression test.
    # This test uses solid-color PNGs which have edge_density=0 and would fail the subject gate.
    track_cfg = config.track(story.track)
    track_cfg.setdefault("quality_gates", {})["subject_min_edge_density"] = 0.0

    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses=responses),
        "aesthetic_scorer": FakeAestheticScorer(fixed_score=8.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert artifacts["scenes_total"] == 2
    assert artifacts["scenes_succeeded"] == 2
    for scene in story.scenes:
        # Selected candidate must NOT be candidate_0 (the dark one).
        assert scene.keyframe_path is not None
        assert "candidate_0.png" not in str(scene.keyframe_path)


async def test_stage_subject_gate_end_to_end_picks_subject_passing_candidate(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """End-to-end: Stage with FakeComfyClient producing three candidates:
    - Cand 0: solid color (passes brightness, fails subject gate)
    - Cand 1: checkerboard (passes both gates, high LAION score)
    - Cand 2: checkerboard (passes both gates, lower LAION score)

    Subject gate with min_edge_density=0.020 should filter out cand 0 and select
    cand 1 (highest LAION among subject-passing candidates).
    """
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    config = _setup_config(tmp_project, repo_root)
    wf_template = load_workflow("flux_dev_keyframe", config_dir=config.config_dir)

    # 1-scene story with 3 candidates
    story = _build_story(n=1)
    scene = story.scenes[0]
    fixtures_dir = tmp_project / "fixtures"
    fixtures_dir.mkdir()
    responses: dict[str, list[Path]] = {}
    track_cfg = config.track(story.track)
    track_visual = dict(track_cfg.get("visual", {}))

    negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")

    # Build fixtures and responses for 3 candidates
    fixture_paths = []
    for cand_idx in range(3):
        p = fixtures_dir / f"scene_0_candidate_{cand_idx}.png"
        if cand_idx == 0:
            # Solid grey (fails subject gate due to edge_density=0)
            make_synthetic_png(p, kind="grey", value=200, size=(256, 256))
        else:
            # Checkerboard (passes subject gate)
            make_synthetic_png(p, kind="checkerboard", size=(256, 256), block=16)
        fixture_paths.append(p)

        seed = scene.index * 1000 + cand_idx
        wf = inject(
            wf_template,
            prompt=scene.visual_prompt,
            negative_prompt=negative_text,
            seed=seed,
            width=768,
            height=1344,
            output_prefix=f"scene_{scene.index:03d}_candidate_{cand_idx}",
        )
        responses[workflow_signature(wf)] = [p]

    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story_path = story_dir / "story.json"
    story.save(story_path)

    # Build score_map using output paths (not fixture paths).
    # The stage places candidates at: story_dir / "keyframes" / "scene_NNN" / "candidate_{i}.png"
    output_dir = story_dir / "keyframes" / f"scene_{scene.index:03d}"
    score_map = {
        output_dir / "candidate_0.png": 9.0,  # Unused (fails subject gate first)
        output_dir / "candidate_1.png": 8.0,  # Highest among subject-passing
        output_dir / "candidate_2.png": 7.0,  # Lower score, but still eligible
    }

    # Enable subject gate (default 0.020)
    track_cfg = config.track(story.track)
    track_cfg.setdefault("quality_gates", {})["subject_min_edge_density"] = 0.020

    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses=responses),
        "aesthetic_scorer": MappedFakeScorer(scores_by_path=score_map, default=0.0),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert artifacts["scenes_total"] == 1
    assert artifacts["scenes_succeeded"] == 1
    assert artifacts["scenes_via_fallback"] == 0

    # Verify keyframe_path is candidate_1 (not candidate_0, which is subject-failing).
    # This proves the subject gate filtered out the solid-color candidate and selected
    # the highest-scoring checkerboard candidate.
    assert scene.keyframe_path is not None
    assert "candidate_1.png" in str(scene.keyframe_path)
    assert "candidate_0.png" not in str(scene.keyframe_path)

    # Verify no fallback (selection was among eligible candidates, not forced fallback)
    assert scene.validation.get("keyframe_selected_via_fallback") is False

    # Verify all 3 candidates were generated and persisted
    assert len(scene.keyframe_candidates) == 3
    assert len(scene.keyframe_scores) == 3


async def test_stage_clip_gate_filters_low_similarity_end_to_end(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """S7.1.A3.5: Stage-layer CLIP gate -- low-similarity candidate is dropped,
    the LAION-best CLIP-passing candidate wins.

    All 3 candidates render fine (real fixture PNGs), but candidate_0 returns
    clip_similarity=0.05 (below 0.20 threshold from atmospheric_horror.yaml).
    Candidates 1 and 2 return 0.30. The stage must select candidate 1
    (higher LAION score among CLIP-passing) over candidate 0 (highest LAION
    overall but blocked by CLIP).
    """
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=1)
    scene = story.scenes[0]
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story_path = story_dir / "story.json"
    story.save(story_path)
    responses = _build_responses_for_story(story, repo_root)

    output_dir = story_dir / "keyframes" / f"scene_{scene.index:03d}"
    score_map = {
        output_dir / "candidate_0.png": 9.0,  # would win on LAION but CLIP rejects
        output_dir / "candidate_1.png": 8.0,  # winner
        output_dir / "candidate_2.png": 7.0,
    }
    clip_map = {
        output_dir / "candidate_0.png": 0.05,  # below 0.20 threshold
        output_dir / "candidate_1.png": 0.30,
        output_dir / "candidate_2.png": 0.30,
    }

    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses=responses),
        "aesthetic_scorer": MappedFakeScorer(
            scores_by_path=score_map,
            default=0.0,
            clip_similarities_by_path=clip_map,
            clip_similarity_default=0.0,
        ),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    artifacts = await stage.run(story, ctx)

    assert artifacts["scenes_total"] == 1
    assert artifacts["scenes_succeeded"] == 1
    assert artifacts["scenes_via_fallback"] == 0
    assert scene.keyframe_path is not None
    assert "candidate_1.png" in str(scene.keyframe_path)
    assert "candidate_0.png" not in str(scene.keyframe_path)
    # candidate_0's score zeroed by CLIP failure (was 9.0 originally)
    assert scene.keyframe_scores[0] == 0.0


async def test_stage_clip_gate_halts_when_all_candidates_fail(  # noqa: ANN001
    tmp_project, repo_root,
) -> None:
    """S7.1.A3.5: when every candidate misses the CLIP threshold, the Stage
    surfaces KeyframeGenerationError -- the scene shouldn't quietly keep
    a degenerate keyframe just because LAION scored it.
    """
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        KeyframeGeneratorStage,
    )
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    config = _setup_config(tmp_project, repo_root)
    story = _build_story(n=1)
    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story_path = story_dir / "story.json"
    story.save(story_path)
    responses = _build_responses_for_story(story, repo_root)

    config.settings["test"] = {
        "comfy_client": FakeComfyClient(responses=responses),
        "aesthetic_scorer": MappedFakeScorer(
            scores_by_path={},
            default=8.0,
            clip_similarities_by_path={},
            clip_similarity_default=0.05,  # all candidates fail CLIP gate
        ),
        "mp_hands_factory": make_fake_hands_factory(None),
    }
    ctx = PipelineContext(config=config, logger=__import__("logging").getLogger("test"))

    stage = KeyframeGeneratorStage()
    with pytest.raises(KeyframeGenerationError, match="clip"):
        await stage.run(story, ctx)
