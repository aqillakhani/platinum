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
        # Match exact aesthetic_text construction from generate_for_scene
        aesthetic_text = " ".join(
            s for s in (track_visual.get("aesthetic"), scene.visual_prompt) if s
        )
        negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")
        for i, seed in enumerate(
            (scene.index * 1000, scene.index * 1000 + 1, scene.index * 1000 + 2)
        ):
            wf = inject(
                wf_template,
                prompt=aesthetic_text,
                negative_prompt=negative_text,
                seed=seed,
                width=1024,
                height=1024,
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
        return httpx.Response(200, json={"score": 7.5})

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
        aesthetic_text = " ".join(
            s for s in (track_visual.get("aesthetic"), scene.visual_prompt) if s
        )
        negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")
        for cand_idx in range(3):
            p = fixtures_dir / f"s{scene.index}_c{cand_idx}.png"
            value = 0 if cand_idx == 0 else 200          # cand 0 dark, others bright
            make_synthetic_png(p, kind="grey", value=value)

            # Match seeds to candidates 0, 1, 2 for this scene
            seed = scene.index * 1000 + cand_idx
            wf = inject(
                wf_template,
                prompt=aesthetic_text,
                negative_prompt=negative_text,
                seed=seed,
                width=1024,
                height=1024,
                output_prefix=f"scene_{scene.index:03d}_candidate_{cand_idx}",
            )
            responses[workflow_signature(wf)] = [p]

    story_dir = tmp_project / "data" / "stories" / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story_path = story_dir / "story.json"
    story.save(story_path)

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
