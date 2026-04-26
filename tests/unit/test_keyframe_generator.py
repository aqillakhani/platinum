"""Tests for pipeline/keyframe_generator.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from platinum.models.story import Scene


def test_keyframe_report_is_frozen_and_carries_fields() -> None:
    from platinum.pipeline.keyframe_generator import KeyframeReport

    r = KeyframeReport(
        scene_index=0,
        candidates=[Path("a"), Path("b"), Path("c")],
        scores=[7.0, 5.0, 3.0],
        anatomy_passed=[True, True, False],
        scoring_succeeded=[True, True, True],
        selected_index=0,
        selected_via_fallback=False,
    )
    assert r.scene_index == 0
    assert r.selected_index == 0
    assert not r.selected_via_fallback
    assert r.scoring_succeeded == [True, True, True]
    with pytest.raises(FrozenInstanceError):
        r.scene_index = 1  # type: ignore[misc]


def test_keyframe_generation_error_carries_per_candidate_exceptions() -> None:
    from platinum.pipeline.keyframe_generator import KeyframeGenerationError

    excs: list[BaseException] = [RuntimeError("a"), ValueError("b"), TimeoutError("c")]
    err = KeyframeGenerationError(scene_index=4, exceptions=excs)
    assert err.scene_index == 4
    assert len(err.exceptions) == 3
    assert "scene 4" in str(err) or "scene_index=4" in str(err)


def _scene(idx: int = 0, *, visual_prompt: str = "a candle") -> Scene:
    return Scene(
        id=f"scene_{idx:03d}",
        index=idx,
        narration_text="Once upon a time...",
        narration_duration_seconds=5.0,
        visual_prompt=visual_prompt,
        negative_prompt="bright daylight",
    )


_TRACK_VISUAL = {
    "aesthetic": "cinematic dark, candlelight",
    "negative_prompt": "bright daylight, neon",
}
_GATES = {"aesthetic_min_score": 6.0}


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "keyframes"


def _get_workflow_template() -> dict:
    """Load the flux_dev_keyframe workflow template."""
    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    return load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")


def _build_fake_comfy_with_three_candidates() -> tuple[object, dict]:
    """Configure a FakeComfyClient that returns the 3 fixture PNGs in order
    keyed by the workflow signature each candidate's seed produces.

    Returns (FakeComfyClient, workflow_template).
    """
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject

    wf_template = _get_workflow_template()
    fixtures = _fixture_dir()
    seeds = (0, 1, 2)
    candidates = (
        fixtures / "candidate_0.png",
        fixtures / "candidate_1.png",
        fixtures / "candidate_2.png",
    )
    responses: dict[str, list[Path]] = {}
    for seed, fixture in zip(seeds, candidates, strict=True):
        wf = inject(
            wf_template,
            prompt="cinematic dark, candlelight a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024, height=1024,
            output_prefix=f"scene_000_candidate_{seeds.index(seed)}",
        )
        responses[workflow_signature(wf)] = [fixture]
    return FakeComfyClient(responses=responses), wf_template


async def test_generate_for_scene_happy_path_picks_highest_scoring(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 6.5,
        output_dir / "candidate_1.png": 8.0,
        output_dir / "candidate_2.png": 7.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.scene_index == 0
    assert len(report.candidates) == 3
    assert report.scores == [6.5, 8.0, 7.0]
    assert report.anatomy_passed == [True, True, True]
    assert report.selected_index == 1   # candidate_1 had highest score
    assert not report.selected_via_fallback


async def test_scoring_succeeded_populated_on_happy_path(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 6.5,
        output_dir / "candidate_1.png": 8.0,
        output_dir / "candidate_2.png": 7.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.scoring_succeeded == [True, True, True]


async def test_generate_for_scene_ties_selected_lowest_index(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 7.5,
        output_dir / "candidate_1.png": 7.5,
        output_dir / "candidate_2.png": 6.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.selected_index == 0   # lowest-index tie wins


async def test_generate_for_scene_below_threshold_falls_back(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 4.0,
        output_dir / "candidate_1.png": 5.0,
        output_dir / "candidate_2.png": 5.5,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.selected_index == 0
    assert report.selected_via_fallback is True


async def test_generate_for_scene_anatomy_rejects_high_score_candidate(tmp_path: Path) -> None:
    """High-score candidate fails anatomy; second-highest wins."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 6.0,
        output_dir / "candidate_1.png": 9.0,   # highest, but anatomy will reject
        output_dir / "candidate_2.png": 7.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()

    # A factory that returns "no hands" for cand 0 and 2, "anomaly" for cand 1.
    # The factory is invoked once per check_hand_anomalies call; we route by call index.
    call_idx = {"n": 0}

    def factory():
        from tests._fixtures import make_fake_hands_factory

        n = call_idx["n"]
        call_idx["n"] += 1
        if n == 1:
            return make_fake_hands_factory([21, 22])()  # anomaly
        return make_fake_hands_factory(None)()  # no hands -> passes

    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=factory,
    )
    assert report.anatomy_passed == [True, False, True]
    assert report.selected_index == 2   # cand 2 is the highest passing


async def test_generate_for_scene_missing_visual_prompt_raises(tmp_path: Path) -> None:
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0, visual_prompt=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as exc:
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates=_GATES,
            comfy=FakeComfyClient(responses={}),
            scorer=FakeAestheticScorer(fixed_score=8.0),
            output_dir=tmp_path / "scene_000",
            seeds=(0, 1, 2),
            mp_hands_factory=make_fake_hands_factory(None),
        )
    assert "visual_prompt" in str(exc.value)


async def test_generate_for_scene_isolates_per_candidate_exception(
    tmp_path: Path,
) -> None:
    """One candidate throws; other two succeed; selection picks among survivors."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    fixtures = _fixture_dir()
    output_dir = tmp_path / "scene_000"

    # Configure responses for seeds 0 and 2 only; seed 1 will raise KeyError.
    responses: dict[str, list[Path]] = {}
    for seed, fixture in [(0, fixtures / "candidate_0.png"), (2, fixtures / "candidate_2.png")]:
        wf = inject(
            wf_template,
            prompt="cinematic dark, candlelight a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024,
            height=1024,
            output_prefix=f"scene_000_candidate_{seed}",
        )
        responses[workflow_signature(wf)] = [fixture]
    comfy = FakeComfyClient(responses=responses)

    scene = _scene(idx=0)
    score_map = {
        output_dir / "candidate_0.png": 6.5,
        output_dir / "candidate_2.png": 8.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=_GATES,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.scores[1] == 0.0  # the failed candidate
    assert report.anatomy_passed[1] is False
    assert report.selected_index == 2  # candidate_2 has highest passing score


async def test_generate_for_scene_all_fail_raises_keyframe_generation_error(
    tmp_path: Path,
) -> None:
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        generate_for_scene,
    )
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    wf_template = _get_workflow_template()
    comfy = FakeComfyClient(responses={})  # no responses -> KeyError on every call
    with pytest.raises(KeyframeGenerationError) as exc:
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates=_GATES,
            comfy=comfy,
            scorer=FakeAestheticScorer(fixed_score=8.0),
            output_dir=tmp_path / "scene_000",
            workflow_template=wf_template,
            seeds=(0, 1, 2),
            mp_hands_factory=make_fake_hands_factory(None),
        )
    assert exc.value.scene_index == 0
    assert len(exc.value.exceptions) == 3


def test_generate_for_scene_seeds_default_to_index_offset() -> None:
    """When seeds=None, _seeds_for_scene(scene.index, n) is used."""
    from platinum.pipeline.keyframe_generator import _seeds_for_scene

    assert _seeds_for_scene(0, 3) == (0, 1, 2)
    assert _seeds_for_scene(7, 3) == (7000, 7001, 7002)
    assert _seeds_for_scene(12, 4) == (12000, 12001, 12002, 12003)


def _build_story_with_n_scenes(n: int):
    """Story factory for generate() tests."""
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story

    scenes = [
        Scene(
            id=f"scene_{i:03d}",
            index=i,
            narration_text=f"Scene {i} narration.",
            narration_duration_seconds=5.0,
            visual_prompt=f"a candle in dark hallway scene {i}",
            negative_prompt="bright daylight",
        )
        for i in range(n)
    ]
    return Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=Source(
            type="gutenberg",
            url="http://example.test",
            title="Test Story",
            author="Test",
            raw_text="x",
            fetched_at=datetime(2026, 4, 25),
            license="PD-US",
        ),
        scenes=scenes,
    )


async def test_generate_iterates_all_scenes(tmp_path: Path) -> None:
    """generate() returns one KeyframeReport per scene and mutates Scene
    fields."""
    from platinum.pipeline.keyframe_generator import generate
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    fixtures = _fixture_dir()

    n_scenes = 2
    story = _build_story_with_n_scenes(n_scenes)

    responses: dict[str, list[Path]] = {}
    for scene in story.scenes:
        for i, seed in enumerate((
            scene.index * 1000,
            scene.index * 1000 + 1,
            scene.index * 1000 + 2,
        )):
            wf = inject(
                wf_template,
                prompt=(
                    f"cinematic dark, candlelight a candle in dark hallway "
                    f"scene {scene.index}"
                ),
                negative_prompt="bright daylight",
                seed=seed,
                width=1024,
                height=1024,
                output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
            )
            responses[workflow_signature(wf)] = [fixtures / f"candidate_{i}.png"]
    comfy = FakeComfyClient(responses=responses)

    track_dict = {
        "visual": _TRACK_VISUAL,
        "quality_gates": _GATES,
    }

    class _Cfg:
        config_dir = repo_root / "config"

        def track(self, _id: str) -> dict:
            return track_dict

    reports = await generate(
        story,
        config=_Cfg(),
        comfy=comfy,
        scorer=FakeAestheticScorer(fixed_score=7.5),
        output_root=tmp_path,
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert len(reports) == n_scenes
    assert reports[0].scene_index == 0
    assert reports[1].scene_index == 1
    assert all(r.selected_via_fallback is False for r in reports)
    # Mutated Scene fields:
    for scene in story.scenes:
        assert scene.keyframe_path is not None
        assert len(scene.keyframe_candidates) == 3
        assert len(scene.keyframe_scores) == 3
        assert scene.validation.get("keyframe_anatomy") == [True, True, True]
        assert scene.validation.get("keyframe_selected_via_fallback") is False


async def test_all_scoring_fails_raises_keyframe_generation_error(
    tmp_path: Path,
) -> None:
    """All 3 candidates' scoring throws -> KeyframeGenerationError (infra failure halts).

    Distinguishes infrastructure failure (LAION threw for everyone -> halt) from
    content failure (LAION returned scores, none passed gates -> fallback).
    """
    import httpx

    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        generate_for_scene,
    )
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(0)
    fake_comfy, workflow_template = _build_fake_comfy_with_three_candidates()

    class _ThrowingScorer:
        async def score(self, image_path: Path) -> float:
            raise httpx.ConnectError("score_server unreachable")

    with pytest.raises(KeyframeGenerationError) as excinfo:
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates=_GATES,
            comfy=fake_comfy,
            scorer=_ThrowingScorer(),
            output_dir=tmp_path / "out",
            workflow_template=workflow_template,
            n_candidates=3,
            mp_hands_factory=make_fake_hands_factory(None),
        )
    assert excinfo.value.scene_index == 0
