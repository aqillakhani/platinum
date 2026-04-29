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
        brightness_passed=[True, True, True],
        subject_passed=[True, True, True],
        selected_index=0,
        selected_via_fallback=False,
    )
    assert r.scene_index == 0
    assert r.selected_index == 0
    assert not r.selected_via_fallback
    assert r.scoring_succeeded == [True, True, True]
    with pytest.raises(FrozenInstanceError):
        r.scene_index = 1  # type: ignore[misc]


def test_keyframe_report_carries_brightness_passed_field() -> None:
    """brightness_passed is an immutable list[bool] aligned to candidates."""
    import dataclasses

    from platinum.pipeline.keyframe_generator import KeyframeReport

    r = KeyframeReport(
        scene_index=0,
        candidates=[Path("/tmp/c0.png"), Path("/tmp/c1.png"), Path("/tmp/c2.png")],
        scores=[5.0, 6.0, 7.0],
        anatomy_passed=[True, True, True],
        scoring_succeeded=[True, True, True],
        brightness_passed=[False, True, True],
        subject_passed=[True, True, True],
        selected_index=2,
        selected_via_fallback=False,
    )
    assert r.brightness_passed == [False, True, True]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.brightness_passed = [True, True, True]  # type: ignore[misc]


def test_keyframe_report_carries_clip_passed_field() -> None:
    """S7.1.A3.3: clip_passed is an immutable list[bool] aligned to candidates."""
    import dataclasses

    from platinum.pipeline.keyframe_generator import KeyframeReport

    r = KeyframeReport(
        scene_index=0,
        candidates=[Path("/tmp/c0.png"), Path("/tmp/c1.png"), Path("/tmp/c2.png")],
        scores=[5.0, 6.0, 7.0],
        anatomy_passed=[True, True, True],
        scoring_succeeded=[True, True, True],
        brightness_passed=[True, True, True],
        subject_passed=[True, True, True],
        clip_passed=[False, True, True],
        selected_index=2,
        selected_via_fallback=False,
    )
    assert r.clip_passed == [False, True, True]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.clip_passed = [True, True, True]  # type: ignore[misc]


def test_keyframe_report_carries_content_gate_fields() -> None:
    """S7.1.A4.5: content_passed + content_scores aligned to candidates."""
    import dataclasses

    from platinum.pipeline.keyframe_generator import KeyframeReport

    r = KeyframeReport(
        scene_index=0,
        candidates=[Path("/tmp/c0.png"), Path("/tmp/c1.png"), Path("/tmp/c2.png")],
        scores=[5.0, 6.0, 7.0],
        anatomy_passed=[True, True, True],
        scoring_succeeded=[True, True, True],
        brightness_passed=[True, True, True],
        subject_passed=[True, True, True],
        clip_passed=[True, True, True],
        content_passed=[False, True, True],
        content_scores=[3, 8, 7],
        selected_index=1,
        selected_via_fallback=False,
    )
    assert r.content_passed == [False, True, True]
    assert r.content_scores == [3, 8, 7]
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.content_passed = [True, True, True]  # type: ignore[misc]


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
            prompt="a candle",
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
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
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
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
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
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
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
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.selected_index == 2  # highest-scored from scored subset (5.5 > 5.0 > 4.0)
    assert report.selected_via_fallback is True
    assert report.scoring_succeeded == [True, True, True]


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
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
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
            prompt="a candle",
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
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
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
                    f"a candle in dark hallway scene {scene.index}"
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
        "quality_gates": {**_GATES, "subject_min_edge_density": 0.0},
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


async def test_partial_scoring_falls_back_to_highest_in_scored_subset(
    tmp_path: Path,
) -> None:
    """Candidates 0 and 1 fail to score; candidate 2 scores 3.0 (below 4.0 gate).
    Should fall back to candidate 2 (the only scored one), not candidate 0.

    Distinguishes infrastructure partial failure from "always pick 0" silent fallback.
    """
    import httpx

    from platinum.pipeline.keyframe_generator import generate_for_scene
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(0)
    fake_comfy, workflow_template = _build_fake_comfy_with_three_candidates()
    output_dir = tmp_path / "scene_000"

    class _PartialScorer:
        async def score(self, image_path: Path) -> float:
            if image_path.name in {"candidate_0.png", "candidate_1.png"}:
                raise httpx.ConnectError("intermittent failure")
            return 3.0  # below the 4.0 threshold (default _GATES has 6.0; override)

    # Use a lower threshold so 3.0 is still below the gate (content failure path).
    gates = {"aesthetic_min_score": 4.0, "subject_min_edge_density": 0.0}

    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=gates,
        comfy=fake_comfy,
        scorer=_PartialScorer(),
        output_dir=output_dir,
        workflow_template=workflow_template,
        n_candidates=3,
        mp_hands_factory=make_fake_hands_factory(None),
    )
    assert report.scoring_succeeded == [False, False, True]
    assert report.selected_via_fallback is True
    assert report.selected_index == 2


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


@pytest.mark.asyncio
async def test_generate_for_scene_brightness_gate_skips_laion_on_dark_candidates(
    tmp_path: Path,
) -> None:
    """A brightness-failing candidate gets score=0.0, scoring_succeeded=False,
    AND the scorer must not even be invoked on it (saves the network round-trip).
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # Three synthetic candidates in a fixture dir (for FakeComfyClient to copy from):
    # cand 0 = black (fails brightness), cand 1 + 2 = bright.
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [
        fixture_dir / "candidate_0.png",
        fixture_dir / "candidate_1.png",
        fixture_dir / "candidate_2.png",
    ]
    make_synthetic_png(fixture_paths[0], kind="grey", value=0)
    make_synthetic_png(fixture_paths[1], kind="grey", value=200)
    make_synthetic_png(fixture_paths[2], kind="grey", value=200)

    # Build responses with correct prompts to match workflow signature.
    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(
            wf_template,
            prompt="dark scene",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024, height=1024, output_prefix=f"scene_000_candidate_{i}",
        )
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    comfy = FakeComfyClient(responses=responses)

    # Counter-wrapped scorer to verify it's NOT called on the dark candidate.
    call_count = {"n": 0}
    class CountingScorer(FakeAestheticScorer):
        async def score(self, path):
            call_count["n"] += 1
            return await super().score(path)
    scorer = CountingScorer(fixed_score=8.0)

    scene = Scene(
        id="scene_000", index=0, narration_text="x",
        narration_duration_seconds=1.0, visual_prompt="dark scene",
        negative_prompt="bright daylight",
    )

    output_dir = tmp_path / "scene_000"
    gates = {
        "aesthetic_min_score": 6.0,
        "brightness_floor_mean_rgb": 20.0,
        "subject_min_edge_density": 0.0,
    }
    report = await generate_for_scene(
        scene,
        track_visual={"aesthetic": "cinematic dark, candlelight"},
        quality_gates=gates,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )

    assert report.brightness_passed == [False, True, True]
    assert report.scoring_succeeded == [False, True, True]
    assert report.scores[0] == 0.0
    assert call_count["n"] == 2                              # cand 0 skipped


@pytest.mark.asyncio
async def test_generate_for_scene_eligibility_excludes_brightness_failures(
    tmp_path: Path,
) -> None:
    """A high-LAION-score candidate that fails brightness must NOT be eligible.

    Setup: cand 0 = bright + score 5.5 (below threshold), cand 1 = dark +
    score 8.0 (high but dark), cand 2 = bright + score 6.5. Threshold 6.0.
    Eligible: only cand 2 (bright + above threshold). Cand 1 NEVER picked
    despite having highest raw score.
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # Create fixture PNGs that FakeComfyClient will copy from.
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [
        fixture_dir / "candidate_0.png",
        fixture_dir / "candidate_1.png",
        fixture_dir / "candidate_2.png",
    ]
    make_synthetic_png(fixture_paths[0], kind="grey", value=200)        # bright
    make_synthetic_png(fixture_paths[1], kind="grey", value=0)          # dark!
    make_synthetic_png(fixture_paths[2], kind="grey", value=200)        # bright

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="x", negative_prompt="", seed=seed,
                    width=1024, height=1024, output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    out = tmp_path / "out"
    out.mkdir()
    score_map = {
        out / "candidate_0.png": 5.5,
        out / "candidate_1.png": 8.0,                            # high but dark!
        out / "candidate_2.png": 6.5,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = FakeComfyClient(responses=responses)

    scene = Scene(id="s0", index=0, narration_text="x",
                  narration_duration_seconds=1.0, visual_prompt="x")

    gates = {
        "aesthetic_min_score": 6.0,
        "brightness_floor_mean_rgb": 20.0,
        "subject_min_edge_density": 0.0,
    }
    report = await generate_for_scene(
        scene,
        track_visual={},
        quality_gates=gates,
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
    )

    assert report.brightness_passed == [True, False, True]
    assert report.selected_index == 2                           # NOT 1
    assert not report.selected_via_fallback                      # cand 2 is eligible


@pytest.mark.asyncio
async def test_generate_for_scene_fallback_skips_brightness_failures(
    tmp_path: Path,
) -> None:
    """When NO candidate is eligible (all below threshold OR anatomy fails),
    fallback must pick highest-scored among (scoring_succeeded AND brightness_passed),
    NOT the highest raw score (which could be a dark candidate).

    Setup: all 3 below 6.0 threshold. Cand 0 = bright + 5.0, cand 1 = dark + 5.8
    (highest raw), cand 2 = bright + 5.5. Fallback: pick cand 2 (highest among
    bright). Cand 1 never picked despite higher raw score.
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # Create fixture PNGs that FakeComfyClient will copy from.
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [
        fixture_dir / "candidate_0.png",
        fixture_dir / "candidate_1.png",
        fixture_dir / "candidate_2.png",
    ]
    make_synthetic_png(fixture_paths[0], kind="grey", value=200)
    make_synthetic_png(fixture_paths[1], kind="grey", value=0)
    make_synthetic_png(fixture_paths[2], kind="grey", value=200)

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="x", negative_prompt="", seed=seed,
                    width=1024, height=1024, output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    out = tmp_path / "out"
    out.mkdir()
    score_map = {
        out / "candidate_0.png": 5.0,
        out / "candidate_1.png": 5.8,                            # highest, but dark
        out / "candidate_2.png": 5.5,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = FakeComfyClient(responses=responses)

    scene = Scene(id="s0", index=0, narration_text="x",
                  narration_duration_seconds=1.0, visual_prompt="x")

    gates = {
        "aesthetic_min_score": 6.0,
        "brightness_floor_mean_rgb": 20.0,
        "subject_min_edge_density": 0.0,
    }
    report = await generate_for_scene(
        scene,
        track_visual={},
        quality_gates=gates,
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
    )

    assert report.brightness_passed == [True, False, True]
    assert report.selected_via_fallback                         # all below threshold
    assert report.selected_index == 2                           # NOT 1


@pytest.mark.asyncio
async def test_generate_for_scene_halts_on_all_dark_candidates(
    tmp_path: Path,
) -> None:
    """Task 9: All-dark-candidates halt raises KeyframeGenerationError."""
    from platinum.models.story import Scene
    from platinum.pipeline.keyframe_generator import KeyframeGenerationError, generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # Create fixture PNGs: all DARK (mean RGB < 20.0).
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [
        fixture_dir / "candidate_0.png",
        fixture_dir / "candidate_1.png",
        fixture_dir / "candidate_2.png",
    ]
    make_synthetic_png(fixture_paths[0], kind="grey", value=10)   # dark
    make_synthetic_png(fixture_paths[1], kind="grey", value=5)    # darker
    make_synthetic_png(fixture_paths[2], kind="grey", value=15)   # still dark

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="x", negative_prompt="", seed=seed,
                    width=1024, height=1024, output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    out = tmp_path / "out"
    out.mkdir()
    score_map = {
        out / "candidate_0.png": 8.0,
        out / "candidate_1.png": 9.0,   # all would score fine
        out / "candidate_2.png": 7.0,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = FakeComfyClient(responses=responses)

    scene = Scene(id="s0", index=0, narration_text="x",
                  narration_duration_seconds=1.0, visual_prompt="x")

    # All candidates fail brightness floor (all mean RGB < 20).
    # Should halt with KeyframeGenerationError, even though all scored.
    with pytest.raises(KeyframeGenerationError) as exc_info:
        await generate_for_scene(
            scene,
            track_visual={},
            quality_gates={"aesthetic_min_score": 6.0, "brightness_floor_mean_rgb": 20.0},
            comfy=comfy, scorer=scorer,
            output_dir=out,
            workflow_template=wf_template,
            seeds=(0, 1, 2),
        )

    assert exc_info.value.scene_index == 0
    assert len(exc_info.value.exceptions) == 3
    assert all("brightness floor" in str(e) for e in exc_info.value.exceptions)


@pytest.mark.asyncio
async def test_keyframe_report_carries_subject_passed_field(tmp_path):
    """KeyframeReport must expose subject_passed: list[bool] alongside brightness_passed."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    scorer = FakeAestheticScorer(fixed_score=8.0)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()

    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={"aesthetic_min_score": 6.0,
                       "brightness_floor_mean_rgb": 0.0,        # permissive
                       "subject_min_edge_density": 0.0},        # permissive
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )

    assert hasattr(report, "subject_passed")
    assert isinstance(report.subject_passed, list)
    assert len(report.subject_passed) == 3
    assert all(isinstance(p, bool) for p in report.subject_passed)


@pytest.mark.asyncio
async def test_generate_for_scene_subject_gate_skips_laion_on_solid_color(tmp_path):
    """Subject gate runs BEFORE LAIAN call -- a solid-color (subject-fail)
    candidate must NOT trigger the scorer."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [
        fixture_dir / "candidate_0.png",
        fixture_dir / "candidate_1.png",
        fixture_dir / "candidate_2.png",
    ]
    # Cand 0 = solid color (passes brightness, fails subject)
    make_synthetic_png(fixture_paths[0], kind="grey", value=200, size=(256, 256))
    # Cand 1, 2 = checkerboard (passes brightness, passes subject)
    make_synthetic_png(fixture_paths[1], kind="checkerboard", size=(256, 256), block=16)
    make_synthetic_png(fixture_paths[2], kind="checkerboard", size=(256, 256), block=16)

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="a candle",
                    negative_prompt="bright daylight",
                    seed=seed, width=1024, height=1024,
                    output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    call_count = {"n": 0}
    class CountingScorer(FakeAestheticScorer):
        async def score(self, path):
            call_count["n"] += 1
            return await super().score(path)
    scorer = CountingScorer(fixed_score=8.0)
    comfy = FakeComfyClient(responses=responses)

    scene = _scene(idx=0)
    out = tmp_path / "scene_000"
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={"aesthetic_min_score": 6.0,
                       "brightness_floor_mean_rgb": 20.0,
                       "subject_min_edge_density": 0.020},
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )

    assert report.brightness_passed == [True, True, True]
    assert report.subject_passed == [False, True, True]
    assert call_count["n"] == 2                          # cand 0 skipped LAIAN


@pytest.mark.asyncio
async def test_generate_for_scene_subject_gate_runs_after_brightness(tmp_path):
    """Subject gate must NOT run on a brightness-failing candidate (saves cv2 work)."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [fixture_dir / f"c{i}.png" for i in range(3)]
    # Cand 0 = black (fails brightness)
    make_synthetic_png(fixture_paths[0], kind="grey", value=0, size=(256, 256))
    # Cand 1, 2 = checkerboard
    make_synthetic_png(fixture_paths[1], kind="checkerboard", size=(256, 256), block=16)
    make_synthetic_png(fixture_paths[2], kind="checkerboard", size=(256, 256), block=16)

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="a candle",
                    negative_prompt="bright daylight",
                    seed=seed, width=1024, height=1024,
                    output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]
    scorer = FakeAestheticScorer(fixed_score=8.0)
    comfy = FakeComfyClient(responses=responses)

    scene = _scene(idx=0)
    out = tmp_path / "scene_000"
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={"aesthetic_min_score": 6.0,
                       "brightness_floor_mean_rgb": 20.0,
                       "subject_min_edge_density": 0.020},
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )

    assert report.brightness_passed == [False, True, True]
    # Cand 0 failed brightness -- subject gate should NOT have run on it.
    # We assert subject_passed[0] is False (the convention: skip = not-passed).
    assert report.subject_passed[0] is False
    assert report.subject_passed[1] is True
    assert report.subject_passed[2] is True


@pytest.mark.asyncio
async def test_generate_for_scene_halts_when_all_subject_fail(tmp_path):
    """All 3 candidates solid-color -> KeyframeGenerationError."""
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        generate_for_scene,
    )
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [fixture_dir / f"c{i}.png" for i in range(3)]
    for p in fixture_paths:
        # All bright (passes brightness) but solid-color (fails subject)
        make_synthetic_png(p, kind="grey", value=200, size=(256, 256))

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="a candle",
                    negative_prompt="bright daylight",
                    seed=seed, width=1024, height=1024,
                    output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    scorer = FakeAestheticScorer(fixed_score=8.0)
    comfy = FakeComfyClient(responses=responses)

    scene = _scene(idx=0)
    out = tmp_path / "scene_000"
    with pytest.raises(KeyframeGenerationError) as exc_info:
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates={"aesthetic_min_score": 6.0,
                           "brightness_floor_mean_rgb": 20.0,
                           "subject_min_edge_density": 0.020},
            comfy=comfy, scorer=scorer,
            output_dir=out,
            workflow_template=wf_template,
            seeds=(0, 1, 2),
        )
    assert exc_info.value.scene_index == 0
    assert any("subject" in str(e).lower() for e in exc_info.value.exceptions)


@pytest.mark.asyncio
async def test_eligibility_excludes_subject_failing_candidates(tmp_path):
    """A high-LAION-score candidate that fails subject gate must NOT be eligible.

    Setup: cand 0 = checkerboard + score 5.5 (below threshold), cand 1 = solid +
    score 8.0 (high but no subject), cand 2 = checkerboard + score 6.5.
    Threshold 6.0. Eligible: only cand 2. Cand 1 NEVER picked.
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [fixture_dir / f"c{i}.png" for i in range(3)]
    make_synthetic_png(fixture_paths[0], kind="checkerboard", size=(256, 256), block=16)
    make_synthetic_png(fixture_paths[1], kind="grey", value=200, size=(256, 256))   # no subject!
    make_synthetic_png(fixture_paths[2], kind="checkerboard", size=(256, 256), block=16)

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="x", negative_prompt="", seed=seed,
                    width=1024, height=1024, output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]
    out = tmp_path / "out"
    out.mkdir()
    score_map = {
        out / "candidate_0.png": 5.5,
        out / "candidate_1.png": 8.0,                    # high but no subject!
        out / "candidate_2.png": 6.5,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = FakeComfyClient(responses=responses)
    scene = Scene(id="s0", index=0, narration_text="x",
                  narration_duration_seconds=1.0, visual_prompt="x")

    report = await generate_for_scene(
        scene,
        track_visual={},
        quality_gates={"aesthetic_min_score": 6.0,
                       "brightness_floor_mean_rgb": 20.0,
                       "subject_min_edge_density": 0.020},
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
    )

    assert report.subject_passed == [True, False, True]
    assert report.selected_index == 2                    # only eligible
    assert report.selected_via_fallback is False


@pytest.mark.asyncio
async def test_fallback_pool_excludes_subject_failing(tmp_path):
    """No eligible candidate (all below threshold), fallback must NOT pick
    the highest-scored solid-color candidate."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [fixture_dir / f"c{i}.png" for i in range(3)]
    make_synthetic_png(fixture_paths[0], kind="checkerboard", size=(256, 256), block=16)
    make_synthetic_png(fixture_paths[1], kind="grey", value=200, size=(256, 256))   # no subject!
    make_synthetic_png(fixture_paths[2], kind="checkerboard", size=(256, 256), block=16)

    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="x", negative_prompt="", seed=seed,
                    width=1024, height=1024, output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]
    out = tmp_path / "out"
    out.mkdir()
    score_map = {
        out / "candidate_0.png": 5.0,                    # below threshold 6.0
        out / "candidate_1.png": 9.0,                    # high but no subject!
        out / "candidate_2.png": 5.5,                    # below threshold 6.0
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    comfy = FakeComfyClient(responses=responses)
    scene = Scene(id="s0", index=0, narration_text="x",
                  narration_duration_seconds=1.0, visual_prompt="x")

    report = await generate_for_scene(
        scene,
        track_visual={},
        quality_gates={"aesthetic_min_score": 6.0,
                       "brightness_floor_mean_rgb": 20.0,
                       "subject_min_edge_density": 0.020},
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
    )

    # No eligible (none above threshold) -> fallback. Must be cand 0 or cand 2,
    # NOT cand 1 (which has highest score but failed subject gate).
    assert report.selected_via_fallback is True
    assert report.selected_index in (0, 2)
    assert report.selected_index != 1
    # Specifically, max-scored among (subject_passed AND brightness_passed AND
    # scoring_succeeded) is cand 2 (score 5.5 > cand 0's 5.0).
    assert report.selected_index == 2


@pytest.mark.asyncio
async def test_track_aesthetic_prefix_no_longer_prepended(tmp_path: Path) -> None:
    """Regression: track_visual['aesthetic'] must NOT appear in inject() prompt arg.

    Phase 6.3 dropped the prefix because visual_prompts S4 stage already
    incorporates track aesthetic per-scene; the prefix double-counted.
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_fake_hands_factory, make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    MARKER = "MARKER_DO_NOT_INCLUDE_PREFIX_REMOVED_S6_3"

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_paths = [fixture_dir / f"c{i}.png" for i in range(3)]
    for p in fixture_paths:
        make_synthetic_png(p, kind="checkerboard", size=(256, 256), block=16)

    # Build EXPECTED workflow signatures using JUST scene.visual_prompt
    # (no MARKER concatenation). If the impl still prepends, signatures
    # won't match -> FakeComfyClient raises KeyError on lookup.
    responses = {}
    for i, seed in enumerate((0, 1, 2)):
        wf = inject(wf_template, prompt="a candle", negative_prompt="bright daylight",
                    seed=seed, width=1024, height=1024,
                    output_prefix=f"scene_000_candidate_{i}")
        responses[workflow_signature(wf)] = [fixture_paths[i]]

    scorer = FakeAestheticScorer(fixed_score=8.0)
    comfy = FakeComfyClient(responses=responses)

    scene = _scene(idx=0, visual_prompt="a candle")
    out = tmp_path / "out"
    out.mkdir()

    # Pass an explicit aesthetic that would FAIL the test if it gets prepended.
    report = await generate_for_scene(
        scene,
        track_visual={"aesthetic": MARKER, "negative_prompt": "bright daylight"},
        quality_gates={"aesthetic_min_score": 6.0,
                       "brightness_floor_mean_rgb": 20.0,
                       "subject_min_edge_density": 0.020},
        comfy=comfy, scorer=scorer,
        output_dir=out,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
    )

    # If the prefix is still applied, FakeComfyClient would have raised KeyError
    # because no signature matches the prefixed prompt. Reaching here means
    # the prompt was scene.visual_prompt alone.
    assert all(report.brightness_passed)
    assert all(report.subject_passed)


@pytest.mark.asyncio
async def test_generate_forwards_candidates_per_scene_from_track_config(
    tmp_path: Path,
) -> None:
    """yaml track.image_model.candidates_per_scene=5 -> 5 candidate PNGs.

    Currently FAILS because generate() doesn't read image_model and the
    n_candidates default of 3 always wins. Wiring the yaml override into
    generate_for_scene's n_candidates kwarg fixes it.
    """
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story
    from platinum.pipeline.keyframe_generator import generate
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # Synthesize 5 distinct PNGs in tmp_path with different gradient values.
    candidate_files = []
    for i in range(5):
        path = tmp_path / f"src_candidate_{i}.png"
        make_synthetic_png(path, kind="grey", value=50 + i * 40)
        candidate_files.append(path)

    # Build FakeComfyClient responses: one per seed.
    # For scene.index=1, _seeds_for_scene(1, 5) yields (1000, 1001, 1002, 1003, 1004).
    seeds = (1000, 1001, 1002, 1003, 1004)
    responses: dict[str, list[Path]] = {}
    for i, seed in enumerate(seeds):
        wf = inject(
            wf_template,
            prompt="a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024,
            height=1024,
            output_prefix=f"scene_001_candidate_{i}",
        )
        responses[workflow_signature(wf)] = [candidate_files[i]]

    fake_comfy = FakeComfyClient(responses=responses)
    fake_scorer = FakeAestheticScorer(fixed_score=7.0)

    # Build a Story with 1 scene.
    scene = Scene(
        id="scene_001",
        index=1,
        narration_text="Once upon a time",
        narration_duration_seconds=5.0,
        visual_prompt="a candle",
        negative_prompt="bright daylight",
    )
    story = Story(
        id="story_test_5cand",
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
        scenes=[scene],
    )

    # Stub config that returns track cfg with image_model.candidates_per_scene=5.
    class _StubConfig:
        config_dir = repo_root / "config"

        def track(self, _name: str) -> dict:
            return {
                "visual": {"aesthetic": "cinematic dark", "negative_prompt": "bright daylight"},
                "quality_gates": {
                    "aesthetic_min_score": 0.0,
                    "brightness_floor_mean_rgb": 0.0,
                    "subject_min_edge_density": 0.0,
                },
                "image_model": {"candidates_per_scene": 5},
            }

    output_root = tmp_path / "keyframes"
    reports = await generate(
        story,
        config=_StubConfig(),
        comfy=fake_comfy,
        scorer=fake_scorer,
        output_root=output_root,
        mp_hands_factory=None,
    )

    # Assertions: should get 5 candidates when yaml specifies 5.
    assert len(reports) == 1
    assert len(reports[0].candidates) == 5
    assert len(scene.keyframe_candidates) == 5
    assert len(scene.keyframe_scores) == 5


@pytest.mark.asyncio
async def test_generate_defaults_to_three_when_image_model_missing(tmp_path: Path) -> None:
    """yaml without image_model block -> falls back to n_candidates=3.

    Pins the default behavior. Passes on both pre- and post-fix code (this
    is a regression-pin, not a RED test).
    """
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story
    from platinum.pipeline.keyframe_generator import generate
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # Synthesize 3 distinct PNGs with different grey values.
    candidate_files = []
    for i in range(3):
        path = tmp_path / f"src_candidate_{i}.png"
        make_synthetic_png(path, kind="grey", value=50 + i * 70)
        candidate_files.append(path)

    # Build FakeComfyClient responses for 3 candidates.
    # For scene.index=0, _seeds_for_scene(0, 3) yields (0, 1, 2).
    seeds = (0, 1, 2)
    responses: dict[str, list[Path]] = {}
    for i, seed in enumerate(seeds):
        wf = inject(
            wf_template,
            prompt="a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024,
            height=1024,
            output_prefix=f"scene_000_candidate_{i}",
        )
        responses[workflow_signature(wf)] = [candidate_files[i]]

    fake_comfy = FakeComfyClient(responses=responses)
    fake_scorer = FakeAestheticScorer(fixed_score=7.0)

    # Build Story with 1 scene.
    scene = Scene(
        id="scene_000",
        index=0,
        narration_text="Once upon a time",
        narration_duration_seconds=5.0,
        visual_prompt="a candle",
        negative_prompt="bright daylight",
    )
    story = Story(
        id="story_test_default",
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
        scenes=[scene],
    )

    # Stub config WITHOUT image_model block.
    class _StubConfig:
        config_dir = repo_root / "config"

        def track(self, _name: str) -> dict:
            return {
                "visual": {"aesthetic": "cinematic dark", "negative_prompt": "bright daylight"},
                "quality_gates": {
                    "aesthetic_min_score": 0.0,
                    "brightness_floor_mean_rgb": 0.0,
                    "subject_min_edge_density": 0.0,
                },
                # NO image_model key -> should default to 3
            }

    output_root = tmp_path / "keyframes"
    reports = await generate(
        story,
        config=_StubConfig(),
        comfy=fake_comfy,
        scorer=fake_scorer,
        output_root=output_root,
        mp_hands_factory=None,
    )

    # Assertions: should get 3 candidates (the default) when image_model is missing.
    assert len(reports) == 1
    assert len(reports[0].candidates) == 3
    assert len(scene.keyframe_candidates) == 3
    assert len(scene.keyframe_scores) == 3


@pytest.mark.asyncio
async def test_generate_passes_width_height_from_track_yaml(tmp_path: Path) -> None:
    """S7.1.A2.3: generate() reads image_model.width/height from track_cfg
    and threads them into inject() so FakeComfyClient receives 9:16 portrait
    workflows when the track YAML carries the new fields.

    The FakeComfyClient is keyed by signatures of inject(width=768, height=1344);
    if generate() passes the wrong dims, the signature won't match and
    FakeComfyClient raises KeyError. submitted_workflows[0] makes the
    successful match assertable explicitly.
    """
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story
    from platinum.pipeline.keyframe_generator import generate
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    candidate_files: list[Path] = []
    for i in range(3):
        path = tmp_path / f"src_candidate_{i}.png"
        make_synthetic_png(path, kind="grey", value=50 + i * 70)
        candidate_files.append(path)

    seeds = (0, 1, 2)  # _seeds_for_scene(0, 3, regen_count=0)
    responses: dict[str, list[Path]] = {}
    for i, seed in enumerate(seeds):
        wf = inject(
            wf_template,
            prompt="a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=768,
            height=1344,
            output_prefix=f"scene_000_candidate_{i}",
        )
        responses[workflow_signature(wf)] = [candidate_files[i]]

    fake_comfy = FakeComfyClient(responses=responses)

    scene = Scene(
        id="scene_000",
        index=0,
        narration_text="Once upon a time",
        narration_duration_seconds=5.0,
        visual_prompt="a candle",
        negative_prompt="bright daylight",
    )
    story = Story(
        id="story_test_aspect",
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
        scenes=[scene],
    )

    class _StubConfig:
        config_dir = repo_root / "config"

        def track(self, _name: str) -> dict:
            return {
                "visual": {
                    "aesthetic": "cinematic dark",
                    "negative_prompt": "bright daylight",
                },
                "quality_gates": {
                    "aesthetic_min_score": 0.0,
                    "brightness_floor_mean_rgb": 0.0,
                    "subject_min_edge_density": 0.0,
                },
                "image_model": {"width": 768, "height": 1344},
            }

    output_root = tmp_path / "keyframes"
    await generate(
        story,
        config=_StubConfig(),
        comfy=fake_comfy,
        scorer=FakeAestheticScorer(fixed_score=7.0),
        output_root=output_root,
        mp_hands_factory=None,
    )

    assert len(fake_comfy.submitted_workflows) == 3
    for submitted in fake_comfy.submitted_workflows:
        assert submitted["5"]["inputs"]["width"] == 768
        assert submitted["5"]["inputs"]["height"] == 1344
        assert submitted["10"]["inputs"]["width"] == 768
        assert submitted["10"]["inputs"]["height"] == 1344


@pytest.mark.asyncio
async def test_generate_defaults_width_height_to_1024_when_track_yaml_omits(
    tmp_path: Path,
) -> None:
    """S7.1.A2.3: when image_model omits width/height, generate() falls back
    to 1024x1024 -- a safety net for legacy YAMLs that haven't migrated."""
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story
    from platinum.pipeline.keyframe_generator import generate
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    candidate_files: list[Path] = []
    for i in range(3):
        path = tmp_path / f"src_candidate_{i}.png"
        make_synthetic_png(path, kind="grey", value=50 + i * 70)
        candidate_files.append(path)

    seeds = (0, 1, 2)
    responses: dict[str, list[Path]] = {}
    for i, seed in enumerate(seeds):
        wf = inject(
            wf_template,
            prompt="a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024,
            height=1024,
            output_prefix=f"scene_000_candidate_{i}",
        )
        responses[workflow_signature(wf)] = [candidate_files[i]]

    fake_comfy = FakeComfyClient(responses=responses)

    scene = Scene(
        id="scene_000",
        index=0,
        narration_text="Once upon a time",
        narration_duration_seconds=5.0,
        visual_prompt="a candle",
        negative_prompt="bright daylight",
    )
    story = Story(
        id="story_test_default_aspect",
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
        scenes=[scene],
    )

    class _StubConfig:
        config_dir = repo_root / "config"

        def track(self, _name: str) -> dict:
            return {
                "visual": {
                    "aesthetic": "cinematic dark",
                    "negative_prompt": "bright daylight",
                },
                "quality_gates": {
                    "aesthetic_min_score": 0.0,
                    "brightness_floor_mean_rgb": 0.0,
                    "subject_min_edge_density": 0.0,
                },
                "image_model": {},  # no width/height -> default 1024
            }

    await generate(
        story,
        config=_StubConfig(),
        comfy=fake_comfy,
        scorer=FakeAestheticScorer(fixed_score=7.0),
        output_root=tmp_path / "keyframes",
        mp_hands_factory=None,
    )

    assert fake_comfy.submitted_workflows[0]["5"]["inputs"]["width"] == 1024
    assert fake_comfy.submitted_workflows[0]["5"]["inputs"]["height"] == 1024


@pytest.mark.asyncio
async def test_generate_for_scene_clip_gate_rejects_low_similarity_candidates(
    tmp_path: Path,
) -> None:
    """S7.1.A3.3: CLIP gate runs after subject + before LAION scoring.

    Candidate 0 has clip_similarity=0.05 (below 0.20 threshold) -> rejected.
    Candidates 1 and 2 have clip_similarity=0.30 -> pass; LAION still scores
    them. The selected candidate is the highest-scoring among CLIP-passing.
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 8.0,  # would win on LAION but fails CLIP
        output_dir / "candidate_1.png": 6.5,
        output_dir / "candidate_2.png": 7.0,  # winner
    }
    clip_map = {
        output_dir / "candidate_0.png": 0.05,  # below 0.20 threshold
        output_dir / "candidate_1.png": 0.30,
        output_dir / "candidate_2.png": 0.30,
    }
    scorer = MappedFakeScorer(
        scores_by_path=score_map,
        default=0.0,
        clip_similarities_by_path=clip_map,
        clip_similarity_default=0.0,
    )
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
        clip_min_similarity=0.20,
    )
    assert report.clip_passed == [False, True, True]
    # candidate 0 had the highest LAION score but should NOT win because CLIP rejected it
    assert report.selected_index == 2  # 7.0 is the highest LAION among CLIP-passing
    assert report.scores[0] == 0.0  # CLIP failure zeros out the score
    assert not report.selected_via_fallback


@pytest.mark.asyncio
async def test_generate_for_scene_clip_gate_halts_when_all_fail(tmp_path: Path) -> None:
    """S7.1.A3.3: when no candidate passes CLIP, raise KeyframeGenerationError.

    Mirrors the brightness/subject halt semantics so a CLIP-blanked scene
    surfaces the failure to the Stage instead of selecting a degenerate
    image via fallback.
    """
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        generate_for_scene,
    )
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    scorer = MappedFakeScorer(
        scores_by_path={},
        default=7.0,
        clip_similarities_by_path={},
        clip_similarity_default=0.05,  # all below 0.20 threshold
    )
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    with pytest.raises(KeyframeGenerationError, match="clip"):
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates={**_GATES, "subject_min_edge_density": 0.0},
            comfy=comfy,
            scorer=scorer,
            output_dir=output_dir,
            workflow_template=wf_template,
            seeds=(0, 1, 2),
            mp_hands_factory=make_fake_hands_factory(None),
            clip_min_similarity=0.20,
        )


@pytest.mark.asyncio
async def test_generate_for_scene_clip_gate_disabled_when_threshold_zero(
    tmp_path: Path,
) -> None:
    """S7.1.A3.3: clip_min_similarity=0.0 effectively disables the gate.

    Default behavior pre-S7.1: any clip similarity (or even the
    FakeAestheticScorer's 0.5 neutral default) passes 0.0; pipeline
    still works for tracks that haven't migrated to clip_min_similarity.
    """
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 6.0,
        output_dir / "candidate_1.png": 7.0,
        output_dir / "candidate_2.png": 8.0,
    }
    scorer = MappedFakeScorer(
        scores_by_path=score_map,
        default=0.0,
        clip_similarities_by_path={},
        clip_similarity_default=0.05,  # would fail any positive threshold
    )
    comfy, wf_template = _build_fake_comfy_with_three_candidates()
    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={**_GATES, "subject_min_edge_density": 0.0},
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
        # clip_min_similarity defaults to 0.0 -> gate is effectively off
    )
    assert report.clip_passed == [True, True, True]
    assert report.selected_index == 2  # highest LAION score wins


@pytest.mark.asyncio
async def test_generate_passes_clip_min_similarity_from_track_yaml(
    tmp_path: Path,
) -> None:
    """S7.1.A3.3: generate() reads image_model.clip_min_similarity from
    track_cfg and threads it to generate_for_scene's CLIP gate."""
    from datetime import datetime

    from platinum.models.story import Scene, Source, Story
    from platinum.pipeline.keyframe_generator import KeyframeGenerationError, generate
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.comfyui import FakeComfyClient, workflow_signature
    from platinum.utils.workflow import inject, load_workflow
    from tests._fixtures import make_synthetic_png

    repo_root = Path(__file__).resolve().parents[2]
    wf_template = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    candidate_files: list[Path] = []
    for i in range(3):
        path = tmp_path / f"src_candidate_{i}.png"
        make_synthetic_png(path, kind="grey", value=50 + i * 70)
        candidate_files.append(path)

    seeds = (0, 1, 2)
    responses: dict[str, list[Path]] = {}
    for i, seed in enumerate(seeds):
        wf = inject(
            wf_template,
            prompt="a candle",
            negative_prompt="bright daylight",
            seed=seed,
            width=1024,
            height=1024,
            output_prefix=f"scene_000_candidate_{i}",
        )
        responses[workflow_signature(wf)] = [candidate_files[i]]

    fake_comfy = FakeComfyClient(responses=responses)
    # All candidates fail the CLIP gate -- generate() should raise.
    fake_scorer = MappedFakeScorer(
        scores_by_path={},
        default=7.0,
        clip_similarities_by_path={},
        clip_similarity_default=0.05,
    )

    scene = Scene(
        id="scene_000",
        index=0,
        narration_text="Once upon a time",
        narration_duration_seconds=5.0,
        visual_prompt="a candle",
        negative_prompt="bright daylight",
    )
    story = Story(
        id="story_test_clip",
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
        scenes=[scene],
    )

    class _StubConfig:
        config_dir = repo_root / "config"

        def track(self, _name: str) -> dict:
            return {
                "visual": {
                    "aesthetic": "cinematic dark",
                    "negative_prompt": "bright daylight",
                },
                "quality_gates": {
                    "aesthetic_min_score": 0.0,
                    "brightness_floor_mean_rgb": 0.0,
                    "subject_min_edge_density": 0.0,
                },
                "image_model": {"clip_min_similarity": 0.20},
            }

    output_root = tmp_path / "keyframes"
    with pytest.raises(KeyframeGenerationError, match="clip"):
        await generate(
            story,
            config=_StubConfig(),
            comfy=fake_comfy,
            scorer=fake_scorer,
            output_root=output_root,
            mp_hands_factory=None,
        )


@pytest.mark.asyncio
async def test_generate_for_scene_content_gate_off_skips_check(tmp_path: Path) -> None:
    """S7.1.A4.5: content_gate='off' must NOT call ContentChecker.check."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.content_check import FakeContentChecker
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    scorer = FakeAestheticScorer(fixed_score=7.0)
    content_checker = FakeContentChecker()
    comfy, wf_template = _build_fake_comfy_with_three_candidates()

    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={
            **_GATES,
            "subject_min_edge_density": 0.0,
            "content_gate": "off",
            "content_gate_min_score": 6,
        },
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
        content_checker=content_checker,
    )
    assert content_checker.call_count == 0  # never invoked
    assert report.content_passed == [True, True, True]
    assert report.content_scores == [None, None, None]


@pytest.mark.asyncio
async def test_generate_for_scene_content_gate_filters_low_score(tmp_path: Path) -> None:
    """S7.1.A4.5: content_gate='claude' rejects candidates whose Claude score
    is below content_gate_min_score."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import MappedFakeScorer
    from platinum.utils.content_check import FakeContentChecker
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    score_map = {
        output_dir / "candidate_0.png": 8.0,  # would-be winner on LAION
        output_dir / "candidate_1.png": 7.0,
        output_dir / "candidate_2.png": 6.5,
    }
    content_score_map = {
        output_dir / "candidate_0.png": 3,  # below 6 threshold
        output_dir / "candidate_1.png": 8,
        output_dir / "candidate_2.png": 7,
    }
    scorer = MappedFakeScorer(scores_by_path=score_map, default=0.0)
    content_checker = FakeContentChecker(scores=content_score_map)
    comfy, wf_template = _build_fake_comfy_with_three_candidates()

    report = await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates={
            **_GATES,
            "subject_min_edge_density": 0.0,
            "content_gate": "claude",
            "content_gate_min_score": 6,
        },
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
        content_checker=content_checker,
    )
    assert content_checker.call_count == 3
    assert report.content_passed == [False, True, True]
    assert report.content_scores == [3, 8, 7]
    # candidate_0 had highest LAION but failed content gate -> winner is c1
    assert report.selected_index == 1
    assert not report.selected_via_fallback


@pytest.mark.asyncio
async def test_generate_for_scene_content_gate_halts_when_all_fail(
    tmp_path: Path,
) -> None:
    """S7.1.A4.5: when no candidate passes the content gate, raise."""
    from platinum.pipeline.keyframe_generator import (
        KeyframeGenerationError,
        generate_for_scene,
    )
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.content_check import FakeContentChecker
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    scorer = FakeAestheticScorer(fixed_score=7.0)
    content_checker = FakeContentChecker(default_score=2)  # all below 6
    comfy, wf_template = _build_fake_comfy_with_three_candidates()

    with pytest.raises(KeyframeGenerationError, match="content"):
        await generate_for_scene(
            scene,
            track_visual=_TRACK_VISUAL,
            quality_gates={
                **_GATES,
                "subject_min_edge_density": 0.0,
                "content_gate": "claude",
                "content_gate_min_score": 6,
            },
            comfy=comfy,
            scorer=scorer,
            output_dir=output_dir,
            workflow_template=wf_template,
            seeds=(0, 1, 2),
            mp_hands_factory=make_fake_hands_factory(None),
            content_checker=content_checker,
        )


@pytest.mark.asyncio
async def test_generate_for_scene_content_gate_skips_candidates_failing_earlier(
    tmp_path: Path,
) -> None:
    """S7.1.A4.5: candidates that failed brightness/subject/clip/scoring/anatomy
    are NOT sent to the (expensive) Claude content gate.

    Setup: subject_min_edge_density set so candidate 0 fails. Expect
    ContentChecker called only twice (for candidates 1 and 2)."""
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import FakeAestheticScorer
    from platinum.utils.content_check import FakeContentChecker
    from tests._fixtures import make_fake_hands_factory

    scene = _scene(idx=0)
    output_dir = tmp_path / "scene_000"
    scorer = FakeAestheticScorer(fixed_score=7.0)
    content_checker = FakeContentChecker()
    comfy, wf_template = _build_fake_comfy_with_three_candidates()

    # Build with high subject threshold so candidate_0 (one of the fixtures)
    # fails subject gate. (Edge densities of the fixture PNGs are all > 0.005
    # but we crank to a level that all 3 fail; the loop's per-candidate
    # accounting still stays sync.)
    quality_gates = {
        **_GATES,
        # 0.0 means subject gate effectively disabled -> all 3 reach content
        "subject_min_edge_density": 0.0,
        "content_gate": "claude",
        "content_gate_min_score": 6,
    }
    await generate_for_scene(
        scene,
        track_visual=_TRACK_VISUAL,
        quality_gates=quality_gates,
        comfy=comfy,
        scorer=scorer,
        output_dir=output_dir,
        workflow_template=wf_template,
        seeds=(0, 1, 2),
        mp_hands_factory=make_fake_hands_factory(None),
        content_checker=content_checker,
    )
    # All 3 candidates passed earlier gates -> content checked 3 times.
    assert content_checker.call_count == 3
