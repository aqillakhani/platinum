"""Keyframe generator pipeline -- per-scene Flux candidate generation + selection.

For each Scene:
  1. Generate N candidates (default 3) via ComfyClient at deterministic seeds.
  2. Score each via AestheticScorer; check_hand_anomalies for anatomy gate.
  3. Select the highest-scoring candidate that passes both gates;
     fall back to candidate 0 if none qualify.

Pure functions (`generate_for_scene`, `generate`) take all dependencies as
args; the impure `KeyframeGeneratorStage` pulls dependencies from `ctx`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from platinum.pipeline.stage import Stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class KeyframeReport:
    """Per-scene generation report. Persisted onto Scene fields by the Stage."""

    scene_index: int
    candidates: list[Path]
    scores: list[float]
    anatomy_passed: list[bool]
    scoring_succeeded: list[bool]
    brightness_passed: list[bool]
    subject_passed: list[bool]
    selected_index: int
    selected_via_fallback: bool


class KeyframeGenerationError(RuntimeError):
    """Raised when ALL candidates for a scene threw during generation.

    Carries the per-candidate exception list so the Stage can record a
    useful error string in the StageRun log.
    """

    def __init__(self, *, scene_index: int, exceptions: Sequence[BaseException]) -> None:
        self.scene_index = scene_index
        self.exceptions = list(exceptions)
        joined = "; ".join(f"{type(e).__name__}: {e}" for e in self.exceptions)
        super().__init__(f"all candidates failed for scene_index={scene_index}: {joined}")


def _seeds_for_scene(scene_index: int, n: int) -> tuple[int, ...]:
    """Deterministic seeds: scene_index*1000 + offset."""
    return tuple(scene_index * 1000 + i for i in range(n))


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


async def generate_for_scene(
    scene: Any,
    *,
    track_visual: dict[str, Any],
    quality_gates: dict[str, Any],
    comfy: Any,
    scorer: Any,
    output_dir: Path,
    workflow_template: dict[str, Any] | None = None,
    config_dir: Path | None = None,
    n_candidates: int = 3,
    seeds: Sequence[int] | None = None,
    width: int = 1024,
    height: int = 1024,
    mp_hands_factory: Callable[[], Any] | None = None,
) -> KeyframeReport:
    """Generate N candidates, score + anatomy-check each, return a KeyframeReport.

    `workflow_template` is loaded from `config_dir/workflows/flux_dev_keyframe.json`
    if None. `seeds` defaults to `_seeds_for_scene(scene.index, n_candidates)`.
    """
    from platinum.utils.validate import check_hand_anomalies
    from platinum.utils.workflow import inject, load_workflow

    if not scene.visual_prompt:
        raise ValueError(
            f"scene {scene.index} has no visual_prompt; run `platinum adapt` first"
        )
    if workflow_template is None:
        if config_dir is None:
            raise ValueError("either workflow_template or config_dir is required")
        workflow_template = load_workflow("flux_dev_keyframe", config_dir=config_dir)
    use_seeds = tuple(seeds) if seeds is not None else _seeds_for_scene(scene.index, n_candidates)
    if len(use_seeds) != n_candidates:
        raise ValueError(f"seeds length {len(use_seeds)} != n_candidates {n_candidates}")

    output_dir.mkdir(parents=True, exist_ok=True)

    aesthetic_text = " ".join(
        s for s in (track_visual.get("aesthetic"), scene.visual_prompt) if s
    )
    negative_text = scene.negative_prompt or track_visual.get("negative_prompt", "")

    candidate_paths: list[Path] = []
    candidate_exceptions: list[BaseException | None] = []
    for i, seed in enumerate(use_seeds):
        wf = inject(
            workflow_template,
            prompt=aesthetic_text,
            negative_prompt=negative_text,
            seed=seed,
            width=width,
            height=height,
            output_prefix=f"scene_{scene.index:03d}_candidate_{i}",
        )
        path = output_dir / f"candidate_{i}.png"
        try:
            await comfy.generate_image(workflow=wf, output_path=path)
            candidate_paths.append(path)
            candidate_exceptions.append(None)
        except Exception as exc:  # noqa: BLE001 -- per-candidate isolation is intentional
            logger.warning(
                "scene %d candidate %d generate_image failed: %r", scene.index, i, exc
            )
            candidate_paths.append(path)
            candidate_exceptions.append(exc)

    if all(e is not None for e in candidate_exceptions):
        raise KeyframeGenerationError(
            scene_index=scene.index,
            exceptions=[e for e in candidate_exceptions if e is not None],
        )

    # Read brightness floor from quality_gates (default 20.0 -- atmospheric_horror floor).
    brightness_floor = float(quality_gates.get("brightness_floor_mean_rgb", 20.0))
    subject_floor = float(quality_gates.get("subject_min_edge_density", 0.020))

    scores: list[float] = []
    anatomy_passed: list[bool] = []
    scoring_succeeded: list[bool] = []
    brightness_passed: list[bool] = []
    subject_passed: list[bool] = []
    for path, candidate_exc in zip(candidate_paths, candidate_exceptions, strict=True):
        if candidate_exc is not None or not path.exists():
            scores.append(0.0)
            anatomy_passed.append(False)
            scoring_succeeded.append(False)
            brightness_passed.append(False)
            subject_passed.append(False)
            continue

        # NEW: brightness gate runs BEFORE the LAION call (saves the round-trip).
        from platinum.utils.validate import check_image_brightness, check_image_subject
        bright = check_image_brightness(path, min_mean_rgb=brightness_floor)
        brightness_passed.append(bright.passed)
        if not bright.passed:
            logger.warning(
                "scene %d candidate %s brightness floor failed: %s",
                scene.index, path.name, bright.reason,
            )
            scores.append(0.0)
            anatomy_passed.append(False)
            scoring_succeeded.append(False)
            subject_passed.append(False)
            continue

        subject = check_image_subject(path, min_edge_density=subject_floor)
        subject_passed.append(subject.passed)
        if not subject.passed:
            logger.warning(
                "scene %d candidate %s subject gate failed: %s",
                scene.index, path.name, subject.reason,
            )
            scores.append(0.0)
            anatomy_passed.append(False)
            scoring_succeeded.append(False)
            continue

        try:
            score = await scorer.score(path)
            scoring_ok = True
        except Exception as scorer_exc:  # noqa: BLE001
            logger.warning("scorer failed for %s: %r", path, scorer_exc)
            score = 0.0
            scoring_ok = False
        if not _is_finite(score):
            score = 0.0
            scoring_ok = False
        scores.append(float(score))
        scoring_succeeded.append(scoring_ok)
        result = check_hand_anomalies(path, mp_hands_factory=mp_hands_factory)
        anatomy_passed.append(result.passed)

    if not any(brightness_passed):
        raise KeyframeGenerationError(
            scene_index=scene.index,
            exceptions=[
                RuntimeError(f"brightness floor failed for candidate {i} (path={p})")
                for i, p in enumerate(candidate_paths)
            ],
        )

    if not any(subject_passed):
        raise KeyframeGenerationError(
            scene_index=scene.index,
            exceptions=[
                RuntimeError(f"subject gate failed for candidate {i} (path={p})")
                for i, p in enumerate(candidate_paths)
            ],
        )

    if not any(scoring_succeeded):
        raise KeyframeGenerationError(
            scene_index=scene.index,
            exceptions=[
                RuntimeError(f"scoring failed for candidate {i} (path={p})")
                for i, p in enumerate(candidate_paths)
            ],
        )

    threshold = float(quality_gates.get("aesthetic_min_score", 0.0))
    eligible = [
        i for i, (s, a, ok, b, sj) in enumerate(
            zip(scores, anatomy_passed, scoring_succeeded, brightness_passed,
                subject_passed, strict=True)
        )
        if ok and b and sj and s >= threshold and a
    ]
    if eligible:
        max_score = max(scores[i] for i in eligible)
        selected_index = next(i for i in eligible if scores[i] == max_score)
        selected_via_fallback = False
    else:
        # Content failure: fall back to highest-scored among candidates that
        # passed scoring AND brightness AND subject gates. Failing candidates
        # are NEVER selected, even in fallback (would persist a degenerate
        # image even though we know it's bad).
        fallback_pool = [
            i for i, (ok, b, sj) in enumerate(
                zip(scoring_succeeded, brightness_passed, subject_passed, strict=True)
            )
            if ok and b and sj
        ]
        # If empty, the brightness/subject/scoring halt above already raised
        # KeyframeGenerationError; reaching here guarantees fallback_pool is non-empty.
        max_score_in_pool = max(scores[i] for i in fallback_pool)
        selected_index = next(
            i for i in fallback_pool if scores[i] == max_score_in_pool
        )
        selected_via_fallback = True

    return KeyframeReport(
        scene_index=scene.index,
        candidates=candidate_paths,
        scores=scores,
        anatomy_passed=anatomy_passed,
        scoring_succeeded=scoring_succeeded,
        brightness_passed=brightness_passed,
        subject_passed=subject_passed,
        selected_index=selected_index,
        selected_via_fallback=selected_via_fallback,
    )


async def generate(
    story: Any,
    *,
    config: Any,
    comfy: Any,
    scorer: Any,
    output_root: Path,
    mp_hands_factory: Callable[[], Any] | None = None,
    scene_filter: set[int] | None = None,
) -> list[KeyframeReport]:
    """Run keyframe generation for every scene whose keyframe_path is None.

    Mutates each scene in-place: keyframe_candidates, keyframe_scores,
    keyframe_path, validation["keyframe_anatomy"],
    validation["keyframe_selected_via_fallback"].

    If scene_filter is set, only scenes whose index is in scene_filter are processed.
    """
    from platinum.utils.workflow import load_workflow

    track_cfg = config.track(story.track)
    track_visual = dict(track_cfg.get("visual", {}))
    quality_gates = dict(track_cfg.get("quality_gates", {}))
    workflow_template = load_workflow("flux_dev_keyframe", config_dir=config.config_dir)

    reports: list[KeyframeReport] = []
    for scene in story.scenes:
        if scene.keyframe_path is not None:
            logger.info(
                "scene %d already has keyframe_path=%s; skipping (resume)",
                scene.index,
                scene.keyframe_path,
            )
            continue
        if scene_filter is not None and scene.index not in scene_filter:
            logger.info("scene %d not in scene_filter; skipping", scene.index)
            continue
        scene_dir = output_root / f"scene_{scene.index:03d}"
        report = await generate_for_scene(
            scene,
            track_visual=track_visual,
            quality_gates=quality_gates,
            comfy=comfy,
            scorer=scorer,
            output_dir=scene_dir,
            workflow_template=workflow_template,
            mp_hands_factory=mp_hands_factory,
        )
        scene.keyframe_candidates = list(report.candidates)
        scene.keyframe_scores = list(report.scores)
        scene.keyframe_path = report.candidates[report.selected_index]
        scene.validation["keyframe_anatomy"] = list(report.anatomy_passed)
        scene.validation["keyframe_selected_via_fallback"] = (
            report.selected_via_fallback
        )
        reports.append(report)
        logger.info(
            "scene %d selected candidate %d (score=%.2f, fallback=%s)",
            scene.index,
            report.selected_index,
            report.scores[report.selected_index],
            report.selected_via_fallback,
        )
    return reports


class KeyframeGeneratorStage(Stage):
    """Per-story keyframe generation stage. See module docstring."""

    name: ClassVar[str] = "keyframe_generator"

    async def run(self, story: Any, ctx: Any) -> dict[str, Any]:
        from platinum.utils.aesthetics import RemoteAestheticScorer
        from platinum.utils.comfyui import HttpComfyClient

        test_overrides = ctx.config.settings.get("test", {})
        injected_comfy = test_overrides.get("comfy_client")
        injected_scorer = test_overrides.get("aesthetic_scorer")
        comfy = injected_comfy or HttpComfyClient(
            host=ctx.config.settings.get("comfyui", {}).get(
                "host", "http://localhost:8188"
            ),
        )
        scorer = injected_scorer or RemoteAestheticScorer(
            host=ctx.config.settings.get("aesthetics", {}).get("host", "")
        )
        mp_hands_factory = test_overrides.get("mp_hands_factory")

        # Compute output_root: prefer the story_path-derived dir if ctx supports it,
        # else fall back to data/stories/<id>/keyframes/.
        story_dir: Path
        if hasattr(ctx, "story_path"):
            try:
                story_dir = Path(ctx.story_path(story)).parent
            except Exception:  # noqa: BLE001 -- defensive fallback
                story_dir = Path("data/stories") / story.id
        else:
            story_dir = Path("data/stories") / story.id
        output_root = story_dir / "keyframes"

        runtime_overrides = ctx.config.settings.get("runtime", {})
        scene_filter_raw = runtime_overrides.get("scene_filter")
        scene_filter: set[int] | None = (
            set(scene_filter_raw) if scene_filter_raw is not None else None
        )

        scenes_total = len(story.scenes)
        try:
            try:
                reports = await generate(
                    story,
                    config=ctx.config,
                    comfy=comfy,
                    scorer=scorer,
                    output_root=output_root,
                    mp_hands_factory=mp_hands_factory,
                    scene_filter=scene_filter,
                )
            except KeyframeGenerationError:
                # Save what we have so far before re-raising.
                try:
                    if hasattr(ctx, "story_path"):
                        story.save(ctx.story_path(story))
                except Exception:  # noqa: BLE001 -- save best-effort on failure path
                    logger.exception(
                        "failed to save story.json after KeyframeGenerationError"
                    )
                raise

            # Atomic save at end of stage.
            try:
                if hasattr(ctx, "story_path"):
                    story.save(ctx.story_path(story))
            except Exception:  # noqa: BLE001 -- save best-effort
                logger.exception("failed to save story.json after stage completion")
        finally:
            # Close any clients the Stage constructed (skip test-injected ones --
            # tests own their lifecycle). Best-effort: never let cleanup errors
            # mask the original exception.
            if injected_scorer is None and hasattr(scorer, "aclose"):
                try:
                    await scorer.aclose()
                except Exception:  # noqa: BLE001
                    logger.exception("failed to aclose RemoteAestheticScorer")

        scenes_via_fallback = sum(1 for r in reports if r.selected_via_fallback)
        scenes_succeeded_now = sum(1 for s in story.scenes if s.keyframe_path is not None)
        return {
            "scenes_total": scenes_total,
            "scenes_succeeded": scenes_succeeded_now,
            "scenes_via_fallback": scenes_via_fallback,
            "scenes_failed": scenes_total - scenes_succeeded_now,
            "reports": [
                {
                    "scene_index": r.scene_index,
                    "selected_index": r.selected_index,
                    "selected_score": r.scores[r.selected_index],
                    "selected_via_fallback": r.selected_via_fallback,
                }
                for r in reports
            ],
        }
