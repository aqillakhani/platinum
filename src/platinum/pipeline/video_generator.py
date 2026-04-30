"""Video generator pipeline -- per-scene Wan 2.2 I2V (S8 Phase A).

For each Scene with a populated keyframe_path:
  1. Upload keyframe to ComfyUI.
  2. Submit Wan 2.2 I2V workflow with keyframe + visual_prompt as conditioning.
  3. Run quality gates (duration_match, black_frames, motion).
  4. On content failure: retry ONCE with new seed.
  5. On 2nd content fail or any infrastructure failure: raise VideoGenerationError.

Pure functions (`generate_video_for_scene`, `generate_video`) take all
dependencies as args; the impure `VideoGeneratorStage` pulls dependencies
from `ctx`. Per-scene atomic save with resume via Scene.video_path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoReport:
    """Per-scene generation report. Persisted onto Scene fields by the Stage."""

    scene_index: int
    success: bool
    mp4_path: Path | None
    duration_seconds: float
    gates_passed: dict[str, bool]
    retry_used: int


class VideoGenerationError(RuntimeError):
    """Raised when a scene's video generation cannot complete.

    Carries a structured reason and a `retryable` flag distinguishing
    infrastructure failures (network, OOM, missing weights -- not retryable
    by re-seeding) from content failures (gates fail twice -- in principle
    retryable by the user with fresh seeds).
    """

    def __init__(
        self,
        *,
        scene_index: int,
        reason: str,
        retryable: bool = False,
    ) -> None:
        self.scene_index = scene_index
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"scene_index={scene_index}: {reason} (retryable={retryable})")


REQUIRED_GATES_KEYS = (
    "duration_target_seconds",
    "duration_tolerance_seconds",
    "black_frame_max_ratio",
    "motion_min_flow",
)


def _seed_for_scene(scene_index: int, *, retry: int = 0) -> int:
    """Deterministic seed for a scene's video generation.

    seed = scene_index * 1000 + retry. With retry in {0, 1} only two seeds
    are ever used per scene (initial + one retry).
    """
    return scene_index * 1000 + retry


async def generate_video_for_scene(
    scene,  # platinum.models.story.Scene
    *,
    workflow_template: dict,
    comfy,                # ComfyClient (Fake or Http)
    output_path: Path,
    gates_cfg: dict,
    width: int = 1280,
    height: int = 720,
    frame_count: int = 80,
    fps: int = 16,
) -> VideoReport:
    """Generate one Wan 2.2 I2V clip for a single scene.

    Runs 3 quality gates after generation: duration, black_frames, motion.
    On content failure, retries once with a new seed. Returns VideoReport
    on success or raises VideoGenerationError on infra failure or 2nd
    content fail.
    """
    from platinum.utils.validate import (
        check_black_frames,
        check_duration_match,
        check_motion,
    )
    from platinum.utils.workflow import inject_video

    if scene.keyframe_path is None:
        raise VideoGenerationError(
            scene_index=scene.index,
            reason=f"scene {scene.index} has no keyframe_path",
            retryable=False,
        )
    if not Path(scene.keyframe_path).exists():
        raise VideoGenerationError(
            scene_index=scene.index,
            reason=f"keyframe_path does not exist: {scene.keyframe_path}",
            retryable=False,
        )

    # 1. Upload keyframe to ComfyUI.
    server_filename = await comfy.upload_image(Path(scene.keyframe_path))

    # 2. Try up to twice: initial attempt (retry=0) and one retry (retry=1).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_reasons: list[str] = []

    for retry in (0, 1):
        seed = _seed_for_scene(scene.index, retry=retry)
        workflow = inject_video(
            workflow_template,
            image_in=server_filename,
            prompt=scene.visual_prompt or "",
            seed=seed,
            output_prefix=f"scene_{scene.index:03d}_raw",
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
        )

        # 3. Submit & download.
        try:
            await comfy.generate_image(workflow=workflow, output_path=output_path)
        except VideoGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 -- infra failures bubble out
            raise VideoGenerationError(
                scene_index=scene.index,
                reason=f"comfy generate_image failed: {exc!r}",
                retryable=False,
            ) from exc

        # 4. Run quality gates in order: duration -> black_frames -> motion.
        target_s = float(gates_cfg["duration_target_seconds"])
        tol_s = float(gates_cfg["duration_tolerance_seconds"])
        duration_result = check_duration_match(
            output_path, target_seconds=target_s, tolerance_seconds=tol_s
        )
        black_result = check_black_frames(
            output_path,
            max_black_ratio=float(gates_cfg["black_frame_max_ratio"]),
        )
        motion_result = check_motion(
            output_path,
            min_flow_magnitude=float(gates_cfg["motion_min_flow"]),
        )

        gates_passed = {
            "duration": duration_result.passed,
            "black_frames": black_result.passed,
            "motion": motion_result.passed,
        }

        if all(gates_passed.values()):
            # All gates passed; return success.
            return VideoReport(
                scene_index=scene.index,
                success=True,
                mp4_path=output_path,
                duration_seconds=float(duration_result.metric),
                gates_passed=gates_passed,
                retry_used=retry,
            )

        # Gate failure: collect reasons and clean up for next retry.
        last_reasons = []
        for k, r in (
            ("duration", duration_result),
            ("black_frames", black_result),
            ("motion", motion_result),
        ):
            if not r.passed:
                last_reasons.append(f"{k} gate failed: {r.reason}")
        # Best-effort cleanup of the failed MP4.
        try:
            output_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    # Both attempts exhausted; raise with collected reasons from final attempt.
    raise VideoGenerationError(
        scene_index=scene.index,
        reason="; ".join(last_reasons),
        retryable=True,
    )


async def generate_video(
    story,                       # platinum.models.story.Story
    *,
    workflow_template: dict,
    comfy,
    output_root: Path,
    gates_cfg: dict,
    scene_filter: set[int] | None = None,
    width: int = 1280,
    height: int = 720,
    frame_count: int = 80,
    fps: int = 16,
) -> list[VideoReport]:
    """Run video generation for every scene that needs one.

    Validates preconditions up-front: gates_cfg has all required keys,
    every targeted scene has keyframe_path set and the file exists.
    Halts with VideoGenerationError(retryable=False) if any precondition
    fails -- no comfy calls made.

    Mutates each scene in-place: video_path, video_duration_seconds.

    Resume semantics: a scene whose video_path is already set AND points
    at an existing file is skipped. scene_filter (set of scene indexes)
    further restricts the set processed.
    """
    # Precondition 1: gates_cfg has all required keys.
    missing_keys = [k for k in REQUIRED_GATES_KEYS if k not in gates_cfg]
    if missing_keys:
        raise VideoGenerationError(
            scene_index=-1,
            reason=f"gates_cfg missing keys: {missing_keys}",
            retryable=False,
        )

    # Precondition 2: every targeted scene has a keyframe.
    for scene in story.scenes:
        if scene_filter is not None and scene.index not in scene_filter:
            continue
        if scene.video_path is not None and Path(scene.video_path).exists():
            continue  # resume -- skip precondition for already-done scenes
        if scene.keyframe_path is None:
            raise VideoGenerationError(
                scene_index=scene.index,
                reason=f"scene {scene.index} has no keyframe_path",
                retryable=False,
            )
        if not Path(scene.keyframe_path).exists():
            raise VideoGenerationError(
                scene_index=scene.index,
                reason=f"scene {scene.index} keyframe_path missing on disk: "
                       f"{scene.keyframe_path}",
                retryable=False,
            )

    reports: list[VideoReport] = []
    for scene in story.scenes:
        if scene.video_path is not None and Path(scene.video_path).exists():
            logger.info(
                "scene %d already has video_path=%s; skipping (resume)",
                scene.index, scene.video_path,
            )
            continue
        if scene_filter is not None and scene.index not in scene_filter:
            logger.info("scene %d not in scene_filter; skipping", scene.index)
            continue
        output_path = output_root / f"scene_{scene.index:03d}_raw.mp4"
        report = await generate_video_for_scene(
            scene,
            workflow_template=workflow_template,
            comfy=comfy,
            output_path=output_path,
            gates_cfg=gates_cfg,
            width=width, height=height, frame_count=frame_count, fps=fps,
        )
        scene.video_path = report.mp4_path
        scene.video_duration_seconds = report.duration_seconds
        reports.append(report)
        logger.info(
            "scene %d video generated (retry=%d, duration=%.2fs)",
            scene.index, report.retry_used, report.duration_seconds,
        )
    return reports
