# Session 8 — Wan 2.2 I2V Phase A — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate a 720p × 5s Wan 2.2 I2V clip per Cask scene from each scene's keyframe, with three quality gates, retry-once on content failure, and halt-on-fail with per-scene resume.

**Architecture:** New `pipeline/video_generator.py` mirrors `keyframe_generator.py`. New `inject_video()` sibling to `inject()` in `utils/workflow.py`. Reuses existing `ComfyClient` Protocol (`upload_image`, `generate_image` already cover the wire). New `config/workflows/wan22_i2v.json` references `kijai/ComfyUI-WanVideoWrapper` nodes installed by `vast_setup.sh`. Two-rental verify (probe scenes 1/8/16 → full 16-scene Cask). Cross-scene last-frame chaining deferred to S8.B.

**Tech Stack:** Python 3.11+, asyncio, httpx, ComfyUI HTTP API, Wan 2.2-I2V-A14B, OpenCV (cv2 + cv2.VideoWriter for fixtures), pytest + pytest-asyncio.

**Design doc:** `docs/plans/2026-04-30-session-8-wan22-i2v-design.md` (commit `bfcd4e9`).

**Total tasks:** 18.

**Pre-flight checks before starting:**
1. `git status` — confirm clean working tree on `main` at `bfcd4e9` or later.
2. `python -m pytest -q` — confirm 506-521 baseline tests pass.
3. `ruff check src tests scripts` — clean.
4. `mypy src` — only the 2 pre-existing deferrals (config.py:15 yaml stubs; sources/registry.py:30 SourceFetcher call-arg).

**After every task:**
- `python -m pytest -q` clean.
- `ruff check src tests scripts` clean.
- `mypy src` no new errors.
- Commit with conventional-commit prefix: `feat(s8.N):` / `test(s8.N):` / `chore(s8.N):`.
- **Do NOT add Co-Authored-By trailers** — disabled globally.
- **Do NOT push to origin.** S8 lands on `main` locally; push only after live verify produces actual MP4 output (matches S7.1 rule).

---

## Task 1: `inject_video()` in workflow.py

**Files:**
- Modify: `src/platinum/utils/workflow.py` (add new function alongside existing `inject`)
- Test: `tests/unit/test_workflow.py` (add 5 tests under a new `class TestInjectVideo:` block)

**Background:** The existing `inject()` requires `positive_prompt / negative_prompt / empty_latent / sampler / save_image` roles — none of which apply to the Wan 2.2 video workflow. `inject_video()` is a sibling function with its own required roles: `image_in`, `prompt`, `seed`, `video_out`. Optional roles: `width`, `height`, `frame_count`, `fps` (each is mutated only if the role is present in `_meta.role`, same pattern S6.3 used for `model_sampling_flux`).

**Step 1: Write the failing tests**

Append to `tests/unit/test_workflow.py`:

```python
class TestInjectVideo:
    """inject_video for Wan 2.2 I2V workflows."""

    def _wf_template(self) -> dict:
        # Minimal Wan-shape workflow: 4 required-role nodes + 4 optional-role
        # nodes. inputs are the only mutable surface inject_video touches.
        return {
            "_meta": {
                "role": {
                    "image_in": "100",
                    "prompt": "101",
                    "seed": "102",
                    "video_out": "103",
                    "width": "104",
                    "height": "104",   # often the same node as width (e.g., a sampler)
                    "frame_count": "105",
                    "fps": "106",
                }
            },
            "100": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "101": {"class_type": "WanT5TextEncode", "inputs": {"text": ""}},
            "102": {"class_type": "WanSampler", "inputs": {"seed": 0}},
            "103": {"class_type": "VHS_VideoCombine", "inputs": {"filename_prefix": ""}},
            "104": {"class_type": "WanSampler", "inputs": {"width": 0, "height": 0}},
            "105": {"class_type": "WanLatentVideo", "inputs": {"length": 0}},
            "106": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 0}},
        }

    def test_returns_new_dict_does_not_mutate_input(self) -> None:
        from platinum.utils.workflow import inject_video

        wf = self._wf_template()
        out = inject_video(
            wf,
            image_in="scene_001.png",
            prompt="A dimly lit crypt",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        assert out is not wf
        # Input untouched.
        assert wf["100"]["inputs"]["image"] == ""
        assert wf["101"]["inputs"]["text"] == ""
        # Output mutated.
        assert out["100"]["inputs"]["image"] == "scene_001.png"
        assert out["101"]["inputs"]["text"] == "A dimly lit crypt"
        assert out["102"]["inputs"]["seed"] == 1000

    def test_required_roles_raise_on_missing(self) -> None:
        from platinum.utils.workflow import inject_video

        wf = self._wf_template()
        # Drop the prompt role.
        del wf["_meta"]["role"]["prompt"]
        with pytest.raises(KeyError, match="prompt"):
            inject_video(
                wf,
                image_in="x.png",
                prompt="x",
                seed=0,
                output_prefix="x",
            )

    def test_optional_dimensions_skipped_when_role_absent(self) -> None:
        from platinum.utils.workflow import inject_video

        wf = self._wf_template()
        del wf["_meta"]["role"]["width"]
        del wf["_meta"]["role"]["height"]
        out = inject_video(
            wf,
            image_in="x.png",
            prompt="x",
            seed=0,
            output_prefix="x",
            width=1280,   # silently ignored
            height=720,
        )
        # Node 104's width/height inputs unchanged because the role wasn't there.
        assert out["104"]["inputs"]["width"] == 0
        assert out["104"]["inputs"]["height"] == 0

    def test_optional_frame_count_and_fps_set_when_role_present(self) -> None:
        from platinum.utils.workflow import inject_video

        wf = self._wf_template()
        out = inject_video(
            wf,
            image_in="x.png",
            prompt="x",
            seed=42,
            output_prefix="x",
            frame_count=80,
            fps=16,
        )
        assert out["105"]["inputs"]["length"] == 80
        assert out["106"]["inputs"]["frame_rate"] == 16

    def test_output_prefix_threaded_into_video_out(self) -> None:
        from platinum.utils.workflow import inject_video

        wf = self._wf_template()
        out = inject_video(
            wf,
            image_in="x.png",
            prompt="x",
            seed=0,
            output_prefix="scene_007_raw",
        )
        assert out["103"]["inputs"]["filename_prefix"] == "scene_007_raw"
```

**Step 2: Run tests, expect failures**

Run: `python -m pytest tests/unit/test_workflow.py::TestInjectVideo -v`

Expected: 5 failures with `ImportError: cannot import name 'inject_video' from 'platinum.utils.workflow'`.

**Step 3: Write the implementation**

Append to `src/platinum/utils/workflow.py`:

```python
REQUIRED_VIDEO_ROLES = (
    "image_in",
    "prompt",
    "seed",
    "video_out",
)


def inject_video(
    workflow: dict[str, Any],
    *,
    image_in: str,
    prompt: str,
    seed: int,
    output_prefix: str,
    width: int | None = None,
    height: int | None = None,
    frame_count: int | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    """Return a new workflow dict with Wan 2.2 I2V variable fields swapped in.

    Required _meta.role entries: image_in, prompt, seed, video_out.

    Optional _meta.role entries (each mutated only if present in roles):
      width, height -- typically a WanSampler or KSampler node's inputs.
      frame_count   -- WanLatentVideo node's `length` input (frames per clip).
      fps           -- VHS_VideoCombine node's `frame_rate` input.

    image_in is the server-side filename returned by ComfyClient.upload_image
    (NOT a local path); the LoadImage node references files by name relative
    to ComfyUI's `input/` directory.
    """
    out = copy.deepcopy(workflow)
    image_id = _resolve_role(out, "image_in")
    prompt_id = _resolve_role(out, "prompt")
    seed_id = _resolve_role(out, "seed")
    video_id = _resolve_role(out, "video_out")
    out[image_id]["inputs"]["image"] = image_in
    out[prompt_id]["inputs"]["text"] = prompt
    out[seed_id]["inputs"]["seed"] = seed
    out[video_id]["inputs"]["filename_prefix"] = output_prefix

    roles = out.get("_meta", {}).get("role", {})
    if width is not None and "width" in roles:
        out[roles["width"]]["inputs"]["width"] = width
    if height is not None and "height" in roles:
        out[roles["height"]]["inputs"]["height"] = height
    if frame_count is not None and "frame_count" in roles:
        out[roles["frame_count"]]["inputs"]["length"] = frame_count
    if fps is not None and "fps" in roles:
        out[roles["fps"]]["inputs"]["frame_rate"] = fps

    return out
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_workflow.py -v`
Expected: all tests pass (existing inject tests + 5 new TestInjectVideo tests).

`ruff check src tests scripts` — clean.
`mypy src` — no new errors.

**Step 5: Commit**

```bash
git add src/platinum/utils/workflow.py tests/unit/test_workflow.py
git commit -m "feat(s8.1): inject_video for Wan 2.2 I2V workflows"
```

---

## Task 2: `VideoReport` + `VideoGenerationError`

**Files:**
- Create: `src/platinum/pipeline/video_generator.py` (skeleton — just the dataclass and exception for now)
- Test: `tests/unit/test_video_generator.py` (new file)

**Step 1: Write the failing tests**

Create `tests/unit/test_video_generator.py`:

```python
"""Unit tests for video_generator pipeline (S8 Phase A)."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestVideoReport:
    def test_dataclass_fields(self) -> None:
        from platinum.pipeline.video_generator import VideoReport

        report = VideoReport(
            scene_index=7,
            success=True,
            mp4_path=Path("data/stories/x/clips/scene_007_raw.mp4"),
            duration_seconds=5.0,
            gates_passed={"duration": True, "black_frames": True, "motion": True},
            retry_used=0,
        )
        assert report.scene_index == 7
        assert report.success is True
        assert report.duration_seconds == 5.0
        assert report.retry_used == 0
        assert report.gates_passed["motion"] is True

    def test_frozen_immutable(self) -> None:
        from platinum.pipeline.video_generator import VideoReport
        from dataclasses import FrozenInstanceError

        report = VideoReport(
            scene_index=0,
            success=False,
            mp4_path=None,
            duration_seconds=0.0,
            gates_passed={},
            retry_used=1,
        )
        with pytest.raises(FrozenInstanceError):
            report.success = True  # type: ignore[misc]


class TestVideoGenerationError:
    def test_carries_scene_index_and_reason(self) -> None:
        from platinum.pipeline.video_generator import VideoGenerationError

        exc = VideoGenerationError(
            scene_index=3,
            reason="motion gate failed: flow=0.05 < min=0.30",
            retryable=True,
        )
        assert exc.scene_index == 3
        assert exc.reason == "motion gate failed: flow=0.05 < min=0.30"
        assert exc.retryable is True
        assert "scene_index=3" in str(exc)
        assert "motion gate failed" in str(exc)

    def test_retryable_default_false(self) -> None:
        from platinum.pipeline.video_generator import VideoGenerationError

        exc = VideoGenerationError(scene_index=0, reason="comfy http 500")
        assert exc.retryable is False
```

**Step 2: Run tests, expect ImportError**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 4 failures with `ModuleNotFoundError: No module named 'platinum.pipeline.video_generator'`.

**Step 3: Write the implementation**

Create `src/platinum/pipeline/video_generator.py`:

```python
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
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 4 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.2): VideoReport + VideoGenerationError"
```

---

## Task 3: `_seed_for_scene` helper

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py` (add `class TestSeedForScene:`)

**Step 1: Write the failing tests**

Append to `tests/unit/test_video_generator.py`:

```python
class TestSeedForScene:
    def test_base_seed_is_index_times_thousand(self) -> None:
        from platinum.pipeline.video_generator import _seed_for_scene

        assert _seed_for_scene(0, retry=0) == 0
        assert _seed_for_scene(1, retry=0) == 1000
        assert _seed_for_scene(7, retry=0) == 7000
        assert _seed_for_scene(15, retry=0) == 15000

    def test_retry_increments_seed_by_one(self) -> None:
        from platinum.pipeline.video_generator import _seed_for_scene

        assert _seed_for_scene(7, retry=1) == 7001

    def test_disjoint_from_keyframe_seeds(self) -> None:
        """keyframe_generator uses scene*1000 + candidate_idx (0,1,2 typically).

        Video uses scene*1000 + retry (0,1). Both fit in same 1000-block but
        retry=0/1 collide with candidate=0/1 only if user runs both with the
        same scene -- which is fine because keyframes write to PNG and
        video to MP4. The test just confirms the formula stays simple.
        """
        from platinum.pipeline.video_generator import _seed_for_scene

        assert _seed_for_scene(7, retry=0) == 7000
        assert _seed_for_scene(7, retry=1) == 7001
```

**Step 2: Run tests, expect failure**

Run: `python -m pytest tests/unit/test_video_generator.py::TestSeedForScene -v`
Expected: 3 failures with `ImportError: cannot import name '_seed_for_scene'`.

**Step 3: Write the implementation**

Append to `src/platinum/pipeline/video_generator.py` (just below the dataclass/exception block):

```python
def _seed_for_scene(scene_index: int, *, retry: int = 0) -> int:
    """Deterministic seed for a scene's video generation.

    seed = scene_index * 1000 + retry. With retry in {0, 1} only two seeds
    are ever used per scene (initial + one retry).
    """
    return scene_index * 1000 + retry
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 7 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.3): _seed_for_scene deterministic helper"
```

---

## Task 4: `generate_video_for_scene` happy path

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py`

**Background:** This task wires the function up end-to-end against a `FakeComfyClient` returning a happy-path MP4 fixture. No gates yet — we add them in Task 5. The Fake's `upload_image` returns a deterministic name; `generate_image` copies the configured fixture MP4 to `output_path`.

**Step 1: Write the failing test**

Append to `tests/unit/test_video_generator.py`:

```python
class TestGenerateVideoForSceneHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_writes_mp4_and_returns_report(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import (
            VideoReport,
            generate_video_for_scene,
        )
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        # Pre-bake a 5s synthetic MP4 with motion as the Fake's response.
        fixture_mp4 = tmp_path / "wan_output.mp4"
        make_test_video_with_motion(fixture_mp4, n_frames=80, fps=16, size=(64, 64))

        # Workflow template with all required + optional video roles.
        workflow_template = {
            "_meta": {"role": {
                "image_in": "100", "prompt": "101", "seed": "102",
                "video_out": "103", "width": "104", "height": "104",
                "frame_count": "105", "fps": "106",
            }},
            "100": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "101": {"class_type": "WanT5TextEncode", "inputs": {"text": ""}},
            "102": {"class_type": "WanSampler",
                    "inputs": {"seed": 0, "width": 0, "height": 0}},
            "103": {"class_type": "VHS_VideoCombine",
                    "inputs": {"filename_prefix": "", "frame_rate": 0}},
            "104": {"class_type": "WanSampler",
                    "inputs": {"width": 0, "height": 0}},
            "105": {"class_type": "WanLatentVideo", "inputs": {"length": 0}},
            "106": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 0}},
        }

        # Pre-compute the signature the Fake will see for retry=0.
        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")  # FakeComfyClient.upload_image
                                           # only reads the path .name.
        wf_for_signature = inject_video(
            workflow_template,
            image_in="scene_001.png",  # what FakeComfyClient.upload_image returns
            prompt="a dimly lit crypt",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        sig = workflow_signature(wf_for_signature)
        comfy = FakeComfyClient(responses={sig: [fixture_mp4]})

        # Mock Scene with the minimum surface video_generator needs.
        from types import SimpleNamespace
        scene = SimpleNamespace(
            index=1,
            visual_prompt="a dimly lit crypt",
            keyframe_path=keyframe,
            video_path=None,
        )

        report = await generate_video_for_scene(
            scene,
            workflow_template=workflow_template,
            comfy=comfy,
            output_path=tmp_path / "clips" / "scene_001_raw.mp4",
            gates_cfg={
                "duration_target_seconds": 5.0,
                "duration_tolerance_seconds": 0.2,
                "black_frame_max_ratio": 0.05,
                "motion_min_flow": 0.0,   # gates not yet implemented
            },
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )

        assert isinstance(report, VideoReport)
        assert report.scene_index == 1
        assert report.success is True
        assert report.retry_used == 0
        assert report.mp4_path is not None
        assert report.mp4_path.exists()
        assert (tmp_path / "clips" / "scene_001_raw.mp4").exists()
```

**Step 2: Run test, expect failure**

Run: `python -m pytest tests/unit/test_video_generator.py::TestGenerateVideoForSceneHappyPath -v`
Expected: failure with `ImportError: cannot import name 'generate_video_for_scene'`.

**Step 3: Write the implementation**

Append to `src/platinum/pipeline/video_generator.py`:

```python
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

    Happy path only -- gates and retry added in subsequent tasks.
    """
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

    # 2. Build the workflow.
    seed = _seed_for_scene(scene.index, retry=0)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    await comfy.generate_image(workflow=workflow, output_path=output_path)

    return VideoReport(
        scene_index=scene.index,
        success=True,
        mp4_path=output_path,
        duration_seconds=float(frame_count) / float(fps),
        gates_passed={"duration": True, "black_frames": True, "motion": True},
        retry_used=0,
    )
```

**Step 4: Run test, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 8 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.4): generate_video_for_scene happy path (no gates yet)"
```

---

## Task 5: Add 3 quality gates (duration / black_frames / motion)

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py`

**Background:** Wire the existing `validate.py` primitives into `generate_video_for_scene`. Gates run in this order: duration → black_frames → motion. If any fail and `retry_used == 0`, we'll add retry in Task 6 (this task only adds the *check* — failure path is `VideoGenerationError(retryable=True)` immediately, no retry yet).

**Step 1: Write the failing tests**

Append to `tests/unit/test_video_generator.py`:

```python
class TestGenerateVideoForSceneGates:
    @pytest.mark.asyncio
    async def test_black_frame_gate_fails_when_video_is_solid_black(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import (
            VideoGenerationError,
            generate_video_for_scene,
        )
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video

        black_mp4 = tmp_path / "black.mp4"
        make_test_video(black_mp4, n_frames=80, fps=16, color=(0, 0, 0), size=(64, 64))

        workflow_template = _wan_template_for_tests()
        wf_for_sig = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(responses={workflow_signature(wf_for_sig): [black_mp4]})

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace
        scene = SimpleNamespace(index=1, visual_prompt="x", keyframe_path=keyframe,
                                video_path=None)

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video_for_scene(
                scene,
                workflow_template=workflow_template,
                comfy=comfy,
                output_path=tmp_path / "clips" / "scene_001_raw.mp4",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 0.05,
                    "motion_min_flow": 0.0,
                },
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is True
        assert "black_frames" in excinfo.value.reason

    @pytest.mark.asyncio
    async def test_motion_gate_fails_on_static_video(self, tmp_path: Path) -> None:
        from platinum.pipeline.video_generator import (
            VideoGenerationError,
            generate_video_for_scene,
        )
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video

        # Static gray (no motion).
        static_mp4 = tmp_path / "static.mp4"
        make_test_video(static_mp4, n_frames=80, fps=16, color=(120, 120, 120), size=(64, 64))

        workflow_template = _wan_template_for_tests()
        wf_for_sig = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(responses={workflow_signature(wf_for_sig): [static_mp4]})

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace
        scene = SimpleNamespace(index=1, visual_prompt="x", keyframe_path=keyframe,
                                video_path=None)

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video_for_scene(
                scene,
                workflow_template=workflow_template,
                comfy=comfy,
                output_path=tmp_path / "clips" / "scene_001_raw.mp4",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 1.01,   # disable black gate
                    "motion_min_flow": 0.5,           # require flow >= 0.5
                },
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is True
        assert "motion" in excinfo.value.reason

    @pytest.mark.asyncio
    async def test_duration_gate_fails_when_clip_too_short(self, tmp_path: Path) -> None:
        from platinum.pipeline.video_generator import (
            VideoGenerationError,
            generate_video_for_scene,
        )
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        # 2-second clip with motion -- duration gate at 5.0 ± 0.2 fails.
        short_mp4 = tmp_path / "short.mp4"
        make_test_video_with_motion(short_mp4, n_frames=32, fps=16, size=(64, 64))  # 2.0s

        workflow_template = _wan_template_for_tests()
        wf_for_sig = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(responses={workflow_signature(wf_for_sig): [short_mp4]})

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace
        scene = SimpleNamespace(index=1, visual_prompt="x", keyframe_path=keyframe,
                                video_path=None)

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video_for_scene(
                scene,
                workflow_template=workflow_template,
                comfy=comfy,
                output_path=tmp_path / "clips" / "scene_001_raw.mp4",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 1.01,
                    "motion_min_flow": 0.0,
                },
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is True
        assert "duration" in excinfo.value.reason


def _wan_template_for_tests() -> dict:
    """Shared minimal Wan workflow template for the test module."""
    return {
        "_meta": {"role": {
            "image_in": "100", "prompt": "101", "seed": "102",
            "video_out": "103", "width": "104", "height": "104",
            "frame_count": "105", "fps": "106",
        }},
        "100": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "101": {"class_type": "WanT5TextEncode", "inputs": {"text": ""}},
        "102": {"class_type": "WanSampler", "inputs": {"seed": 0}},
        "103": {"class_type": "VHS_VideoCombine",
                "inputs": {"filename_prefix": "", "frame_rate": 0}},
        "104": {"class_type": "WanSampler", "inputs": {"width": 0, "height": 0}},
        "105": {"class_type": "WanLatentVideo", "inputs": {"length": 0}},
        "106": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 0}},
    }
```

Add `import pytest` and the existing imports at the top of the test file if not already present. Make sure pytest_asyncio is wired (the project uses `pytest.mark.asyncio` -- check `pyproject.toml`'s `[tool.pytest.ini_options]` for `asyncio_mode = "auto"` or per-test marks).

**Step 2: Run tests, expect failure**

Run: `python -m pytest tests/unit/test_video_generator.py::TestGenerateVideoForSceneGates -v`
Expected: 3 failures because the impl doesn't run gates yet.

**Step 3: Update implementation**

Replace the body of `generate_video_for_scene` after the `await comfy.generate_image(...)` call with gate-aware logic:

```python
async def generate_video_for_scene(
    scene,
    *,
    workflow_template: dict,
    comfy,
    output_path: Path,
    gates_cfg: dict,
    width: int = 1280,
    height: int = 720,
    frame_count: int = 80,
    fps: int = 16,
) -> VideoReport:
    from platinum.utils.workflow import inject_video
    from platinum.utils.validate import (
        check_duration_match,
        check_black_frames,
        check_motion,
    )

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

    server_filename = await comfy.upload_image(Path(scene.keyframe_path))

    seed = _seed_for_scene(scene.index, retry=0)
    workflow = inject_video(
        workflow_template,
        image_in=server_filename,
        prompt=scene.visual_prompt or "",
        seed=seed,
        output_prefix=f"scene_{scene.index:03d}_raw",
        width=width, height=height, frame_count=frame_count, fps=fps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    await comfy.generate_image(workflow=workflow, output_path=output_path)

    target_s = float(gates_cfg["duration_target_seconds"])
    tol_s = float(gates_cfg["duration_tolerance_seconds"])
    duration_result = check_duration_match(
        output_path, expected_seconds=target_s, tolerance_seconds=tol_s
    )
    black_result = check_black_frames(
        output_path,
        max_black_ratio=float(gates_cfg["black_frame_max_ratio"]),
    )
    motion_result = check_motion(
        output_path,
        min_mean_flow=float(gates_cfg["motion_min_flow"]),
    )
    gates_passed = {
        "duration": duration_result.passed,
        "black_frames": black_result.passed,
        "motion": motion_result.passed,
    }
    if not all(gates_passed.values()):
        failed = [k for k, v in gates_passed.items() if not v]
        reasons = []
        for k, r in (("duration", duration_result), ("black_frames", black_result),
                     ("motion", motion_result)):
            if not r.passed:
                reasons.append(f"{k} gate failed: {r.reason}")
        # Best-effort cleanup of the failed MP4.
        try:
            output_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        raise VideoGenerationError(
            scene_index=scene.index,
            reason="; ".join(reasons),
            retryable=True,
        )

    return VideoReport(
        scene_index=scene.index,
        success=True,
        mp4_path=output_path,
        duration_seconds=float(duration_result.metric),
        gates_passed=gates_passed,
        retry_used=0,
    )
```

**Important:** Verify `check_duration_match`, `check_black_frames`, and `check_motion` parameter names match the existing primitives in `src/platinum/utils/validate.py`. Adjust kwarg names if they differ (the validate.py reads above show `expected_seconds`/`tolerance_seconds` style — confirm before submitting). If the actual signatures use different kwargs (e.g., `target_duration_s` vs `expected_seconds`), update both this code and the test thresholds to match.

Check exact signatures with:
```bash
python -c "from platinum.utils.validate import check_duration_match, check_black_frames, check_motion; help(check_duration_match); help(check_black_frames); help(check_motion)"
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 11 passing tests (8 prior + 3 new).

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.5): wire 3 quality gates (duration/black/motion) into generate_video_for_scene"
```

---

## Task 6: Add retry-once on content failure

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py`

**Background:** Currently a content-failure raises immediately. This task makes it retry once with `seed = _seed_for_scene(scene.index, retry=1)` before raising.

**Step 1: Write the failing test**

Append to `tests/unit/test_video_generator.py`:

```python
class TestGenerateVideoForSceneRetry:
    @pytest.mark.asyncio
    async def test_retry_once_on_first_content_fail(self, tmp_path: Path) -> None:
        """Fake returns black MP4 on first call (gate fails), motion MP4 on
        second (gate passes). Function should succeed with retry_used=1.
        """
        from platinum.pipeline.video_generator import generate_video_for_scene
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video, make_test_video_with_motion

        black_mp4 = tmp_path / "black.mp4"
        motion_mp4 = tmp_path / "motion.mp4"
        make_test_video(black_mp4, n_frames=80, fps=16, color=(0, 0, 0), size=(64, 64))
        make_test_video_with_motion(motion_mp4, n_frames=80, fps=16, size=(64, 64))

        workflow_template = _wan_template_for_tests()
        # Different seeds -> different signatures -> two distinct response slots.
        wf_seed_0 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x", seed=1001, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(
            responses={
                workflow_signature(wf_seed_0): [black_mp4],
                workflow_signature(wf_seed_1): [motion_mp4],
            }
        )

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace
        scene = SimpleNamespace(index=1, visual_prompt="x", keyframe_path=keyframe,
                                video_path=None)

        report = await generate_video_for_scene(
            scene,
            workflow_template=workflow_template,
            comfy=comfy,
            output_path=tmp_path / "clips" / "scene_001_raw.mp4",
            gates_cfg={
                "duration_target_seconds": 5.0,
                "duration_tolerance_seconds": 0.2,
                "black_frame_max_ratio": 0.05,
                "motion_min_flow": 0.0,
            },
            width=1280, height=720, frame_count=80, fps=16,
        )

        assert report.success is True
        assert report.retry_used == 1
        assert len(comfy.calls) == 2  # initial + 1 retry
```

**Step 2: Run test, expect failure**

Run: `python -m pytest tests/unit/test_video_generator.py::TestGenerateVideoForSceneRetry -v`
Expected: 1 failure — current impl raises on first content fail without retrying.

**Step 3: Update implementation**

Refactor the gate-fail path to loop over `retry in (0, 1)`. Replace the gate-and-raise block with a loop:

```python
async def generate_video_for_scene(
    scene,
    *,
    workflow_template: dict,
    comfy,
    output_path: Path,
    gates_cfg: dict,
    width: int = 1280,
    height: int = 720,
    frame_count: int = 80,
    fps: int = 16,
) -> VideoReport:
    from platinum.utils.workflow import inject_video
    from platinum.utils.validate import (
        check_duration_match, check_black_frames, check_motion,
    )

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

    server_filename = await comfy.upload_image(Path(scene.keyframe_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_reasons: list[str] = []
    last_gates: dict[str, bool] = {}
    last_duration: float = 0.0
    for retry in (0, 1):
        seed = _seed_for_scene(scene.index, retry=retry)
        workflow = inject_video(
            workflow_template,
            image_in=server_filename,
            prompt=scene.visual_prompt or "",
            seed=seed,
            output_prefix=f"scene_{scene.index:03d}_raw",
            width=width, height=height, frame_count=frame_count, fps=fps,
        )
        await comfy.generate_image(workflow=workflow, output_path=output_path)

        target_s = float(gates_cfg["duration_target_seconds"])
        tol_s = float(gates_cfg["duration_tolerance_seconds"])
        duration_result = check_duration_match(
            output_path, expected_seconds=target_s, tolerance_seconds=tol_s
        )
        black_result = check_black_frames(
            output_path,
            max_black_ratio=float(gates_cfg["black_frame_max_ratio"]),
        )
        motion_result = check_motion(
            output_path,
            min_mean_flow=float(gates_cfg["motion_min_flow"]),
        )
        gates_passed = {
            "duration": duration_result.passed,
            "black_frames": black_result.passed,
            "motion": motion_result.passed,
        }
        last_gates = gates_passed
        last_duration = float(duration_result.metric)
        if all(gates_passed.values()):
            return VideoReport(
                scene_index=scene.index,
                success=True,
                mp4_path=output_path,
                duration_seconds=last_duration,
                gates_passed=gates_passed,
                retry_used=retry,
            )
        # Record reasons for potential final raise; clean up failed MP4.
        last_reasons = []
        for k, r in (("duration", duration_result), ("black_frames", black_result),
                     ("motion", motion_result)):
            if not r.passed:
                last_reasons.append(f"{k} gate failed: {r.reason}")
        try:
            output_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    raise VideoGenerationError(
        scene_index=scene.index,
        reason="; ".join(last_reasons),
        retryable=True,
    )
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 12 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.6): retry-once on content failure with disjoint seed"
```

---

## Task 7: Halt-on-second-content-fail and infrastructure failures

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py`

**Background:** The retry logic in Task 6 already raises on second-fail (the `raise` at the end of the loop). This task adds explicit test coverage for that path, plus tests that infrastructure failures (FakeComfyClient raising `ComfyError` / generic exception) are NOT retried — they propagate immediately as `retryable=False`.

**Step 1: Write the failing tests**

Append to `tests/unit/test_video_generator.py`:

```python
class TestGenerateVideoForSceneHalt:
    @pytest.mark.asyncio
    async def test_halt_on_second_content_fail(self, tmp_path: Path) -> None:
        """Both initial and retry produce black MP4. Should raise after 2 attempts."""
        from platinum.pipeline.video_generator import (
            VideoGenerationError, generate_video_for_scene,
        )
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video

        black_mp4 = tmp_path / "black.mp4"
        make_test_video(black_mp4, n_frames=80, fps=16, color=(0, 0, 0), size=(64, 64))

        workflow_template = _wan_template_for_tests()
        # Both seeds map to the same black fixture.
        wf_seed_0 = inject_video(
            workflow_template, image_in="scene_001.png",
            prompt="x", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template, image_in="scene_001.png",
            prompt="x", seed=1001, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(
            responses={
                workflow_signature(wf_seed_0): [black_mp4],
                workflow_signature(wf_seed_1): [black_mp4],
            }
        )

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace
        scene = SimpleNamespace(index=1, visual_prompt="x", keyframe_path=keyframe,
                                video_path=None)

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video_for_scene(
                scene,
                workflow_template=workflow_template,
                comfy=comfy,
                output_path=tmp_path / "clips" / "scene_001_raw.mp4",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 0.05,
                    "motion_min_flow": 0.0,
                },
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is True
        assert len(comfy.calls) == 2
        # Failed MP4 cleaned up.
        assert not (tmp_path / "clips" / "scene_001_raw.mp4").exists()

    @pytest.mark.asyncio
    async def test_infra_failure_not_retried(self, tmp_path: Path) -> None:
        """generate_image raises -> propagate as retryable=False, no retry."""
        from platinum.pipeline.video_generator import (
            VideoGenerationError, generate_video_for_scene,
        )

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace
        scene = SimpleNamespace(index=1, visual_prompt="x", keyframe_path=keyframe,
                                video_path=None)

        # Hand-rolled fake that raises on generate_image.
        class _BrokenComfy:
            calls: list = []
            async def upload_image(self, path):
                return path.name
            async def generate_image(self, *, workflow, output_path):
                self.calls.append(workflow)
                raise RuntimeError("ComfyUI returned 500")

        comfy = _BrokenComfy()

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video_for_scene(
                scene,
                workflow_template=_wan_template_for_tests(),
                comfy=comfy,
                output_path=tmp_path / "clips" / "scene_001_raw.mp4",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 0.05,
                    "motion_min_flow": 0.0,
                },
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is False
        assert "ComfyUI returned 500" in excinfo.value.reason
        # Only one attempt, no retry.
        assert len(comfy.calls) == 1
```

**Step 2: Run tests**

Run: `python -m pytest tests/unit/test_video_generator.py::TestGenerateVideoForSceneHalt -v`

Expected: 1 passing (the second-content-fail test, since the impl already handles it via the loop's final `raise`); 1 failing (the infra-failure test — current impl lets `RuntimeError` propagate raw rather than wrapping as `VideoGenerationError(retryable=False)`).

**Step 3: Update implementation**

Wrap the `await comfy.generate_image(...)` call inside the loop with a try/except that converts non-VideoGenerationError exceptions to `VideoGenerationError(retryable=False)` and breaks out of the retry loop:

Replace this part of the loop body:

```python
        await comfy.generate_image(workflow=workflow, output_path=output_path)
```

With:

```python
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
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 14 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.7): infra-failure path raises retryable=False, no retry"
```

---

## Task 8: `generate_video` whole-story iteration + resume + scene_filter

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py`

**Background:** The whole-story driver iterates all scenes, skips scenes whose `video_path` already points at an existing file (resume), respects `scene_filter` for `--scenes` CLI subset support, and mutates each scene's `video_path` + `video_duration_seconds` after success.

**Step 1: Write the failing tests**

Append to `tests/unit/test_video_generator.py`:

```python
class TestGenerateVideo:
    @pytest.mark.asyncio
    async def test_iterates_all_scenes_setting_video_path(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import generate_video
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        # 3 scenes -> 3 distinct workflow signatures (different seeds).
        motion_mp4 = tmp_path / "motion.mp4"
        make_test_video_with_motion(motion_mp4, n_frames=80, fps=16, size=(64, 64))

        workflow_template = _wan_template_for_tests()
        responses: dict[str, list[Path]] = {}
        for scene_index in (0, 1, 2):
            wf = inject_video(
                workflow_template, image_in=f"scene_{scene_index:03d}.png",
                prompt=f"prompt {scene_index}",
                seed=scene_index * 1000,
                output_prefix=f"scene_{scene_index:03d}_raw",
                width=1280, height=720, frame_count=80, fps=16,
            )
            responses[workflow_signature(wf)] = [motion_mp4]
        comfy = FakeComfyClient(responses=responses)

        # Build 3 fake scenes.
        from types import SimpleNamespace
        scenes = []
        for i in (0, 1, 2):
            kf = tmp_path / f"scene_{i:03d}.png"
            kf.write_bytes(b"fake_png")
            scenes.append(SimpleNamespace(
                index=i, visual_prompt=f"prompt {i}",
                keyframe_path=kf, video_path=None,
                video_duration_seconds=0.0,
            ))
        story = SimpleNamespace(scenes=scenes)

        reports = await generate_video(
            story,
            workflow_template=workflow_template,
            comfy=comfy,
            output_root=tmp_path / "clips",
            gates_cfg={
                "duration_target_seconds": 5.0,
                "duration_tolerance_seconds": 0.2,
                "black_frame_max_ratio": 0.05,
                "motion_min_flow": 0.0,
            },
            width=1280, height=720, frame_count=80, fps=16,
        )

        assert len(reports) == 3
        for i, scene in enumerate(scenes):
            assert scene.video_path is not None
            assert Path(scene.video_path).exists()
            assert scene.video_duration_seconds > 0.0

    @pytest.mark.asyncio
    async def test_skips_scenes_with_existing_video_path(self, tmp_path: Path) -> None:
        """Resume semantics: scene with video_path set + file exists is skipped."""
        from platinum.pipeline.video_generator import generate_video
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        motion_mp4 = tmp_path / "motion.mp4"
        make_test_video_with_motion(motion_mp4, n_frames=80, fps=16, size=(64, 64))

        workflow_template = _wan_template_for_tests()
        wf = inject_video(
            workflow_template, image_in="scene_001.png",
            prompt="prompt 1", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(responses={workflow_signature(wf): [motion_mp4]})

        # Scene 0 already has a video file -> should be skipped.
        existing_video = tmp_path / "clips" / "scene_000_raw.mp4"
        existing_video.parent.mkdir(parents=True, exist_ok=True)
        existing_video.write_bytes(b"existing")

        from types import SimpleNamespace
        kf0 = tmp_path / "scene_000.png"
        kf0.write_bytes(b"fake_png")
        kf1 = tmp_path / "scene_001.png"
        kf1.write_bytes(b"fake_png")
        scenes = [
            SimpleNamespace(
                index=0, visual_prompt="prompt 0",
                keyframe_path=kf0, video_path=existing_video,
                video_duration_seconds=5.0,
            ),
            SimpleNamespace(
                index=1, visual_prompt="prompt 1",
                keyframe_path=kf1, video_path=None,
                video_duration_seconds=0.0,
            ),
        ]
        story = SimpleNamespace(scenes=scenes)

        reports = await generate_video(
            story,
            workflow_template=workflow_template, comfy=comfy,
            output_root=tmp_path / "clips",
            gates_cfg={
                "duration_target_seconds": 5.0,
                "duration_tolerance_seconds": 0.2,
                "black_frame_max_ratio": 0.05,
                "motion_min_flow": 0.0,
            },
            width=1280, height=720, frame_count=80, fps=16,
        )

        # Only one scene was processed (scene 1).
        assert len(reports) == 1
        assert reports[0].scene_index == 1
        assert len(comfy.calls) == 1

    @pytest.mark.asyncio
    async def test_scene_filter_only_processes_listed_scenes(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import generate_video
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        motion_mp4 = tmp_path / "motion.mp4"
        make_test_video_with_motion(motion_mp4, n_frames=80, fps=16, size=(64, 64))

        workflow_template = _wan_template_for_tests()
        responses: dict[str, list[Path]] = {}
        for scene_index in (0, 1, 2):
            wf = inject_video(
                workflow_template, image_in=f"scene_{scene_index:03d}.png",
                prompt=f"prompt {scene_index}", seed=scene_index * 1000,
                output_prefix=f"scene_{scene_index:03d}_raw",
                width=1280, height=720, frame_count=80, fps=16,
            )
            responses[workflow_signature(wf)] = [motion_mp4]
        comfy = FakeComfyClient(responses=responses)

        from types import SimpleNamespace
        scenes = []
        for i in (0, 1, 2):
            kf = tmp_path / f"scene_{i:03d}.png"
            kf.write_bytes(b"fake_png")
            scenes.append(SimpleNamespace(
                index=i, visual_prompt=f"prompt {i}",
                keyframe_path=kf, video_path=None,
                video_duration_seconds=0.0,
            ))
        story = SimpleNamespace(scenes=scenes)

        reports = await generate_video(
            story,
            workflow_template=workflow_template, comfy=comfy,
            output_root=tmp_path / "clips",
            gates_cfg={
                "duration_target_seconds": 5.0,
                "duration_tolerance_seconds": 0.2,
                "black_frame_max_ratio": 0.05,
                "motion_min_flow": 0.0,
            },
            scene_filter={1},  # only process scene index 1
            width=1280, height=720, frame_count=80, fps=16,
        )

        assert len(reports) == 1
        assert reports[0].scene_index == 1
        assert scenes[0].video_path is None  # untouched
        assert scenes[2].video_path is None  # untouched
        assert scenes[1].video_path is not None
```

**Step 2: Run tests, expect failures**

Run: `python -m pytest tests/unit/test_video_generator.py::TestGenerateVideo -v`
Expected: 3 failures with `ImportError: cannot import name 'generate_video'`.

**Step 3: Write the implementation**

Append to `src/platinum/pipeline/video_generator.py`:

```python
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

    Mutates each scene in-place: video_path, video_duration_seconds.

    Resume semantics: a scene whose video_path is already set AND points
    at an existing file is skipped. scene_filter (set of scene indexes)
    further restricts the set processed.
    """
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
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 17 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.8): generate_video iterates story with resume + scene_filter"
```

---

## Task 9: Precondition validation (halt before any GPU work)

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/unit/test_video_generator.py`

**Background:** Before opening any ComfyUI session, validate: every targeted scene has `keyframe_path` set + file exists; `gates_cfg` has all four required keys. If anything fails, raise `VideoGenerationError(retryable=False)` immediately.

**Step 1: Write the failing tests**

Append to `tests/unit/test_video_generator.py`:

```python
class TestGenerateVideoPreconditions:
    @pytest.mark.asyncio
    async def test_missing_keyframe_path_halts_before_any_comfy_call(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import (
            VideoGenerationError, generate_video,
        )
        from platinum.utils.comfyui import FakeComfyClient

        comfy = FakeComfyClient(responses={})
        from types import SimpleNamespace
        scenes = [
            SimpleNamespace(
                index=0, visual_prompt="x",
                keyframe_path=None,  # missing
                video_path=None, video_duration_seconds=0.0,
            ),
        ]
        story = SimpleNamespace(scenes=scenes)

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video(
                story,
                workflow_template=_wan_template_for_tests(), comfy=comfy,
                output_root=tmp_path / "clips",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 0.05,
                    "motion_min_flow": 0.0,
                },
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is False
        assert "keyframe_path" in excinfo.value.reason
        # No comfy calls -- precondition rejected before any work.
        assert len(comfy.calls) == 0

    @pytest.mark.asyncio
    async def test_missing_gates_cfg_key_halts(self, tmp_path: Path) -> None:
        from platinum.pipeline.video_generator import (
            VideoGenerationError, generate_video,
        )
        from platinum.utils.comfyui import FakeComfyClient

        comfy = FakeComfyClient(responses={})
        from types import SimpleNamespace
        scenes = [
            SimpleNamespace(
                index=0, visual_prompt="x",
                keyframe_path=tmp_path / "kf.png",
                video_path=None, video_duration_seconds=0.0,
            ),
        ]
        (tmp_path / "kf.png").write_bytes(b"fake_png")
        story = SimpleNamespace(scenes=scenes)

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video(
                story,
                workflow_template=_wan_template_for_tests(), comfy=comfy,
                output_root=tmp_path / "clips",
                gates_cfg={"duration_target_seconds": 5.0},  # missing other keys
                width=1280, height=720, frame_count=80, fps=16,
            )
        assert excinfo.value.retryable is False
        assert "gates_cfg" in excinfo.value.reason
        assert len(comfy.calls) == 0

    @pytest.mark.asyncio
    async def test_filtered_out_scenes_dont_trigger_precondition_failures(
        self, tmp_path: Path
    ) -> None:
        """A scene with a missing keyframe is fine if it's filtered out."""
        from platinum.pipeline.video_generator import generate_video
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        motion_mp4 = tmp_path / "motion.mp4"
        make_test_video_with_motion(motion_mp4, n_frames=80, fps=16, size=(64, 64))

        workflow_template = _wan_template_for_tests()
        wf = inject_video(
            workflow_template, image_in="scene_001.png",
            prompt="x", seed=1000, output_prefix="scene_001_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        comfy = FakeComfyClient(responses={workflow_signature(wf): [motion_mp4]})

        from types import SimpleNamespace
        kf1 = tmp_path / "scene_001.png"
        kf1.write_bytes(b"fake_png")
        scenes = [
            SimpleNamespace(
                index=0, visual_prompt="x",
                keyframe_path=None,  # missing -- but filtered out
                video_path=None, video_duration_seconds=0.0,
            ),
            SimpleNamespace(
                index=1, visual_prompt="x",
                keyframe_path=kf1, video_path=None,
                video_duration_seconds=0.0,
            ),
        ]
        story = SimpleNamespace(scenes=scenes)

        reports = await generate_video(
            story,
            workflow_template=workflow_template, comfy=comfy,
            output_root=tmp_path / "clips",
            gates_cfg={
                "duration_target_seconds": 5.0,
                "duration_tolerance_seconds": 0.2,
                "black_frame_max_ratio": 0.05,
                "motion_min_flow": 0.0,
            },
            scene_filter={1},
            width=1280, height=720, frame_count=80, fps=16,
        )

        assert len(reports) == 1
        assert reports[0].scene_index == 1
```

**Step 2: Run tests**

Run: `python -m pytest tests/unit/test_video_generator.py::TestGenerateVideoPreconditions -v`
Expected: 2 failures (missing keyframe + missing gates key), 1 passing (filtered-out scene).

**Step 3: Update implementation**

Add a precondition block at the top of `generate_video`:

```python
REQUIRED_GATES_KEYS = (
    "duration_target_seconds",
    "duration_tolerance_seconds",
    "black_frame_max_ratio",
    "motion_min_flow",
)


async def generate_video(
    story,
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

    # ... existing iteration loop ...
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_video_generator.py -v`
Expected: 20 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/unit/test_video_generator.py
git commit -m "feat(s8.9): precondition validation halts before any comfy call"
```

---

## Task 10: `VideoGeneratorStage` skeleton + ctx-based DI

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/integration/test_video_generator_stage.py` (new file)

**Background:** The Stage shell pulls dependencies from `ctx.config.settings`, computes `output_root` from `ctx.story_path(story).parent / "clips"`, mutates the story and saves atomically. Mirrors `KeyframeGeneratorStage` exactly.

**Step 1: Write the failing test**

Create `tests/integration/test_video_generator_stage.py`:

```python
"""Integration tests for VideoGeneratorStage (S8 Phase A)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _wan_template() -> dict:
    return {
        "_meta": {"role": {
            "image_in": "100", "prompt": "101", "seed": "102",
            "video_out": "103", "width": "104", "height": "104",
            "frame_count": "105", "fps": "106",
        }},
        "100": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "101": {"class_type": "WanT5TextEncode", "inputs": {"text": ""}},
        "102": {"class_type": "WanSampler", "inputs": {"seed": 0}},
        "103": {"class_type": "VHS_VideoCombine",
                "inputs": {"filename_prefix": "", "frame_rate": 0}},
        "104": {"class_type": "WanSampler", "inputs": {"width": 0, "height": 0}},
        "105": {"class_type": "WanLatentVideo", "inputs": {"length": 0}},
        "106": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 0}},
    }


class TestVideoGeneratorStage:
    @pytest.mark.asyncio
    async def test_stage_runs_end_to_end_with_injected_comfy(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import VideoGeneratorStage
        from platinum.utils.comfyui import FakeComfyClient, workflow_signature
        from platinum.utils.workflow import inject_video

        from tests._fixtures import make_test_video_with_motion

        motion_mp4 = tmp_path / "motion.mp4"
        make_test_video_with_motion(motion_mp4, n_frames=80, fps=16, size=(64, 64))

        workflow_template = _wan_template()
        responses: dict[str, list[Path]] = {}
        for scene_index in (0, 1):
            wf = inject_video(
                workflow_template, image_in=f"scene_{scene_index:03d}.png",
                prompt=f"prompt {scene_index}", seed=scene_index * 1000,
                output_prefix=f"scene_{scene_index:03d}_raw",
                width=1280, height=720, frame_count=80, fps=16,
            )
            responses[workflow_signature(wf)] = [motion_mp4]
        comfy = FakeComfyClient(responses=responses)

        # Build minimal Story-like object.
        kf0 = tmp_path / "scene_000.png"
        kf0.write_bytes(b"fake_png")
        kf1 = tmp_path / "scene_001.png"
        kf1.write_bytes(b"fake_png")
        scenes = [
            SimpleNamespace(
                index=i, visual_prompt=f"prompt {i}",
                keyframe_path=tmp_path / f"scene_{i:03d}.png",
                video_path=None, video_duration_seconds=0.0,
                validation={},
            )
            for i in (0, 1)
        ]
        story = SimpleNamespace(
            id="story_test_001", track="atmospheric_horror",
            scenes=scenes,
            save=lambda *_args, **_kwargs: None,
        )

        # Build minimal ctx.
        story_dir = tmp_path / "stories" / story.id
        story_dir.mkdir(parents=True, exist_ok=True)

        ctx = SimpleNamespace(
            config=SimpleNamespace(
                settings={
                    "test": {
                        "comfy_client": comfy,
                        "workflow_template": workflow_template,
                    },
                    "runtime": {},
                },
                track=lambda _name: {
                    "quality_gates": {
                        "video_gates": {
                            "duration_target_seconds": 5.0,
                            "duration_tolerance_seconds": 0.2,
                            "black_frame_max_ratio": 0.05,
                            "motion_min_flow": 0.0,
                        },
                    },
                    "video_model": {
                        "width": 1280, "height": 720,
                        "frame_count": 80, "fps": 16,
                    },
                },
                config_dir=tmp_path / "config",
            ),
            story_path=lambda _story: story_dir / "story.json",
            db_path=tmp_path / "db",
        )

        stage = VideoGeneratorStage()
        result = await stage.run(story, ctx)

        assert result["scenes_total"] == 2
        assert result["scenes_succeeded"] == 2
        assert result["scenes_failed"] == 0
        for scene in scenes:
            assert scene.video_path is not None
            assert Path(scene.video_path).exists()
```

**Step 2: Run test, expect failure**

Run: `python -m pytest tests/integration/test_video_generator_stage.py -v`
Expected: failure with `ImportError: cannot import name 'VideoGeneratorStage'`.

**Step 3: Write the implementation**

Append to `src/platinum/pipeline/video_generator.py`:

```python
from typing import Any, ClassVar

from platinum.pipeline.stage import Stage


class VideoGeneratorStage(Stage):
    """Per-story video generation stage. See module docstring."""

    name: ClassVar[str] = "video_generator"

    async def run(self, story: Any, ctx: Any) -> dict[str, Any]:
        from platinum.utils.comfyui import HttpComfyClient
        from platinum.utils.workflow import load_workflow

        test_overrides = ctx.config.settings.get("test", {})
        injected_comfy = test_overrides.get("comfy_client")
        injected_template = test_overrides.get("workflow_template")
        comfy = injected_comfy or HttpComfyClient(
            host=ctx.config.settings.get("comfyui", {}).get(
                "host", "http://localhost:8188"
            ),
        )

        if injected_template is not None:
            workflow_template = injected_template
        else:
            workflow_template = load_workflow(
                "wan22_i2v", config_dir=ctx.config.config_dir
            )

        track_cfg = ctx.config.track(story.track)
        quality_gates = dict(track_cfg.get("quality_gates", {}))
        video_gates = dict(quality_gates.get("video_gates", {}))
        video_model_cfg = dict(track_cfg.get("video_model", {}))
        width = int(video_model_cfg.get("width", 1280))
        height = int(video_model_cfg.get("height", 720))
        frame_count = int(video_model_cfg.get("frame_count", 80))
        fps = int(video_model_cfg.get("fps", 16))

        # Resolve story dir (matches keyframe_generator pattern).
        if hasattr(ctx, "story_path"):
            try:
                story_dir = Path(ctx.story_path(story)).parent
            except Exception:  # noqa: BLE001
                story_dir = Path("data/stories") / story.id
        else:
            story_dir = Path("data/stories") / story.id
        output_root = story_dir / "clips"

        runtime = ctx.config.settings.get("runtime", {})
        scene_filter_raw = runtime.get("scene_filter")
        scene_filter: set[int] | None = (
            set(scene_filter_raw) if scene_filter_raw is not None else None
        )

        scenes_total = len(story.scenes)
        try:
            reports = await generate_video(
                story,
                workflow_template=workflow_template,
                comfy=comfy,
                output_root=output_root,
                gates_cfg=video_gates,
                scene_filter=scene_filter,
                width=width, height=height,
                frame_count=frame_count, fps=fps,
            )
        except VideoGenerationError:
            try:
                if hasattr(ctx, "story_path"):
                    story.save(ctx.story_path(story))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to save story.json after VideoGenerationError"
                )
            raise

        try:
            if hasattr(ctx, "story_path"):
                story.save(ctx.story_path(story))
        except Exception:  # noqa: BLE001
            logger.exception("failed to save story.json after stage completion")

        scenes_succeeded = sum(
            1 for s in story.scenes
            if s.video_path is not None and Path(s.video_path).exists()
        )
        return {
            "scenes_total": scenes_total,
            "scenes_succeeded": scenes_succeeded,
            "scenes_failed": scenes_total - scenes_succeeded,
            "retries_used": sum(r.retry_used for r in reports),
            "reports": [
                {
                    "scene_index": r.scene_index,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "retry_used": r.retry_used,
                }
                for r in reports
            ],
        }
```

**Step 4: Run test, expect pass**

Run: `python -m pytest tests/integration/test_video_generator_stage.py -v`
Expected: pass.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/integration/test_video_generator_stage.py
git commit -m "feat(s8.10): VideoGeneratorStage shell with ctx-based DI"
```

---

## Task 11: Stage `finally` aclose for ComfyClient lifecycle

**Files:**
- Modify: `src/platinum/pipeline/video_generator.py`
- Test: `tests/integration/test_video_generator_stage.py`

**Background:** S6.1 Lesson #6: Stage-constructed clients must be `aclose()`'d in `finally`. Test-injected clients are NOT closed (tests own their lifecycle).

**Step 1: Write the failing test**

Append to `tests/integration/test_video_generator_stage.py`:

```python
class TestStageResourceCleanup:
    @pytest.mark.asyncio
    async def test_stage_constructed_comfy_is_aclosed(self, tmp_path: Path) -> None:
        """When the Stage constructs its own HttpComfyClient (no test override),
        Stage.run must aclose it after both success and failure paths.
        """
        from platinum.pipeline.video_generator import VideoGeneratorStage

        # We assert by instrumenting a stand-in ComfyClient with an aclose tracker.
        class _TrackingComfy:
            aclose_called = False
            async def upload_image(self, path):
                return path.name
            async def generate_image(self, *, workflow, output_path):
                # Force fail so the run halts -- we still want aclose.
                raise RuntimeError("fail")
            async def aclose(self):
                self.aclose_called = True

        comfy = _TrackingComfy()
        # Intentionally NOT injecting via settings["test"] so we can assert
        # what Stage.run does with a Stage-constructed client. We achieve
        # this by monkeypatching HttpComfyClient at the import site.
        import platinum.pipeline.video_generator as vgm

        original_http = vgm.__dict__.get("HttpComfyClient", None)

        def _factory(host, **_kwargs):
            return comfy

        # Patch the constructor reference looked up inside Stage.run.
        # Stage.run's local "from platinum.utils.comfyui import HttpComfyClient"
        # picks up the symbol at call time, so patching the source module works.
        from platinum.utils import comfyui as cu
        original = cu.HttpComfyClient
        cu.HttpComfyClient = _factory  # type: ignore[assignment, misc]
        try:
            from types import SimpleNamespace
            kf = tmp_path / "scene_000.png"
            kf.write_bytes(b"fake_png")
            scenes = [SimpleNamespace(
                index=0, visual_prompt="x", keyframe_path=kf,
                video_path=None, video_duration_seconds=0.0, validation={},
            )]
            story = SimpleNamespace(
                id="story_test", track="atmospheric_horror",
                scenes=scenes, save=lambda *_a, **_k: None,
            )
            ctx = SimpleNamespace(
                config=SimpleNamespace(
                    settings={"test": {"workflow_template": _wan_template()},
                              "runtime": {}},
                    track=lambda _n: {
                        "quality_gates": {"video_gates": {
                            "duration_target_seconds": 5.0,
                            "duration_tolerance_seconds": 0.2,
                            "black_frame_max_ratio": 0.05,
                            "motion_min_flow": 0.0,
                        }},
                        "video_model": {"width": 1280, "height": 720,
                                        "frame_count": 80, "fps": 16},
                    },
                    config_dir=tmp_path / "config",
                ),
                story_path=lambda _s: tmp_path / "stories" / "story.json",
                db_path=tmp_path / "db",
            )
            (tmp_path / "stories").mkdir(parents=True, exist_ok=True)

            stage = VideoGeneratorStage()
            with pytest.raises(Exception):
                await stage.run(story, ctx)
        finally:
            cu.HttpComfyClient = original  # type: ignore[assignment, misc]

        assert comfy.aclose_called is True
```

**Step 2: Run test, expect failure**

Run: `python -m pytest tests/integration/test_video_generator_stage.py::TestStageResourceCleanup -v`
Expected: failure (the current Stage doesn't `aclose` anything).

**Step 3: Update Stage.run to wrap iteration in try/finally**

Replace the existing `try ... except VideoGenerationError ... raise` block in `VideoGeneratorStage.run` with:

```python
        scenes_total = len(story.scenes)
        try:
            try:
                reports = await generate_video(
                    story,
                    workflow_template=workflow_template,
                    comfy=comfy,
                    output_root=output_root,
                    gates_cfg=video_gates,
                    scene_filter=scene_filter,
                    width=width, height=height,
                    frame_count=frame_count, fps=fps,
                )
            except VideoGenerationError:
                try:
                    if hasattr(ctx, "story_path"):
                        story.save(ctx.story_path(story))
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "failed to save story.json after VideoGenerationError"
                    )
                raise

            try:
                if hasattr(ctx, "story_path"):
                    story.save(ctx.story_path(story))
            except Exception:  # noqa: BLE001
                logger.exception("failed to save story.json after stage completion")
        finally:
            # Close any clients the Stage constructed (skip test-injected ones).
            if injected_comfy is None and hasattr(comfy, "aclose"):
                try:
                    await comfy.aclose()
                except Exception:  # noqa: BLE001
                    logger.exception("failed to aclose ComfyClient")
```

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/integration/test_video_generator_stage.py -v`
Expected: 2 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/pipeline/video_generator.py tests/integration/test_video_generator_stage.py
git commit -m "feat(s8.11): VideoGeneratorStage closes Stage-constructed ComfyClient in finally"
```

---

## Task 12: Per-track YAML `video_gates` block + parametrized config test

**Files:**
- Modify: `config/tracks/atmospheric_horror.yaml`
- Modify: `config/tracks/folktales_world_myths.yaml`
- Modify: `config/tracks/childrens_fables.yaml`
- Modify: `config/tracks/scifi_concept.yaml`
- Modify: `config/tracks/slice_of_life.yaml`
- Test: `tests/integration/test_quality_gates_config.py` (extend existing tests)

**Step 1: Write the failing test**

Append to `tests/integration/test_quality_gates_config.py` a new test (or add to existing parametrized group):

```python
import pytest

@pytest.mark.parametrize(
    "track,expected_motion_floor",
    [
        ("atmospheric_horror", 0.3),
        ("folktales_world_myths", 0.5),
        ("childrens_fables", 0.7),
        ("scifi_concept", 0.5),
        ("slice_of_life", 0.6),
    ],
)
def test_video_gates_block_present_per_track(
    track: str, expected_motion_floor: float
) -> None:
    """Every track YAML has video_gates with all 4 keys; track-tuned motion floor."""
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path("config/tracks") / f"{track}.yaml").read_text(encoding="utf-8")
    )
    quality_gates = cfg["track"]["quality_gates"]
    assert "video_gates" in quality_gates, f"{track}: video_gates block missing"
    vg = quality_gates["video_gates"]
    assert vg["duration_target_seconds"] == 5.0
    assert vg["duration_tolerance_seconds"] == 0.2
    assert vg["black_frame_max_ratio"] == 0.05
    assert vg["motion_min_flow"] == expected_motion_floor
```

**Step 2: Run test, expect failures**

Run: `python -m pytest tests/integration/test_quality_gates_config.py -k video_gates -v`
Expected: 5 parametrized failures, each with `KeyError: 'video_gates'`.

**Step 3: Add `video_gates` block to each track YAML**

For each of the 5 track YAMLs, locate the `track.quality_gates:` block and append a `video_gates:` sub-block:

`config/tracks/atmospheric_horror.yaml`:
```yaml
track:
  quality_gates:
    # ... existing image gates ...
    video_gates:
      duration_target_seconds: 5.0
      duration_tolerance_seconds: 0.2
      black_frame_max_ratio: 0.05
      motion_min_flow: 0.3
```

Repeat with `motion_min_flow: 0.5` for `folktales_world_myths.yaml`, `0.7` for `childrens_fables.yaml`, `0.5` for `scifi_concept.yaml`, `0.6` for `slice_of_life.yaml`.

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/integration/test_quality_gates_config.py -v`
Expected: all parametrized cases pass.

**Step 5: Commit**

```bash
git add config/tracks/*.yaml tests/integration/test_quality_gates_config.py
git commit -m "feat(s8.12): per-track video_gates block (motion_min_flow tuned per track)"
```

---

## Task 13: CLI `platinum video <story>` command

**Files:**
- Modify: `src/platinum/cli.py`
- Test: `tests/integration/test_video_command.py` (new file)

**Background:** Mirror `platinum keyframes` exactly. Required arg: `story_id`. Options: `--scenes` (csv), `--dry-run`, `--rerun-all`. Resolves Story from `data/stories/<id>/story.json`, builds Config + ctx, invokes `VideoGeneratorStage.run`, prints JSON summary at exit.

**Step 1: Write the failing tests**

Create `tests/integration/test_video_command.py`:

```python
"""Integration tests for `platinum video` CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


class TestVideoCommand:
    def test_dry_run_lists_targets_without_running_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from platinum.cli import app
        # Set CWD to tmp_path so data/stories/<id>/ is discoverable there.
        monkeypatch.chdir(tmp_path)

        # Build a minimal story directory with one scene whose keyframe_path
        # is set + the keyframe file actually exists.
        story_dir = tmp_path / "data" / "stories" / "story_test_001"
        keyframes_dir = story_dir / "keyframes" / "scene_000"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        kf = keyframes_dir / "candidate_0.png"
        kf.write_bytes(b"fake_png")
        story_json = story_dir / "story.json"
        story_data = {
            "id": "story_test_001",
            "track": "atmospheric_horror",
            "source": {"id": "src", "kind": "test", "title": "t",
                       "url": "https://x", "license": "PD", "fetched_at": "2026-04-30T00:00:00",
                       "raw_text": ""},
            "adapted": None,
            "scenes": [{
                "id": "scene_000", "index": 0,
                "narration_text": "n", "narration_duration_seconds": 5.0,
                "visual_prompt": "x", "negative_prompt": None,
                "keyframe_path": str(kf), "video_path": None,
                "video_duration_seconds": 0.0,
                "music_cue": None, "sfx_cues": [],
                "validation": {}, "review_status": "PENDING",
            }],
            "audio": {}, "review_status": "PENDING",
        }
        story_json.write_text(json.dumps(story_data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["video", "story_test_001", "--dry-run"])
        assert result.exit_code == 0, result.stdout
        assert "scene_000" in result.stdout or "scene 0" in result.stdout

    def test_missing_story_id_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from platinum.cli import app
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(app, ["video", "nonexistent_story_id"])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    def test_scenes_filter_parses_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--scenes 0,2 should produce a scene_filter set in runtime settings.

        We verify by checking dry-run output mentions only the targeted scenes.
        """
        from platinum.cli import app
        monkeypatch.chdir(tmp_path)

        story_dir = tmp_path / "data" / "stories" / "story_test_002"
        story_dir.mkdir(parents=True, exist_ok=True)
        for idx in (0, 1, 2):
            kdir = story_dir / "keyframes" / f"scene_{idx:03d}"
            kdir.mkdir(parents=True, exist_ok=True)
            (kdir / "candidate_0.png").write_bytes(b"fake_png")

        story_json = story_dir / "story.json"
        scenes = []
        for idx in (0, 1, 2):
            scenes.append({
                "id": f"scene_{idx:03d}", "index": idx,
                "narration_text": "n", "narration_duration_seconds": 5.0,
                "visual_prompt": "x", "negative_prompt": None,
                "keyframe_path": str(
                    story_dir / "keyframes" / f"scene_{idx:03d}" / "candidate_0.png"
                ),
                "video_path": None, "video_duration_seconds": 0.0,
                "music_cue": None, "sfx_cues": [],
                "validation": {}, "review_status": "PENDING",
            })
        story_data = {
            "id": "story_test_002", "track": "atmospheric_horror",
            "source": {"id": "src", "kind": "test", "title": "t",
                       "url": "https://x", "license": "PD",
                       "fetched_at": "2026-04-30T00:00:00", "raw_text": ""},
            "adapted": None, "scenes": scenes,
            "audio": {}, "review_status": "PENDING",
        }
        story_json.write_text(json.dumps(story_data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            app, ["video", "story_test_002", "--scenes", "0,2", "--dry-run"]
        )
        assert result.exit_code == 0, result.stdout
        assert "scene_000" in result.stdout or "scene 0" in result.stdout
        assert "scene_002" in result.stdout or "scene 2" in result.stdout
        # scene 1 NOT in dry-run output.
        assert "scene_001" not in result.stdout
```

**Step 2: Run tests, expect failures**

Run: `python -m pytest tests/integration/test_video_command.py -v`
Expected: 3 failures with `Error: No such command 'video'`.

**Step 3: Implement the CLI command**

Add to `src/platinum/cli.py`:

```python
@app.command()
def video(
    story: str = typer.Argument(..., help="Story id (under data/stories/)."),
    scenes: str | None = typer.Option(
        None, "--scenes",
        help="Comma-separated 1-indexed scene numbers (e.g., '1,8,16'). "
             "Default: all scenes with keyframe_path set.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would be generated; submit no workflows.",
    ),
    rerun_all: bool = typer.Option(
        False, "--rerun-all",
        help="Force regeneration of all scenes (ignore existing video_path).",
    ),
) -> None:
    """Generate Wan 2.2 I2V clips for each scene's keyframe."""
    from platinum.config import Config
    from platinum.models.story import Story
    from platinum.pipeline.context import build_default_context
    from platinum.pipeline.video_generator import VideoGeneratorStage

    cfg = Config()
    story_path = cfg.data_dir / "stories" / story / "story.json"
    if not story_path.exists():
        console.print(f"[red]Story not found:[/red] {story} ({story_path})")
        raise typer.Exit(code=1)
    story_obj = Story.load(story_path)

    # Parse --scenes csv into a set of ints (1-indexed -> 0-indexed).
    scene_filter: set[int] | None = None
    if scenes is not None:
        try:
            scene_filter = {int(s.strip()) - 1 for s in scenes.split(",") if s.strip()}
        except ValueError as exc:
            console.print(f"[red]Invalid --scenes value:[/red] {scenes!r} ({exc})")
            raise typer.Exit(code=2) from exc

    # --rerun-all: wipe video_path on every targeted scene.
    if rerun_all:
        for scene_obj in story_obj.scenes:
            if scene_filter is None or scene_obj.index in scene_filter:
                scene_obj.video_path = None

    if dry_run:
        console.print("[yellow]Dry run -- no workflows submitted.[/yellow]")
        for scene_obj in story_obj.scenes:
            if scene_filter is not None and scene_obj.index not in scene_filter:
                continue
            if scene_obj.video_path is not None:
                console.print(
                    f"  scene_{scene_obj.index:03d}: SKIP (video_path set)"
                )
                continue
            if scene_obj.keyframe_path is None:
                console.print(
                    f"  scene_{scene_obj.index:03d}: HALT (no keyframe_path)"
                )
                continue
            console.print(
                f"  scene_{scene_obj.index:03d}: TARGET ({scene_obj.keyframe_path})"
            )
        return

    # Stage execution path.
    runtime = cfg.settings.setdefault("runtime", {})
    runtime["scene_filter"] = scene_filter

    ctx = build_default_context(cfg, story_obj)
    stage = VideoGeneratorStage()
    result = asyncio.run(stage.run(story_obj, ctx))
    console.print(json.dumps(result, indent=2, default=str))


# Add at top of cli.py if not already imported:
import json
```

If `build_default_context` doesn't exist (check `src/platinum/pipeline/context.py`), use the existing `keyframes` command as the template — replicate however it builds its ctx. Adjust the import accordingly.

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/integration/test_video_command.py -v`
Expected: 3 passing tests.

**Step 5: Commit**

```bash
git add src/platinum/cli.py tests/integration/test_video_command.py
git commit -m "feat(s8.13): platinum video CLI with --scenes/--dry-run/--rerun-all"
```

---

## Task 14: `config/workflows/wan22_i2v.json` + signature integration test

**Files:**
- Create: `config/workflows/wan22_i2v.json`
- Test: `tests/integration/test_workflow_files.py` (extend if exists, else new)

**Background:** Author the actual ComfyUI workflow JSON for Wan 2.2 I2V using `kijai/ComfyUI-WanVideoWrapper` nodes. Required `_meta.role` keys: `image_in`, `prompt`, `seed`, `video_out`. Optional: `width`, `height`, `frame_count`, `fps`. The exact node `class_type` strings depend on the WanVideoWrapper extension's published API.

**Step 1: Write the failing test**

Create or extend `tests/integration/test_workflow_files.py`:

```python
"""Integration tests for shipped workflow JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from platinum.utils.workflow import inject_video


class TestWan22I2VWorkflow:
    def test_workflow_loads_and_has_required_roles(self) -> None:
        path = Path("config/workflows/wan22_i2v.json")
        assert path.exists(), f"missing workflow file: {path}"
        wf = json.loads(path.read_text(encoding="utf-8"))
        roles = wf.get("_meta", {}).get("role", {})
        for required in ("image_in", "prompt", "seed", "video_out"):
            assert required in roles, f"missing required role: {required}"
            assert roles[required] in wf, (
                f"role {required} -> node {roles[required]} not present in workflow"
            )

    def test_inject_video_round_trip_succeeds(self) -> None:
        path = Path("config/workflows/wan22_i2v.json")
        wf = json.loads(path.read_text(encoding="utf-8"))
        out = inject_video(
            wf,
            image_in="scene_000.png",
            prompt="probe",
            seed=42,
            output_prefix="scene_000_raw",
            width=1280, height=720, frame_count=80, fps=16,
        )
        # Each required role's node has the expected mutation visible.
        roles = out["_meta"]["role"]
        assert out[roles["image_in"]]["inputs"]["image"] == "scene_000.png"
        assert out[roles["prompt"]]["inputs"]["text"] == "probe"
        assert out[roles["seed"]]["inputs"]["seed"] == 42
        assert out[roles["video_out"]]["inputs"]["filename_prefix"] == "scene_000_raw"
```

**Step 2: Run tests, expect failures**

Run: `python -m pytest tests/integration/test_workflow_files.py::TestWan22I2VWorkflow -v`
Expected: 2 failures (file doesn't exist).

**Step 3: Author `config/workflows/wan22_i2v.json`**

Author a minimal Wan 2.2 I2V workflow. **CRITICAL:** node `class_type` strings must match the WanVideoWrapper extension API. Reference: <https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/example_workflows>. As of S8 design time (2026-04-30), a minimal I2V graph uses these wrapper-provided nodes:

- `WanVideoModelLoader` (high-noise expert)
- `WanVideoModelLoader` (low-noise expert) -- second instance
- `WanVideoVAELoader`
- `WanVideoT5TextEncoder` (UMT5)
- `WanVideoTextEncode` (positive prompt encoding)
- `LoadImage` (input keyframe)
- `WanVideoImageEncode` (image-to-latent for I2V conditioning)
- `WanVideoSampler` (handles MoE expert switching internally)
- `WanVideoDecode` (latent → frames)
- `VHS_VideoCombine` (frames → MP4)

Suggested `wan22_i2v.json`:

```json
{
  "_meta": {
    "comment": "Wan 2.2 I2V-A14B workflow (S8 Phase A). Requires ComfyUI-WanVideoWrapper.",
    "role": {
      "image_in": "10",
      "prompt": "20",
      "seed": "30",
      "video_out": "60",
      "width": "30",
      "height": "30",
      "frame_count": "40",
      "fps": "60"
    }
  },
  "1": {
    "class_type": "WanVideoModelLoader",
    "inputs": {
      "model": "wan2_2_i2v_high_noise.safetensors",
      "base_precision": "fp16",
      "load_device": "main_device"
    }
  },
  "2": {
    "class_type": "WanVideoModelLoader",
    "inputs": {
      "model": "wan2_2_i2v_low_noise.safetensors",
      "base_precision": "fp16",
      "load_device": "main_device"
    }
  },
  "3": {
    "class_type": "WanVideoVAELoader",
    "inputs": {
      "model_name": "wan2_2_vae.pth",
      "precision": "fp16"
    }
  },
  "4": {
    "class_type": "WanVideoT5TextEncoder",
    "inputs": {
      "model_name": "umt5_xxl.pth",
      "precision": "bf16"
    }
  },
  "10": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "PLACEHOLDER.png"
    }
  },
  "20": {
    "class_type": "WanVideoTextEncode",
    "inputs": {
      "t5_encoder": ["4", 0],
      "text": "PLACEHOLDER",
      "negative_text": ""
    }
  },
  "25": {
    "class_type": "WanVideoImageEncode",
    "inputs": {
      "vae": ["3", 0],
      "image": ["10", 0]
    }
  },
  "30": {
    "class_type": "WanVideoSampler",
    "inputs": {
      "model_high_noise": ["1", 0],
      "model_low_noise": ["2", 0],
      "image_embeds": ["25", 0],
      "text_embeds": ["20", 0],
      "seed": 0,
      "steps": 30,
      "cfg": 6.0,
      "scheduler": "unipc",
      "width": 1280,
      "height": 720
    }
  },
  "40": {
    "class_type": "WanVideoLatentVideo",
    "inputs": {
      "samples": ["30", 0],
      "length": 80
    }
  },
  "50": {
    "class_type": "WanVideoDecode",
    "inputs": {
      "vae": ["3", 0],
      "samples": ["40", 0]
    }
  },
  "60": {
    "class_type": "VHS_VideoCombine",
    "inputs": {
      "images": ["50", 0],
      "frame_rate": 16,
      "filename_prefix": "wan_output",
      "format": "video/h264-mp4"
    }
  }
}
```

**IMPORTANT:** This is a *starting* shape. The exact node names and input keys for WanVideoWrapper drift between versions. During Rental 1 probe (Step 7 of the S8 runbook), the executor may need to update node names/inputs to match what the live extension actually exposes. Update this file if Rental 1 surfaces discrepancies — preserve the `_meta.role` mapping; only the node `class_type` and `inputs` keys may need to shift.

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/integration/test_workflow_files.py::TestWan22I2VWorkflow -v`
Expected: 2 passing tests.

**Step 5: Commit**

```bash
git add config/workflows/wan22_i2v.json tests/integration/test_workflow_files.py
git commit -m "feat(s8.14): wan22_i2v.json workflow scaffold + signature test"
```

---

## Task 15: `preflight_check.py` Wan additions

**Files:**
- Modify: `scripts/preflight_check.py`
- Test: `tests/unit/test_preflight_check.py`

**Background:** Three new fail-fast checks: Wan workflow JSON valid, Wan weights present (4 files), WanVideoWrapper extension importable.

**Step 1: Write the failing tests**

Append to `tests/unit/test_preflight_check.py`:

```python
class TestWanPreflightChecks:
    def test_check_wan_workflow_valid(self, tmp_path) -> None:
        """Wan workflow JSON has all required _meta.role entries."""
        from scripts.preflight_check import _check_wan_workflow_json

        good = tmp_path / "wan_good.json"
        good.write_text('{"_meta":{"role":{"image_in":"10","prompt":"20","seed":"30","video_out":"60"}},"10":{},"20":{},"30":{},"60":{}}')
        ok, msg = _check_wan_workflow_json(good)
        assert ok, msg

        bad = tmp_path / "wan_bad.json"
        bad.write_text('{"_meta":{"role":{"image_in":"10"}},"10":{}}')
        ok, msg = _check_wan_workflow_json(bad)
        assert not ok
        assert "missing roles" in msg

    def test_check_wan_weights_present(self, tmp_path) -> None:
        from scripts.preflight_check import _check_wan_weights

        # Empty dir -> fail.
        ok, msg = _check_wan_weights(tmp_path)
        assert not ok
        assert "high_noise" in msg or "low_noise" in msg

        # All 4 files present with reasonable size.
        (tmp_path / "diffusion_models").mkdir()
        (tmp_path / "diffusion_models" / "wan2_2_i2v_high_noise.safetensors").write_bytes(
            b"x" * (2_000_000_000)  # 2GB
        )
        (tmp_path / "diffusion_models" / "wan2_2_i2v_low_noise.safetensors").write_bytes(
            b"x" * (2_000_000_000)
        )
        (tmp_path / "vae").mkdir()
        (tmp_path / "vae" / "wan2_2_vae.pth").write_bytes(b"x" * (200_000_000))
        (tmp_path / "text_encoders").mkdir()
        (tmp_path / "text_encoders" / "umt5_xxl.pth").write_bytes(b"x" * (5_000_000_000))
        ok, msg = _check_wan_weights(tmp_path)
        assert ok, msg
```

**Step 2: Run tests, expect failures**

Run: `python -m pytest tests/unit/test_preflight_check.py::TestWanPreflightChecks -v`
Expected: failures (functions don't exist yet).

**Step 3: Add the checks**

Append to `scripts/preflight_check.py`:

```python
WAN_REQUIRED_ROLES = frozenset({"image_in", "prompt", "seed", "video_out"})

WAN_WEIGHT_FILES = (
    ("diffusion_models", "wan2_2_i2v_high_noise.safetensors", 1_000_000_000),
    ("diffusion_models", "wan2_2_i2v_low_noise.safetensors", 1_000_000_000),
    ("vae", "wan2_2_vae.pth", 100_000_000),
    ("text_encoders", "umt5_xxl.pth", 1_000_000_000),
)


def _check_wan_workflow_json(path: Path) -> tuple[bool, str]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"wan workflow JSON load failed: {exc!r}"
    roles = data.get("_meta", {}).get("role", {})
    missing = WAN_REQUIRED_ROLES - set(roles)
    if missing:
        return False, f"wan workflow missing roles: {sorted(missing)}"
    for role, node_id in roles.items():
        if role in WAN_REQUIRED_ROLES and node_id not in data:
            return False, f"wan role '{role}' -> node '{node_id}' not in workflow"
    return True, f"wan workflow OK (roles: {sorted(WAN_REQUIRED_ROLES)})"


def _check_wan_weights(models_dir: Path) -> tuple[bool, str]:
    missing: list[str] = []
    too_small: list[str] = []
    for subdir, filename, min_size in WAN_WEIGHT_FILES:
        p = Path(models_dir) / subdir / filename
        if not p.exists():
            missing.append(str(p))
            continue
        if p.stat().st_size < min_size:
            too_small.append(f"{p} ({p.stat().st_size}B < {min_size}B)")
    if missing:
        return False, f"wan weights missing: {missing}"
    if too_small:
        return False, f"wan weights too small (download incomplete?): {too_small}"
    return True, "wan weights OK (4 files present, sizes plausible)"


def _check_wan_extension_importable() -> tuple[bool, str]:
    """Confirm the ComfyUI-WanVideoWrapper extension can be imported.

    On the box this is just a path/sys.path check -- we don't actually run
    its node registration here.
    """
    import sys
    extension_path = Path("/workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper")
    if not extension_path.exists():
        return False, f"WanVideoWrapper not at {extension_path}"
    init_py = extension_path / "__init__.py"
    if not init_py.exists():
        return False, f"WanVideoWrapper __init__.py missing at {init_py}"
    return True, f"WanVideoWrapper present at {extension_path}"
```

Then wire these into the existing main() / argparse so `--video` (or always-on if Wan workflow path is provided) calls them. Look at how the existing checks are invoked and follow the same shape.

**Step 4: Run tests, expect pass**

Run: `python -m pytest tests/unit/test_preflight_check.py -v`
Expected: passing.

**Step 5: Commit**

```bash
git add scripts/preflight_check.py tests/unit/test_preflight_check.py
git commit -m "feat(s8.15): preflight checks for Wan workflow + weights + extension"
```

---

## Task 16: `vast_setup.sh` Wan 2.2 + WanVideoWrapper extension

**Files:**
- Modify: `scripts/vast_setup.sh`

**Background:** Replace the stale single-file Wan URL with the MoE expert files + VAE + UMT5. Clone the WanVideoWrapper extension. No tests — manual verification on the box during Rental 1.

**Step 1: Locate the existing Wan block**

Find the lines flagged in earlier sessions (around line 229-232):
```bash
# Wan 2.2 I2V weights -- not used by S6.1, deferred OK.
dl "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main/diffusion_pytorch_model.safetensors" \
   "$MODELS_DIR/checkpoints/wan22_i2v.safetensors" || \
    log "Wan 2.2 weights -- adjust URL if HuggingFace path changed (deferred to S8)"
```

**Step 2: Replace with MoE expert files + VAE + UMT5**

Replace the block above with:

```bash
# Wan 2.2 I2V-A14B (MoE: 2 experts + VAE + UMT5 text encoder). S8 Phase A.
log "Provisioning Wan 2.2 I2V-A14B weights..."
WAN_BASE="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main"
mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/vae" "$MODELS_DIR/text_encoders"

dl "$WAN_BASE/high_noise_model.safetensors" \
   "$MODELS_DIR/diffusion_models/wan2_2_i2v_high_noise.safetensors" || \
    log "WARN: Wan 2.2 high-noise expert -- adjust URL if HF path moved"
dl "$WAN_BASE/low_noise_model.safetensors" \
   "$MODELS_DIR/diffusion_models/wan2_2_i2v_low_noise.safetensors" || \
    log "WARN: Wan 2.2 low-noise expert -- adjust URL if HF path moved"
dl "$WAN_BASE/Wan2.1_VAE.pth" \
   "$MODELS_DIR/vae/wan2_2_vae.pth" || \
    log "WARN: Wan VAE -- adjust URL if HF path moved"
dl "$WAN_BASE/models_t5_umt5-xxl-enc-bf16.pth" \
   "$MODELS_DIR/text_encoders/umt5_xxl.pth" || \
    log "WARN: UMT5 text encoder -- adjust URL if HF path moved"

# ComfyUI-WanVideoWrapper extension (provides WanVideo* node classes).
log "Cloning ComfyUI-WanVideoWrapper extension..."
WANWRAPPER_DIR="$COMFY_DIR/custom_nodes/ComfyUI-WanVideoWrapper"
if [ -d "$WANWRAPPER_DIR" ]; then
    git -C "$WANWRAPPER_DIR" pull --ff-only || log "WARN: WanVideoWrapper pull failed"
else
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper "$WANWRAPPER_DIR" || \
        log "WARN: WanVideoWrapper clone failed"
fi
if [ -f "$WANWRAPPER_DIR/requirements.txt" ]; then
    pip install -r "$WANWRAPPER_DIR/requirements.txt" || \
        log "WARN: WanVideoWrapper requirements install failed"
fi
```

**Step 3: Manual verify (cannot test offline)**

Re-read the entire script and confirm:
1. Variables `$MODELS_DIR` and `$COMFY_DIR` are defined earlier.
2. The `dl` and `log` shell functions are defined earlier.
3. The new block does not break any pre-existing exit-on-error trap.
4. There are no CRLF line endings introduced (Windows clones can add these — verify with `cat -A scripts/vast_setup.sh | head -5` on a Linux system, or just check git diff for `^M` characters).

**Step 4: Commit**

```bash
git add scripts/vast_setup.sh
git commit -m "feat(s8.16): vast_setup -- Wan 2.2 MoE weights + WanVideoWrapper ext"
```

---

## Task 17: `docs/runbooks/vast-ai-video-smoke.md`

**Files:**
- Create: `docs/runbooks/vast-ai-video-smoke.md`

**Step 1: Author the runbook**

Create `docs/runbooks/vast-ai-video-smoke.md` modeled on `docs/runbooks/vast-ai-keyframe-smoke.md`. Outline:

```markdown
# vast.ai Wan 2.2 I2V smoke runbook (S8 Phase A)

Two-rental probe -> full pattern. Closure: >=14/16 Cask scenes produce
viable 5s clips on first eye-check after Rental 2.

## Prerequisites
- Local main at S8 closeout commit (workflow + Stage + CLI shipped).
- Existing keyframes for story_2026_04_25_001 under
  data/stories/story_2026_04_25_001/keyframes/ (16 scenes x 3 candidates).
- HF_TOKEN with FLUX.1-dev + Wan-AI gated access.
- vastai CLI authenticated, vast_api_key on disk.

## Step 1: Provision A6000
```bash
vastai search offers \
  'gpu_name=RTX_A6000 cpu_ram>=64 disk_space>=80 verified=true' \
  -o 'dph_total'
vastai create instance <id> --image nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04 \
  --disk 80
# wait for instance running, get ssh host:port
```

## Step 2: Run vast_setup.sh
```bash
git archive HEAD | ssh -p <port> root@<host> 'mkdir -p /workspace/platinum && tar -x -C /workspace/platinum'
ssh -p <port> root@<host> 'dos2unix /workspace/platinum/scripts/*.sh && bash /workspace/platinum/scripts/vast_setup.sh'
# expect ~15-20 min for Flux + Wan additions
```

## Step 3: Preflight
```bash
ssh -p <port> root@<host> 'cd /workspace/platinum && python scripts/preflight_check.py --workflow config/workflows/wan22_i2v.json --video'
# expect 5 checks GREEN: HF token, ComfyUI alive, score-server alive,
# Wan workflow valid, Wan weights present + extension importable
```

## Step 4: Rental 1 (probe -- scenes 1, 8, 16)
On local:
```bash
export PLATINUM_COMFYUI_HOST=http://<host>:8188
platinum video story_2026_04_25_001 --scenes 1,8,16
```
Expected ~15 min wall clock. Outputs at:
data/stories/story_2026_04_25_001/clips/scene_{001,008,016}_raw.mp4

Eye-check on local (open in VLC / ffplay):
- Each clip is exactly 5 seconds.
- Has visible motion -- not a still frame, not a slideshow.
- Scene content recognizable from keyframe.
- No corruption, no excessive black frames, no obvious artifacts.

**Closure target Step 4:** >=2 of 3 viable. If <2, investigate (Wan node
class_type drift? Model load OOM? Sampler params off?) before Rental 2.

## Step 5: Rental 2 (full -- 16 scenes)
If Step 4 closure met:
```bash
platinum video story_2026_04_25_001
```
Expected ~80-100 min wall clock. Outputs:
data/stories/story_2026_04_25_001/clips/scene_*_raw.mp4 (16 files).

## Step 6: Eye-check + closure
Local eye-check 16 clips. Closure: >=14/16 viable on first eye-check.
If <14, scope an S8.A.1 follow-up before moving to S9 (upscaling).

## Step 7: Calibration (if needed)
If video_gates fired false-positives during Rental 2 (rejected viable
clips), tighten thresholds in track YAMLs based on actual metrics from
the run. Common axes: motion_min_flow (Cask scenes are sometimes too
static for 0.3 floor), black_frame_max_ratio (transition flashes).

## Step 8: Teardown
```bash
vastai destroy instance <id>
```
S8 ends; S9 (RealESRGAN upscale) is its own session, will rent its own
A6000 / H100.

## Gotcha checklist
- HF_TOKEN must be on box AND have Wan-AI gated access.
- dos2unix on .sh files (Windows CRLF strikes again).
- Wan node class_type may drift; preflight workflow signature check
  catches stale on-box workflow.
- A6000 has been infrastructurally flaky during heavy GPU work
  (S6.2 retro). Reboot if host hangs > 30s.
```

**Step 2: Commit**

```bash
git add docs/runbooks/vast-ai-video-smoke.md
git commit -m "docs(runbooks): vast-ai-video-smoke for S8 two-rental probe -> full"
```

---

## Task 18: Final cumulative review + closeout chore

**Files:**
- Modify (if needed): any file surfaced by review.

**Background:** S6.1's Lesson #6 — final cumulative review catches gaps individual task reviews miss. This task is one careful re-pass over everything S8 touched.

**Step 1: Run full test suite**

```bash
python -m pytest -q
```
Expected: ≥559 tests pass (521 baseline + ~38 S8). 0 failures, 0 skips.

```bash
ruff check src tests scripts
```
Expected: clean.

```bash
mypy src
```
Expected: 2 pre-existing deferrals only.

**Step 2: Cumulative cross-file review**

Re-read end-to-end with these questions:

1. **Resource leaks:** does `VideoGeneratorStage.run` close every Stage-constructed client in `finally`? (Lesson 6.1#6)
2. **Atomic save:** is `story.save` called at every safe checkpoint (after each scene success in `generate_video`, on success at end of Stage, on failure path before re-raising)?
3. **POSIX paths:** any `Path("data\\stories\\...")` Windows separators slipping in? `Story.save` should normalise but check call sites.
4. **Workflow signature:** does the runbook reference the workflow signature check? Does the preflight implementation actually compute it?
5. **Halt-vs-skip:** are all `VideoGenerationError` raises tagged `retryable=True/False` correctly per the design?
6. **Seed disjointness:** does the actual `_seed_for_scene` formula match what `keyframe_generator._seeds_for_scene` produces? They're disjoint (video uses retry 0/1, keyframes use candidate_idx 0/1/2 -- both 0 and 1 collide). Is that OK? (Yes -- different output extensions. But document it.)
7. **gates_cfg key consistency:** every track YAML, the test, the impl, the design doc — all use the same 4 keys exactly?
8. **Wan workflow JSON:** does it actually parse and pass `inject_video` round-trip? Does the integration test catch a broken edit?
9. **Runbook closure target:** matches the design (≥14/16)?

If anything found, fix and add a regression test.

**Step 3: Closeout commit**

```bash
git add -A
git commit -m "chore(s8): Phase 1 close -- workflow scaffolded, Stage + CLI shipped, runbook ready for A6000"
```

If no changes were needed (cumulative review found nothing), skip the commit and just announce closeout.

**Step 4: Update memory**

Update `C:/Users/claws/.claude/projects/C--Users-claws-OneDrive-Desktop-platinum/memory/`:
- Add `project_s8_phase1.md` with file lists, test counts, and a "Phase 2 awaiting A6000 rental" note.
- Update `MEMORY.md` index with the new file.
- Mark `reference_s8_pickup.md` as superseded by `project_s8_phase1.md` for ongoing-state tracking.

**Step 5: Announce closeout to user**

Summary:
- All 18 tasks complete; tests/lint/types green.
- Local commits on `main`. **Not pushed** — S7.1 rule: nothing pushed until live verify produces actual MP4 output.
- Phase 2 (live A6000 rental, two-rental probe→full) awaiting user-driven rental.
- Estimated Phase 2 cost: $7-20 over 2 rentals.
- Runbook at `docs/runbooks/vast-ai-video-smoke.md`.
- Closure: ≥14/16 viable Cask clips on first eye-check after Rental 2.

---

## Closure criteria for S8 Phase 1

- [ ] All 18 tasks committed in order on local `main` (not pushed).
- [ ] `python -m pytest -q` clean: ≥559 passing, 0 failing, 0 skipped.
- [ ] `ruff check src tests scripts` clean.
- [ ] `mypy src` reports only the 2 pre-existing deferrals.
- [ ] `config/workflows/wan22_i2v.json` exists, parses, round-trips through `inject_video`.
- [ ] All 5 track YAMLs have `quality_gates.video_gates` block with all 4 required keys.
- [ ] `vast_setup.sh` references the Wan MoE files + WanVideoWrapper extension.
- [ ] `preflight_check.py` exposes the 3 new Wan checks + integration with main().
- [ ] `docs/runbooks/vast-ai-video-smoke.md` covers two-rental probe → full sequence.
- [ ] Memory updated: `project_s8_phase1.md` added; `MEMORY.md` index updated.

## Phase 2 closure (live verify, separate session)

- [ ] Rental 1: ≥2 of 3 (Cask scenes 1/8/16) viable on local eye-check.
- [ ] Rental 2: ≥14 of 16 Cask scenes viable on local eye-check.
- [ ] Push S8 branch to `origin/main`.
- [ ] Memory updated: `project_s8_verify_complete.md` written.
