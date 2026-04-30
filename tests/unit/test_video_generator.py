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
        from dataclasses import FrozenInstanceError

        from platinum.pipeline.video_generator import VideoReport

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
        # Register both seeds (initial and retry) to handle retry loop
        wf_seed_0 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1001,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
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

        scene = SimpleNamespace(
            index=1,
            visual_prompt="x",
            keyframe_path=keyframe,
            video_path=None,
        )

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
                width=1280,
                height=720,
                frame_count=80,
                fps=16,
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
        make_test_video(
            static_mp4, n_frames=80, fps=16, color=(120, 120, 120), size=(64, 64)
        )

        workflow_template = _wan_template_for_tests()
        # Register both seeds (initial and retry) to handle retry loop
        wf_seed_0 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1001,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        comfy = FakeComfyClient(
            responses={
                workflow_signature(wf_seed_0): [static_mp4],
                workflow_signature(wf_seed_1): [static_mp4],
            }
        )

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace

        scene = SimpleNamespace(
            index=1,
            visual_prompt="x",
            keyframe_path=keyframe,
            video_path=None,
        )

        with pytest.raises(VideoGenerationError) as excinfo:
            await generate_video_for_scene(
                scene,
                workflow_template=workflow_template,
                comfy=comfy,
                output_path=tmp_path / "clips" / "scene_001_raw.mp4",
                gates_cfg={
                    "duration_target_seconds": 5.0,
                    "duration_tolerance_seconds": 0.2,
                    "black_frame_max_ratio": 1.01,  # disable black gate
                    "motion_min_flow": 0.5,  # require flow >= 0.5
                },
                width=1280,
                height=720,
                frame_count=80,
                fps=16,
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
        make_test_video_with_motion(
            short_mp4, n_frames=32, fps=16, size=(64, 64)
        )  # 2.0s

        workflow_template = _wan_template_for_tests()
        # Register both seeds (initial and retry) to handle retry loop
        wf_seed_0 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1001,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        comfy = FakeComfyClient(
            responses={
                workflow_signature(wf_seed_0): [short_mp4],
                workflow_signature(wf_seed_1): [short_mp4],
            }
        )

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace

        scene = SimpleNamespace(
            index=1,
            visual_prompt="x",
            keyframe_path=keyframe,
            video_path=None,
        )

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
                width=1280,
                height=720,
                frame_count=80,
                fps=16,
            )
        assert excinfo.value.retryable is True
        assert "duration" in excinfo.value.reason


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
            prompt="x",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1001,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
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

        scene = SimpleNamespace(
            index=1, visual_prompt="x", keyframe_path=keyframe, video_path=None
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
                "motion_min_flow": 0.0,
            },
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )

        assert report.success is True
        assert report.retry_used == 1
        assert len(comfy.calls) == 2  # initial + 1 retry


class TestGenerateVideoForSceneHalt:
    @pytest.mark.asyncio
    async def test_halt_on_second_content_fail(self, tmp_path: Path) -> None:
        """Both initial and retry produce black MP4. Should raise after 2 attempts."""
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
        # Both seeds map to the same black fixture.
        wf_seed_0 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1000,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
        )
        wf_seed_1 = inject_video(
            workflow_template,
            image_in="scene_001.png",
            prompt="x",
            seed=1001,
            output_prefix="scene_001_raw",
            width=1280,
            height=720,
            frame_count=80,
            fps=16,
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

        scene = SimpleNamespace(
            index=1,
            visual_prompt="x",
            keyframe_path=keyframe,
            video_path=None,
        )

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
                width=1280,
                height=720,
                frame_count=80,
                fps=16,
            )
        assert excinfo.value.retryable is True
        assert len(comfy.calls) == 2
        # Failed MP4 cleaned up.
        assert not (tmp_path / "clips" / "scene_001_raw.mp4").exists()

    @pytest.mark.asyncio
    async def test_infra_failure_not_retried(self, tmp_path: Path) -> None:
        """generate_image raises -> propagate as retryable=False, no retry."""
        from platinum.pipeline.video_generator import (
            VideoGenerationError,
            generate_video_for_scene,
        )

        keyframe = tmp_path / "scene_001.png"
        keyframe.write_bytes(b"fake_png")
        from types import SimpleNamespace

        scene = SimpleNamespace(
            index=1, visual_prompt="x", keyframe_path=keyframe, video_path=None
        )

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
                width=1280,
                height=720,
                frame_count=80,
                fps=16,
            )
        assert excinfo.value.retryable is False
        assert "ComfyUI returned 500" in excinfo.value.reason
        # Only one attempt, no retry.
        assert len(comfy.calls) == 1


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
        for scene in scenes:
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


class TestGenerateVideoPreconditions:
    @pytest.mark.asyncio
    async def test_missing_keyframe_path_halts_before_any_comfy_call(
        self, tmp_path: Path
    ) -> None:
        from platinum.pipeline.video_generator import (
            VideoGenerationError,
            generate_video,
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
            VideoGenerationError,
            generate_video,
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


def _wan_template_for_tests() -> dict:
    """Shared minimal Wan workflow template for the test module."""
    return {
        "_meta": {
            "role": {
                "image_in": "100",
                "prompt": "101",
                "seed": "102",
                "video_out": "103",
                "width": "104",
                "height": "104",
                "frame_count": "105",
                "fps": "106",
            }
        },
        "100": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "101": {"class_type": "WanT5TextEncode", "inputs": {"text": ""}},
        "102": {"class_type": "WanSampler", "inputs": {"seed": 0}},
        "103": {
            "class_type": "VHS_VideoCombine",
            "inputs": {"filename_prefix": "", "frame_rate": 0},
        },
        "104": {"class_type": "WanSampler", "inputs": {"width": 0, "height": 0}},
        "105": {"class_type": "WanLatentVideo", "inputs": {"length": 0}},
        "106": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 0}},
    }
