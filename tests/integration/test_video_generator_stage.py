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
