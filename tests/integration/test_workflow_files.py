"""Integration tests for shipped workflow JSON files."""
from __future__ import annotations

import json
from pathlib import Path

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
