"""Integration tests for shipped workflow JSON files."""
from __future__ import annotations

import json
from pathlib import Path

from platinum.utils.workflow import inject_video


def _as_id_list(role_val: object) -> list[str]:
    """Normalize a role value to a list of node IDs (str -> [str], list passes through)."""
    return [role_val] if isinstance(role_val, str) else list(role_val)  # type: ignore[arg-type]


class TestWan22I2VWorkflow:
    def test_workflow_loads_and_has_required_roles(self) -> None:
        path = Path("config/workflows/wan22_i2v.json")
        assert path.exists(), f"missing workflow file: {path}"
        wf = json.loads(path.read_text(encoding="utf-8"))
        roles = wf.get("_meta", {}).get("role", {})
        for required in ("image_in", "prompt", "seed", "video_out"):
            assert required in roles, f"missing required role: {required}"
            for node_id in _as_id_list(roles[required]):
                assert node_id in wf, (
                    f"role {required} -> node {node_id} not present in workflow"
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
        # Prompt-target node uses WanVideoTextEncode's positive_prompt input.
        assert out[roles["prompt"]]["inputs"]["positive_prompt"] == "probe"
        # Seed role can be either str (single sampler) or list (MoE chain);
        # inject_video writes the same value to every listed sampler node.
        for sid in _as_id_list(roles["seed"]):
            assert out[sid]["inputs"]["seed"] == 42
        assert out[roles["video_out"]]["inputs"]["filename_prefix"] == "scene_000_raw"

    def test_sampler_tuning_post_s8_A_2(self) -> None:
        """S8.A.2: cfg dropped 6.0 -> 2.5, total steps dropped 30 -> 10
        (5 HIGH + 5 LOW samples handoff). Pin so a casual edit can't
        drift back into the over-constrained regime that produced
        narrative-incoherent motion in the S8.18 verify run.
        """
        path = Path("config/workflows/wan22_i2v.json")
        wf = json.loads(path.read_text(encoding="utf-8"))
        seed_nodes = _as_id_list(wf["_meta"]["role"]["seed"])
        assert len(seed_nodes) == 2, "expected MoE pair (HIGH, LOW samplers)"
        high, low = seed_nodes
        assert wf[high]["inputs"]["steps"] == 10
        assert wf[high]["inputs"]["cfg"] == 2.5
        assert wf[high]["inputs"]["end_step"] == 5
        assert wf[low]["inputs"]["steps"] == 10
        assert wf[low]["inputs"]["cfg"] == 2.5
        assert wf[low]["inputs"]["start_step"] == 5

    def test_negative_prompt_includes_coherence_terms(self) -> None:
        """S8.A.2: existing negative prompt was render-quality only
        (blurry, distorted, watermark, ...). Wan 2.2 verify produced
        action duplication + reverse motion + object multiplication;
        community guidance is to add explicit narrative-coherence
        negatives. Pin a few canonical terms so a future regression
        is loud.
        """
        path = Path("config/workflows/wan22_i2v.json")
        wf = json.loads(path.read_text(encoding="utf-8"))
        prompt_node = wf["_meta"]["role"]["prompt"]
        neg = wf[prompt_node]["inputs"]["negative_prompt"].lower()
        for term in (
            "duplicated action",
            "reversed motion",
            "morphing",
            "looping gesture",
            "extra objects",
        ):
            assert term in neg, f"missing coherence negative-prompt term: {term!r}"
