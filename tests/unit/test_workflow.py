"""Tests for utils/workflow.py."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest


def _minimal_workflow() -> dict:
    return {
        "_meta": {
            "title": "minimal",
            "role": {
                "positive_prompt": "3",
                "negative_prompt": "4",
                "empty_latent": "5",
                "sampler": "6",
                "save_image": "8",
            },
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "OLD_POS", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "OLD_NEG", "clip": ["2", 0]}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0, "steps": 20, "cfg": 3.5,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "OLD", "images": ["7", 0]}},
    }


def test_inject_swaps_tagged_node_inputs() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    out = inject(
        wf, prompt="cinematic dread", negative_prompt="cartoon",
        seed=4242, width=1024, height=1024, output_prefix="scene_001",
    )
    assert out["3"]["inputs"]["text"] == "cinematic dread"
    assert out["4"]["inputs"]["text"] == "cartoon"
    assert out["5"]["inputs"]["width"] == 1024
    assert out["5"]["inputs"]["height"] == 1024
    assert out["6"]["inputs"]["seed"] == 4242
    assert out["8"]["inputs"]["filename_prefix"] == "scene_001"


def test_inject_does_not_mutate_input() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    snapshot = copy.deepcopy(wf)
    _ = inject(
        wf, prompt="x", negative_prompt="y", seed=1,
        width=512, height=512, output_prefix="z",
    )
    assert wf == snapshot


def test_inject_leaves_untagged_nodes_alone() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    wf["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}}
    wf["2"] = {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors"}}
    out = inject(
        wf, prompt="a", negative_prompt="b", seed=7,
        width=1024, height=1024, output_prefix="p",
    )
    assert out["1"] == wf["1"]
    assert out["2"] == wf["2"]


def test_inject_raises_keyerror_on_missing_role() -> None:
    from platinum.utils.workflow import inject

    wf = _minimal_workflow()
    del wf["_meta"]["role"]["sampler"]
    with pytest.raises(KeyError) as exc:
        inject(
            wf, prompt="x", negative_prompt="y", seed=1,
            width=512, height=512, output_prefix="z",
        )
    assert "sampler" in str(exc.value)


def test_load_workflow_reads_named_file(tmp_path: Path) -> None:
    """load_workflow takes (name, *, config_dir) -- config_dir injected for testability."""
    from platinum.utils.workflow import load_workflow

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    (workflows_dir / "demo.json").write_text(
        '{"_meta": {"role": {}}, "1": {"class_type": "X", "inputs": {}}}',
        encoding="utf-8",
    )
    out = load_workflow("demo", config_dir=tmp_path)
    assert out["1"]["class_type"] == "X"


def test_load_workflow_raises_on_missing_file(tmp_path: Path) -> None:
    from platinum.utils.workflow import load_workflow

    (tmp_path / "workflows").mkdir()
    with pytest.raises(FileNotFoundError):
        load_workflow("nope", config_dir=tmp_path)


def test_flux_dev_keyframe_workflow_loads_and_injects() -> None:
    """The shipped flux_dev_keyframe.json must round-trip through inject without errors."""
    from platinum.utils.workflow import inject, load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    # All required roles must be present and resolve to actual node ids.
    roles = wf["_meta"]["role"]
    assert {"positive_prompt", "negative_prompt", "empty_latent", "sampler", "save_image"} <= set(
        roles.keys()
    )
    for role_name, node_id in roles.items():
        assert node_id in wf, f"role {role_name!r} points at missing node {node_id!r}"
    out = inject(
        wf,
        prompt="a candle in a dark hallway",
        negative_prompt="bright daylight, neon",
        seed=12345,
        width=1024, height=1024,
        output_prefix="scene_001_candidate_0",
    )
    pos_id = roles["positive_prompt"]
    neg_id = roles["negative_prompt"]
    latent_id = roles["empty_latent"]
    sampler_id = roles["sampler"]
    save_id = roles["save_image"]
    assert out[pos_id]["inputs"]["text"] == "a candle in a dark hallway"
    assert out[neg_id]["inputs"]["text"] == "bright daylight, neon"
    assert out[latent_id]["inputs"]["width"] == 1024
    assert out[latent_id]["inputs"]["height"] == 1024
    assert out[sampler_id]["inputs"]["seed"] == 12345
    assert out[save_id]["inputs"]["filename_prefix"] == "scene_001_candidate_0"


def test_flux_dev_workflow_uses_dpmpp_2m_karras_60steps() -> None:
    """The Flux Dev keyframe workflow uses the BFL-recommended sampler combo.

    Why these values:
      sampler_name=dpmpp_2m + scheduler=karras -- community-validated for
        Flux Dev; produces sharper detail than ComfyUI's euler/simple default.
      steps=60 -- bumped from 30 in S6.3 Phase 2 (commit 11113d0); selected
        LAION moved 5.7-6.2 -> 6.2-6.5 on Cask 8/16 with cleaner subject
        definition. +25s per candidate on A6000, still within budget.
      cfg=1.0 -- reduced from 3.5 in S6.3; guidance moved to FluxGuidance node.
    """
    from platinum.utils.workflow import load_workflow

    wf = load_workflow(
        "flux_dev_keyframe",
        config_dir=Path(__file__).resolve().parents[2] / "config",
    )
    # KSampler node id is "6" per the existing _meta.role layout.
    ksampler_inputs = wf["6"]["inputs"]
    assert ksampler_inputs["sampler_name"] == "dpmpp_2m"
    assert ksampler_inputs["scheduler"] == "karras"
    assert ksampler_inputs["steps"] == 60
    assert ksampler_inputs["cfg"] == 1.0


def test_inject_sets_model_sampling_flux_width_height() -> None:
    """When model_sampling_flux role is present, inject() sets width/height on it."""
    from platinum.utils.workflow import inject

    workflow = {
        "_meta": {
            "role": {
                "positive_prompt": "3",
                "negative_prompt": "4",
                "empty_latent": "5",
                "sampler": "6",
                "save_image": "8",
                "model_sampling_flux": "10",
            }
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 0, "height": 0, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"seed": 0}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": ""}},
        "10": {"class_type": "ModelSamplingFlux",
               "inputs": {"max_shift": 1.15, "base_shift": 0.5,
                          "width": 0, "height": 0, "model": ["1", 0]}},
    }
    out = inject(workflow, prompt="x", negative_prompt="", seed=42,
                 width=1024, height=1024, output_prefix="test")
    assert out["10"]["inputs"]["width"] == 1024
    assert out["10"]["inputs"]["height"] == 1024


def test_inject_works_without_model_sampling_flux_role() -> None:
    """Backwards compat: workflows without the role still inject cleanly."""
    from platinum.utils.workflow import inject

    workflow = {
        "_meta": {"role": {"positive_prompt": "3", "negative_prompt": "4",
                           "empty_latent": "5", "sampler": "6", "save_image": "8"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 0, "height": 0, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"seed": 0}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": ""}},
    }
    out = inject(workflow, prompt="x", negative_prompt="", seed=42)
    assert out["5"]["inputs"]["width"] == 1024


def test_rebuilt_flux_workflow_has_required_roles_and_cfg_1() -> None:
    """Post-S6.3, the workflow has FluxGuidance + ModelSamplingFlux, cfg=1.0."""
    from pathlib import Path

    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    roles = wf.get("_meta", {}).get("role", {})
    # Existing roles preserved
    for role in ("positive_prompt", "negative_prompt", "empty_latent",
                 "sampler", "save_image"):
        assert role in roles, f"role {role!r} missing"
    # New roles registered
    assert roles["model_sampling_flux"] == "10"
    assert roles["flux_guidance"] == "11"

    # KSampler.cfg = 1.0 (was 3.5 in S6.2)
    sampler_id = roles["sampler"]
    assert wf[sampler_id]["inputs"]["cfg"] == 1.0
    # KSampler.model now points at IPAdapterFaceIDApply output (was ["10", 0]
    # pre-S7.1.B1.1; B1.1 inserted IPAdapterFaceIDApply between ModelSamplingFlux
    # and KSampler so face refs can condition the model chain).
    assert wf[sampler_id]["inputs"]["model"] == ["14", 0]
    # KSampler.positive now sources from ControlNetApplyAdvanced_pose (S7.1.B1.3),
    # which itself receives positive from ControlNetApplyAdvanced_depth (S7.1.B1.2).
    # The two ControlNet apply nodes chain between FluxGuidance and KSampler so
    # depth + pose maps can both condition the positive branch.
    assert wf[sampler_id]["inputs"]["positive"] == ["20", 0]

    # Node 10: ModelSamplingFlux exists with proper shifts
    assert wf["10"]["class_type"] == "ModelSamplingFlux"
    assert wf["10"]["inputs"]["max_shift"] == 1.15
    assert wf["10"]["inputs"]["base_shift"] == 0.5
    assert wf["10"]["inputs"]["model"] == ["1", 0]

    # Node 11: FluxGuidance exists with guidance=3.5
    assert wf["11"]["class_type"] == "FluxGuidance"
    assert wf["11"]["inputs"]["guidance"] == 3.5
    assert wf["11"]["inputs"]["conditioning"] == ["3", 0]


def test_flux_dev_keyframe_workflow_default_aspect_is_9_16() -> None:
    """S7.1.A2: shipped JSON defaults to 768x1344 (9:16 portrait).

    The S7 retro showed Cask 16-scene approval gap correlated with the
    1024x1024 square losing cinematic context on Instagram reels / YouTube
    Shorts. The default aspect now matches the target distribution surface;
    callers can still pass width/height to inject() to override.
    """
    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    assert wf["5"]["inputs"]["width"] == 768
    assert wf["5"]["inputs"]["height"] == 1344
    assert wf["10"]["inputs"]["width"] == 768
    assert wf["10"]["inputs"]["height"] == 1344


def test_inject_against_rebuilt_workflow_produces_valid_wiring() -> None:
    """End-to-end inject on the actual config/workflows JSON: width/height
    flow into both EmptyLatentImage AND ModelSamplingFlux."""
    from platinum.utils.workflow import inject, load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    out = inject(wf, prompt="hello world", negative_prompt="ugly",
                 seed=12345, width=1024, height=1024, output_prefix="test")

    # Positive/negative text injected into nodes 3/4
    assert out["3"]["inputs"]["text"] == "hello world"
    assert out["4"]["inputs"]["text"] == "ugly"
    # EmptyLatent dimensions
    assert out["5"]["inputs"]["width"] == 1024
    assert out["5"]["inputs"]["height"] == 1024
    # ModelSamplingFlux dimensions match
    assert out["10"]["inputs"]["width"] == 1024
    assert out["10"]["inputs"]["height"] == 1024
    # Seed + filename
    assert out["6"]["inputs"]["seed"] == 12345
    assert out["8"]["inputs"]["filename_prefix"] == "test"
    # FluxGuidance untouched (guidance is static template constant)
    assert out["11"]["inputs"]["guidance"] == 3.5


def test_workflow_has_ipadapter_nodes_after_b1_1() -> None:
    """S7.1.B1.1: shipped JSON gains IPAdapterFaceID nodes 12, 13, 14.

    Adds three nodes for cross-scene face continuity via IP-Adapter:
      12: IPAdapterModelLoader (loads FLUX.1-Redux-dev IP-Adapter weights)
      13: LoadImage (face reference; placeholder, swapped per call by inject())
      14: IPAdapterFaceIDApply (combines model + ipadapter + face image)

    Rewires the model chain so KSampler.model sources from node 14:
      UNETLoader(1) -> ModelSamplingFlux(10) -> IPAdapterFaceIDApply(14) -> KSampler(6).model

    _meta.role tags ipadapter_loader/face_ref_image/ipadapter_apply give
    inject() stable handles regardless of node-id renumbering.
    """
    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # New nodes exist with expected class types
    assert wf["12"]["class_type"] == "IPAdapterModelLoader"
    assert wf["13"]["class_type"] == "LoadImage"
    assert wf["14"]["class_type"] == "IPAdapterFaceIDApply"

    # _meta.role tags map to the new nodes
    roles = wf["_meta"]["role"]
    assert roles["ipadapter_loader"] == "12"
    assert roles["face_ref_image"] == "13"
    assert roles["ipadapter_apply"] == "14"

    # Wiring into IPAdapterFaceIDApply (node 14):
    #   model from ModelSamplingFlux (node 10)
    #   ipadapter from IPAdapterModelLoader (node 12)
    #   image from LoadImage (node 13)
    apply_inputs = wf["14"]["inputs"]
    assert apply_inputs["model"] == ["10", 0]
    assert apply_inputs["ipadapter"] == ["12", 0]
    assert apply_inputs["image"] == ["13", 0]

    # KSampler.model now sources from node 14 (was ["10", 0] pre-B1.1)
    sampler_id = roles["sampler"]
    assert wf[sampler_id]["inputs"]["model"] == ["14", 0]


def test_workflow_has_controlnet_depth_nodes_after_b1_2() -> None:
    """S7.1.B1.2: shipped JSON gains ControlNet Depth nodes 15, 16, 17.

    15: ControlNetLoader (loads Flux-Depth ControlNet weights)
    16: LoadImage (depth ref; placeholder, swapped per call by inject())
    17: ControlNetApplyAdvanced (applies depth conditioning to positive)

    Rewires the positive conditioning chain so KSampler.positive flows
    through the depth-apply node:
      CLIPTextEncode(3) -> FluxGuidance(11) -> ControlNetApplyAdvanced_depth(17)
                                                    -> KSampler.positive(6)

    The Advanced variant takes positive + negative and returns both, but we
    only consume slot 0 (positive). KSampler.negative stays direct to
    CLIPTextEncode(4) -- the ControlNet conditioning shapes positive only.
    """
    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # New nodes exist with expected class types
    assert wf["15"]["class_type"] == "ControlNetLoader"
    assert wf["16"]["class_type"] == "LoadImage"
    assert wf["17"]["class_type"] == "ControlNetApplyAdvanced"

    # _meta.role tags map to the new nodes
    roles = wf["_meta"]["role"]
    assert roles["controlnet_depth_loader"] == "15"
    assert roles["depth_ref_image"] == "16"
    assert roles["controlnet_depth_apply"] == "17"

    # Wiring into ControlNetApplyAdvanced_depth (node 17):
    #   positive from FluxGuidance (node 11)
    #   negative passes through from CLIPTextEncode neg (node 4)
    #   control_net from ControlNetLoader_depth (node 15)
    #   image from LoadImage_depth (node 16)
    # These four are durable: node 17 always sits between FluxGuidance and
    # whatever comes next on the positive branch (later nodes will chain off
    # 17.slot-0). KSampler.positive endpoint is asserted by the latest layer
    # test (e.g. B1.3 pins it to node 20) since that endpoint shifts as new
    # apply nodes land.
    apply_inputs = wf["17"]["inputs"]
    assert apply_inputs["positive"] == ["11", 0]
    assert apply_inputs["negative"] == ["4", 0]
    assert apply_inputs["control_net"] == ["15", 0]
    assert apply_inputs["image"] == ["16", 0]
    # KSampler.negative stays direct from CLIPTextEncode neg (durable).
    sampler_id = roles["sampler"]
    assert wf[sampler_id]["inputs"]["negative"] == ["4", 0]


def test_workflow_has_controlnet_pose_nodes_after_b1_3() -> None:
    """S7.1.B1.3: shipped JSON gains ControlNet OpenPose nodes 18, 19, 20.

    18: ControlNetLoader (loads Flux-Pose ControlNet weights)
    19: LoadImage (pose ref; placeholder, swapped per call by inject())
    20: ControlNetApplyAdvanced (applies pose conditioning to positive)

    Final positive conditioning chain after B1.3:
      CLIPTextEncode(3) -> FluxGuidance(11) -> ControlNetApplyAdvanced_depth(17)
                                                  -> ControlNetApplyAdvanced_pose(20)
                                                  -> KSampler.positive(6)

    KSampler.negative stays direct from CLIPTextEncode(4) per the design's
    "no negative-side ControlNet" rule -- the apply nodes' slot-1 outputs
    (modified negative) are unused.
    """
    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")

    # New nodes exist with expected class types
    assert wf["18"]["class_type"] == "ControlNetLoader"
    assert wf["19"]["class_type"] == "LoadImage"
    assert wf["20"]["class_type"] == "ControlNetApplyAdvanced"

    # _meta.role tags map to the new nodes
    roles = wf["_meta"]["role"]
    assert roles["controlnet_pose_loader"] == "18"
    assert roles["pose_ref_image"] == "19"
    assert roles["controlnet_pose_apply"] == "20"

    # Wiring into ControlNetApplyAdvanced_pose (node 20):
    #   positive from ControlNetApplyAdvanced_depth (node 17)
    #   negative still direct from CLIPTextEncode neg (node 4)
    #   control_net from ControlNetLoader_pose (node 18)
    #   image from LoadImage_pose (node 19)
    apply_inputs = wf["20"]["inputs"]
    assert apply_inputs["positive"] == ["17", 0]
    assert apply_inputs["negative"] == ["4", 0]
    assert apply_inputs["control_net"] == ["18", 0]
    assert apply_inputs["image"] == ["19", 0]

    # KSampler.positive now sources from node 20 (was ["17", 0] pre-B1.3)
    sampler_id = roles["sampler"]
    assert wf[sampler_id]["inputs"]["positive"] == ["20", 0]
    # KSampler.negative still direct from CLIPTextEncode neg
    assert wf[sampler_id]["inputs"]["negative"] == ["4", 0]
