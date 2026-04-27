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
    # KSampler.model now points at ModelSamplingFlux output
    assert wf[sampler_id]["inputs"]["model"] == ["10", 0]
    # KSampler.positive now points at FluxGuidance output
    assert wf[sampler_id]["inputs"]["positive"] == ["11", 0]

    # Node 10: ModelSamplingFlux exists with proper shifts
    assert wf["10"]["class_type"] == "ModelSamplingFlux"
    assert wf["10"]["inputs"]["max_shift"] == 1.15
    assert wf["10"]["inputs"]["base_shift"] == 0.5
    assert wf["10"]["inputs"]["model"] == ["1", 0]

    # Node 11: FluxGuidance exists with guidance=3.5
    assert wf["11"]["class_type"] == "FluxGuidance"
    assert wf["11"]["inputs"]["guidance"] == 3.5
    assert wf["11"]["inputs"]["conditioning"] == ["3", 0]


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
