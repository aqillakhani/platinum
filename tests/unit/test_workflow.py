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
