"""Tests for scripts/keyframe_quality_smoke.py.

Tests cover argparse + the synthetic-Scene construction. The async run()
function is NOT tested end-to-end here (that requires a live HttpComfyClient
+ RemoteAestheticScorer); coverage for that path is the live runbook smoke
in Phase 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_smoke_driver_argparse_requires_prompt_and_label(monkeypatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import keyframe_quality_smoke

    monkeypatch.setattr(sys, "argv", ["smoke"])
    with pytest.raises(SystemExit):
        keyframe_quality_smoke._main()


def test_smoke_driver_synthetic_scene_index_is_deterministic_per_label() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import keyframe_quality_smoke

    s1 = keyframe_quality_smoke._scene_index_for_label("tuscan-vineyard")
    s2 = keyframe_quality_smoke._scene_index_for_label("tuscan-vineyard")
    s3 = keyframe_quality_smoke._scene_index_for_label("alpine-peak")
    assert s1 == s2                                      # deterministic
    assert s1 != s3                                      # different labels -> different indices
    assert 0 <= s1 < 10000


def test_smoke_driver_loads_rebuilt_workflow() -> None:
    """Smoke driver imports succeed and workflow load returns the rebuilt JSON."""
    from platinum.utils.workflow import load_workflow

    repo_root = Path(__file__).resolve().parents[2]
    wf = load_workflow("flux_dev_keyframe", config_dir=repo_root / "config")
    # Sanity: rebuilt workflow has the new roles (S6.3 contract).
    roles = wf["_meta"]["role"]
    assert "model_sampling_flux" in roles
    assert "flux_guidance" in roles
    # Sanity: KSampler.cfg=1.0 (S6.3) not 3.5 (pre-S6.3)
    assert wf[roles["sampler"]]["inputs"]["cfg"] == 1.0
