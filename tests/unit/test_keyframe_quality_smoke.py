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
