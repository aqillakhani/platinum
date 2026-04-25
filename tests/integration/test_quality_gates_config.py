"""Integration tests: quality_gates block round-trips through Config.track()."""

from __future__ import annotations

from pathlib import Path

from platinum.config import Config

EXPECTED_KEYS = {
    "aesthetic_min_score",
    "black_frame_max_ratio",
    "luminance_threshold",
    "motion_min_flow_magnitude",
    "audio_target_lufs",
    "audio_lufs_tolerance_db",
    "duration_tolerance_seconds",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_atmospheric_horror_yaml_loads_quality_gates() -> None:
    cfg = Config(root=REPO_ROOT)
    track = cfg.track("atmospheric_horror")
    assert "quality_gates" in track, "atmospheric_horror.yaml is missing the quality_gates block"
    gates = track["quality_gates"]
    missing = EXPECTED_KEYS - set(gates.keys())
    assert not missing, f"atmospheric_horror.quality_gates missing keys: {missing}"
    for key in EXPECTED_KEYS:
        assert isinstance(gates[key], (int, float)), f"{key} must be numeric"
    assert 0.0 <= gates["aesthetic_min_score"] <= 10.0
    assert 0.0 <= gates["black_frame_max_ratio"] <= 1.0
    assert gates["audio_target_lufs"] < 0
