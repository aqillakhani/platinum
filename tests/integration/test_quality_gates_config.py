"""Integration tests: quality_gates block round-trips through Config.track()."""

from __future__ import annotations

from pathlib import Path

import pytest

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


ALL_TRACKS = [
    "atmospheric_horror",
    "folktales_world_myths",
    "childrens_fables",
    "scifi_concept",
    "slice_of_life",
]


@pytest.mark.parametrize("track_id", ALL_TRACKS)
def test_all_tracks_have_quality_gates(track_id: str) -> None:
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    assert "quality_gates" in track, f"{track_id} missing quality_gates"
    gates = track["quality_gates"]
    missing = EXPECTED_KEYS - set(gates.keys())
    assert not missing, f"{track_id}.quality_gates missing keys: {missing}"


@pytest.mark.parametrize(
    "track_name, expected_floor",
    [
        ("atmospheric_horror", 5.0),
        ("folktales_world_myths", 60.0),
        ("childrens_fables", 80.0),
        ("scifi_concept", 40.0),
        ("slice_of_life", 60.0),
    ],
)
def test_track_yaml_carries_brightness_floor(track_name: str, expected_floor: float) -> None:
    """Each track YAML must declare its brightness floor under quality_gates.

    Floors are calibrated per-track:
      atmospheric_horror -- chiaroscuro tolerated; recalibrated S6.3 Phase 2
        from 20 -> 5 (was rejecting legitimate 5-10 mean_rgb chiaroscuro).
      folktales / slice_of_life -- daylight typical; 60 catches degenerate output.
      scifi_concept -- mixed (cosmic darkness + bright tech); 40 splits the diff.
      childrens_fables -- bright/saturated; 80 enforces the cheerful palette.
    """
    cfg = Config(root=REPO_ROOT)
    track_cfg = cfg.track(track_name)
    qg = track_cfg.get("quality_gates", {})
    assert qg.get("brightness_floor_mean_rgb") == expected_floor


@pytest.mark.parametrize("track_id", ALL_TRACKS)
def test_track_yaml_image_model_aspect_fields(track_id: str) -> None:
    """S7.1.A2: each track ships 9:16 portrait dimensions for keyframe gen.

    Cinematic aspect drives the keyframe latent (768x1344) and downstream
    video crop (Wan 2.2 inherits the dimensions from the keyframe). The
    aspect_ratio string is informational and consumed by the S8 video
    pipeline.
    """
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    image_model = track.get("image_model", {})
    assert image_model.get("width") == 768, f"{track_id}.image_model.width must be 768"
    assert image_model.get("height") == 1344, f"{track_id}.image_model.height must be 1344"
    assert image_model.get("aspect_ratio") == "9:16", \
        f"{track_id}.image_model.aspect_ratio must be '9:16'"


@pytest.mark.parametrize(
    "track,expected_subject",
    [
        ("atmospheric_horror", 0.005),
        ("folktales_world_myths", 0.030),
        ("childrens_fables", 0.040),
        ("scifi_concept", 0.030),
        ("slice_of_life", 0.030),
    ],
)
def test_track_yaml_subject_min_edge_density(track: str, expected_subject: float) -> None:
    """Each track YAML's quality_gates carries the documented subject_min_edge_density."""
    cfg = Config(root=REPO_ROOT)
    track_cfg = cfg.track(track)
    gates = track_cfg.get("quality_gates", {})
    assert "subject_min_edge_density" in gates, \
        f"track {track} missing subject_min_edge_density"
    assert gates["subject_min_edge_density"] == expected_subject, \
        f"track {track}: expected {expected_subject}, got {gates['subject_min_edge_density']}"


def test_check_audio_levels_round_trips_with_real_ffmpeg(tmp_path: Path) -> None:
    """Generate a tone, measure with loudnorm, verify check_audio_levels matches."""
    from platinum.utils.validate import _measure_lufs, check_audio_levels
    from tests._fixtures import make_test_audio

    audio = tmp_path / "tone.wav"
    make_test_audio(audio, seconds=3.0, freq_hz=1000.0, amplitude=0.25)
    measured = _measure_lufs(audio)
    assert measured > -50.0  # not silent
    assert measured < 0.0    # not clipping above 0 dB FS
    r = check_audio_levels(audio, target_lufs=measured, tolerance_db=0.5)
    assert r.passed
    assert "passed" in r.reason
