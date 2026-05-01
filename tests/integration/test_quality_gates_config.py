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
    "track_id, expected_clip",
    [
        # atmospheric_horror lowered to 0.0 in S7.1 verify run -- 19th-century
        # chiaroscuro is underrepresented in CLIP's training distribution and
        # scored below the 0.20 floor even on viable on-prompt renders.
        # Recalibration deferred to S7.2.
        ("atmospheric_horror", 0.0),
        ("folktales_world_myths", 0.20),
        ("childrens_fables", 0.20),
        ("scifi_concept", 0.20),
        ("slice_of_life", 0.20),
    ],
)
def test_track_yaml_image_model_clip_min_similarity(
    track_id: str, expected_clip: float
) -> None:
    """S7.1.A3.4: each track ships clip_min_similarity for the content gate.

    0.20 is the post-S7-retro baseline for the LAION ViT-L-14 backbone:
    high enough to reject the "moody-but-wrong-content" failure mode that
    dominated the Cask 4/16 approval rate, low enough to keep first-shot
    yield viable on a 3-candidate budget. atmospheric_horror is the
    exception (verify-run calibration -- see parametrize comment).
    """
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    image_model = track.get("image_model", {})
    threshold = image_model.get("clip_min_similarity")
    assert threshold == expected_clip, (
        f"{track_id}.image_model.clip_min_similarity must be "
        f"{expected_clip} (got {threshold!r})"
    )


@pytest.mark.parametrize(
    "track_id, expected_gate, expected_min_score",
    [
        ("atmospheric_horror", "claude", 6),  # S7 Phase 2 the offending track; gate ON
        ("folktales_world_myths", "off", 7),  # other tracks default off until tuned
        ("childrens_fables", "off", 7),
        ("scifi_concept", "off", 7),
        ("slice_of_life", "off", 7),
    ],
)
def test_track_yaml_content_gate_fields(
    track_id: str, expected_gate: str, expected_min_score: int
) -> None:
    """S7.1.A4.4: each track declares its Claude vision content_gate config.

    atmospheric_horror is the only track currently turning the gate ON
    (it's the track whose 4/16 approval rate motivated S7.1). The other
    four tracks ship "off" until we have empirical data to tune them;
    keeping the keys present (rather than absent) avoids None/missing
    branches in keyframe_generator.
    """
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    gates = track.get("quality_gates", {})
    assert gates.get("content_gate") == expected_gate, (
        f"{track_id}.quality_gates.content_gate must be {expected_gate!r}"
    )
    assert gates.get("content_gate_min_score") == expected_min_score, (
        f"{track_id}.quality_gates.content_gate_min_score must be {expected_min_score}"
    )


@pytest.mark.parametrize(
    "track,expected_subject",
    [
        # atmospheric_horror lowered to 0.0 in S7.1 verify run -- chiaroscuro
        # Cask renders score 0.0001-0.002 even though eye-check confirms
        # viable subjects. Recalibration deferred to S7.2 (per-scene
        # brightness-aware threshold or face-detection replacement).
        ("atmospheric_horror", 0.0),
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


@pytest.mark.parametrize(
    "track,expected_motion_floor",
    [
        ("atmospheric_horror", 0.0),
        ("folktales_world_myths", 0.5),
        ("childrens_fables", 0.7),
        ("scifi_concept", 0.5),
        ("slice_of_life", 0.6),
    ],
)
def test_video_gates_block_present_per_track(
    track: str, expected_motion_floor: float
) -> None:
    """Every track YAML has video_gates with all 4 keys; track-tuned motion floor."""
    import yaml

    cfg = yaml.safe_load(
        (Path("config/tracks") / f"{track}.yaml").read_text(encoding="utf-8")
    )
    quality_gates = cfg["track"]["quality_gates"]
    assert "video_gates" in quality_gates, f"{track}: video_gates block missing"
    vg = quality_gates["video_gates"]
    assert vg["duration_target_seconds"] == 5.0
    assert vg["duration_tolerance_seconds"] == 0.2
    assert vg["black_frame_max_ratio"] == 0.05
    assert vg["motion_min_flow"] == expected_motion_floor
