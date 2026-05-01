"""Integration tests: image_model aspect must match video_model aspect.

Catches the S8.18 verify bug where atmospheric_horror shipped video_model
960x544 (landscape) while image_model was 768x1344 (9:16 portrait), causing
WanVideoImageToVideoEncode to stretch portrait keyframes onto a landscape
canvas (vertical 0.40x squash, horizontal 1.25x stretch). The squash is
baked into output pixels — not fixable in post.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from platinum.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_TRACKS = [
    "atmospheric_horror",
    "folktales_world_myths",
    "childrens_fables",
    "scifi_concept",
    "slice_of_life",
]


def _orientation(width: int, height: int) -> str:
    if width < height:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


@pytest.mark.parametrize("track_id", ALL_TRACKS)
def test_track_video_orientation_matches_image_orientation(track_id: str) -> None:
    """If a track declares both image_model and video_model dimensions,
    they must share an orientation (portrait, landscape, or square).

    Cross-orientation pairings cause WanVideoImageToVideoEncode to stretch
    the keyframe — the failure mode the S8.18 verify run shipped to
    production at 960x544 over 768x1344 keyframes.
    """
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    image_model = track.get("image_model", {})
    video_model = track.get("video_model", {})

    iw, ih = image_model.get("width"), image_model.get("height")
    vw, vh = video_model.get("width"), video_model.get("height")

    if not (iw and ih and vw and vh):
        pytest.skip(f"{track_id}: video_model does not declare width/height yet")

    image_orient = _orientation(int(iw), int(ih))
    video_orient = _orientation(int(vw), int(vh))
    assert image_orient == video_orient, (
        f"{track_id}: image_model is {image_orient} ({iw}x{ih}) but "
        f"video_model is {video_orient} ({vw}x{vh}). "
        f"Wan2.2 I2V will stretch portrait keyframes onto a landscape canvas "
        f"(or vice versa). Set video_model dimensions to match image_model "
        f"orientation, divisible by 16 for Wan VAE 8x + DiT patch 2x."
    )


@pytest.mark.parametrize("track_id", ALL_TRACKS)
def test_track_video_aspect_within_tolerance_of_image_aspect(track_id: str) -> None:
    """Aspect ratios must match within 10% even if orientations agree.

    A portrait keyframe at 9:16 (0.5625) sent into a portrait video at
    1:2 (0.5) still gets stretched, just less catastrophically. 10% is the
    largest divergence we accept before the visible distortion starts.
    """
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    image_model = track.get("image_model", {})
    video_model = track.get("video_model", {})

    iw, ih = image_model.get("width"), image_model.get("height")
    vw, vh = video_model.get("width"), video_model.get("height")

    if not (iw and ih and vw and vh):
        pytest.skip(f"{track_id}: video_model does not declare width/height yet")

    image_aspect = float(iw) / float(ih)
    video_aspect = float(vw) / float(vh)
    rel_diff = abs(image_aspect - video_aspect) / image_aspect
    assert rel_diff <= 0.10, (
        f"{track_id}: image_model aspect {image_aspect:.4f} ({iw}x{ih}) "
        f"vs video_model aspect {video_aspect:.4f} ({vw}x{vh}); "
        f"relative difference {rel_diff:.1%} exceeds 10% tolerance. "
        f"Match the video_model aspect to the image_model aspect to "
        f"prevent latent stretching in WanVideoImageToVideoEncode."
    )


@pytest.mark.parametrize("track_id", ALL_TRACKS)
def test_track_video_dimensions_divisible_by_16(track_id: str) -> None:
    """Wan VAE patch math (8x VAE + DiT patch 2x = 16x downsampling) requires
    width and height divisible by 16. Common failure: 540 -> 33.75 latent
    rows -> sampler errors with cryptic "tensor a (67) vs tensor b (66) at
    non-singleton dimension 3" (S8.18 discovery).
    """
    cfg = Config(root=REPO_ROOT)
    track = cfg.track(track_id)
    video_model = track.get("video_model", {})
    vw, vh = video_model.get("width"), video_model.get("height")

    if not (vw and vh):
        pytest.skip(f"{track_id}: video_model does not declare width/height yet")

    assert int(vw) % 16 == 0, f"{track_id}: video_model.width={vw} not divisible by 16"
    assert int(vh) % 16 == 0, f"{track_id}: video_model.height={vh} not divisible by 16"
