"""Unit tests for utils/prompts.py."""

from __future__ import annotations

from pathlib import Path


def test_render_template_substitutes(tmp_path: Path) -> None:
    from platinum.utils.prompts import render_template
    (tmp_path / "atmospheric_horror").mkdir()
    (tmp_path / "atmospheric_horror" / "hello.j2").write_text(
        "Hello, {{ name }}!", encoding="utf-8"
    )
    out = render_template(
        prompts_dir=tmp_path,
        track="atmospheric_horror",
        name="hello.j2",
        context={"name": "world"},
    )
    assert out == "Hello, world!"


def test_render_template_missing_raises_clear(tmp_path: Path) -> None:
    import pytest

    from platinum.utils.prompts import render_template
    with pytest.raises(FileNotFoundError, match="missing.j2"):
        render_template(
            prompts_dir=tmp_path, track="atmospheric_horror",
            name="missing.j2", context={},
        )


def test_system_template_includes_voice_and_aesthetic() -> None:
    """The shipped system.j2 must render the track config's voice direction
    and visual aesthetic."""
    import yaml

    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]
    track_yaml = repo_root / "config" / "tracks" / "atmospheric_horror.yaml"
    track_cfg = yaml.safe_load(track_yaml.read_text(encoding="utf-8"))["track"]

    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="system.j2",
        context={"track": track_cfg},
    )
    assert track_cfg["voice"]["direction"] in out
    assert track_cfg["visual"]["aesthetic"] in out
    assert "[whisper]" in out
    assert "[pause]" in out


def test_adapt_template_renders_with_source_and_target() -> None:
    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]
    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="adapt.j2",
        context={
            "title": "The Cask of Amontillado",
            "author": "Edgar Allan Poe",
            "raw_text": "The thousand injuries of Fortunato I had borne...",
            "target_seconds": 600,
            "pace_wpm": 130,
        },
    )
    assert "Cask of Amontillado" in out
    assert "Edgar Allan Poe" in out
    assert "600" in out
    assert "130" in out
    assert "thousand injuries" in out


def test_breakdown_template_includes_target_and_optional_feedback() -> None:
    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]

    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="breakdown.j2",
        context={
            "narration_script": "It was a dark and stormy night.",
            "target_seconds": 600,
            "pace_wpm": 130,
            "tolerance_seconds": 30,
            "deviation_feedback": "",
            "music_moods": ["ambient_drone", "slow_strings_dread"],
        },
    )
    assert "600" in out
    assert "ambient_drone" in out
    assert "deviation_feedback" not in out  # only the rendered string

    out2 = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="breakdown.j2",
        context={
            "narration_script": "S",
            "target_seconds": 600,
            "pace_wpm": 130,
            "tolerance_seconds": 30,
            "deviation_feedback": "Previous breakdown totalled 540s; lengthen.",
            "music_moods": ["ambient_drone"],
        },
    )
    assert "Previous breakdown totalled 540s" in out2


def test_visual_prompts_template_includes_aesthetic_and_scenes() -> None:
    from platinum.utils.prompts import render_template

    repo_root = Path(__file__).resolve().parents[2]

    out = render_template(
        prompts_dir=repo_root / "config" / "prompts",
        track="atmospheric_horror",
        name="visual_prompts.j2",
        context={
            "aesthetic": "cinematic dark",
            "default_negative": "bright, anime",
            "palette": "deep shadow",
            "scenes": [
                {"index": 1, "narration_text": "It was a dark night."},
                {"index": 2, "narration_text": "He went into the cellar."},
            ],
        },
    )
    assert "cinematic dark" in out
    assert "bright, anime" in out
    assert "1." in out
    assert "dark night" in out
    assert "into the cellar" in out
