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
