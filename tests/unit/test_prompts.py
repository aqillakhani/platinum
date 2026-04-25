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
