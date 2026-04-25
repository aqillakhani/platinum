"""Tests for src/platinum/config.py."""
from __future__ import annotations

from platinum.config import Config


def test_env_var_overrides_comfyui_host(monkeypatch, tmp_path):
    monkeypatch.setenv("PLATINUM_COMFYUI_HOST", "http://override:9999")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text(
        "comfyui:\n  host: http://yaml-default:8188\n"
    )
    cfg = Config(root=tmp_path)
    assert cfg.settings["comfyui"]["host"] == "http://override:9999"


def test_env_var_aesthetics_host_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PLATINUM_AESTHETICS_HOST", "http://override:8189")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text("aesthetics: {}\n")
    cfg = Config(root=tmp_path)
    assert cfg.settings["aesthetics"]["host"] == "http://override:8189"


def test_no_env_var_leaves_yaml_value(monkeypatch, tmp_path):
    monkeypatch.delenv("PLATINUM_COMFYUI_HOST", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text(
        "comfyui:\n  host: http://yaml-default:8188\n"
    )
    cfg = Config(root=tmp_path)
    assert cfg.settings["comfyui"]["host"] == "http://yaml-default:8188"
