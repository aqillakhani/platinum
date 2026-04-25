"""Configuration loader for YAML settings, per-track configs, and .env secrets.

Vendored from `gold/src/gold/config.py` and adapted for platinum's directory layout:
  - `config/settings.yaml`            global app config
  - `config/tracks/<track>.yaml`      per-track style/voice/length/source config
  - `secrets/.env`                    API keys, OAuth tokens, vast.ai SSH details
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]  # platinum/ project root


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins on conflicts; nested dicts merged."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Loads and provides access to all configuration."""

    def __init__(self, root: Path | None = None):
        self.root = root or _ROOT
        self.config_dir = self.root / "config"
        self.secrets_dir = self.root / "secrets"
        self.data_dir = self.root / "data"
        self.library_dir = self.root / "library"

        # Load .env from secrets/
        env_path = self.secrets_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        # Global settings
        self.settings: dict[str, Any] = self._load_yaml("settings.yaml")

        # Env-var overrides for live-rental hosts (set per smoke run).
        # See docs/runbooks/vast-ai-keyframe-smoke.md for details.
        _ENV_OVERRIDES: tuple[tuple[tuple[str, str], str], ...] = (
            (("comfyui", "host"), "PLATINUM_COMFYUI_HOST"),
            (("aesthetics", "host"), "PLATINUM_AESTHETICS_HOST"),
        )
        for (section, key), env_var in _ENV_OVERRIDES:
            env_value = os.environ.get(env_var)
            if env_value:
                self.settings.setdefault(section, {})[key] = env_value

        # Per-track configs
        self.tracks: dict[str, dict[str, Any]] = {}
        tracks_dir = self.config_dir / "tracks"
        if tracks_dir.exists():
            for f in tracks_dir.glob("*.yaml"):
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                track_data = data.get("track", data)
                track_id = track_data.get("id", f.stem)
                self.tracks[track_id] = track_data

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a nested config value via dot notation: `api.anthropic.model`."""
        keys = dotpath.split(".")
        val: Any = self.settings
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
            if val is None:
                return default
        return val

    def env(self, key: str, default: str = "") -> str:
        """Read an environment variable (loaded from secrets/.env)."""
        return os.environ.get(key, default)

    def track(self, track_id: str) -> dict[str, Any]:
        """Get a track's full config; raises KeyError if missing."""
        if track_id not in self.tracks:
            raise KeyError(
                f"Track '{track_id}' not found. Available: {sorted(self.tracks.keys())}"
            )
        return self.tracks[track_id]

    @property
    def db_url(self) -> str:
        """Async SQLite URL for SQLAlchemy."""
        db_path = self.data_dir / "platinum.db"
        return f"sqlite+aiosqlite:///{db_path}"

    @property
    def db_url_sync(self) -> str:
        """Sync SQLite URL for SQLAlchemy / Alembic."""
        db_path = self.data_dir / "platinum.db"
        return f"sqlite:///{db_path}"

    @property
    def stories_dir(self) -> Path:
        return self.data_dir / "stories"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def workflows_dir(self) -> Path:
        return self.config_dir / "workflows"

    @property
    def prompts_dir(self) -> Path:
        return self.config_dir / "prompts"

    @property
    def luts_dir(self) -> Path:
        return self.config_dir / "luts"

    def story_dir(self, story_id: str) -> Path:
        """Per-story working directory; created on demand."""
        path = self.stories_dir / story_id
        path.mkdir(parents=True, exist_ok=True)
        return path
