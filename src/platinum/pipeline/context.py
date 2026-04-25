"""PipelineContext — shared runtime handles passed to every Stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from platinum.config import Config
from platinum.models.story import Story


@dataclass
class PipelineContext:
    """Runtime context for a pipeline run.

    Thin wrapper that bundles configuration and a logger; all paths are
    derived from ``Config`` so there's one source of truth.
    """

    config: Config
    logger: logging.Logger

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @property
    def db_path(self) -> Path:
        return self.config.data_dir / "platinum.db"

    def story_dir(self, story: Story) -> Path:
        """Per-story working directory; created on demand."""
        return self.config.story_dir(story.id)

    def story_path(self, story: Story) -> Path:
        """Canonical ``story.json`` path for a given Story."""
        return self.story_dir(story) / "story.json"
