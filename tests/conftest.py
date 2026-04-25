"""Pytest fixtures for platinum tests."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest

from platinum.config import Config
from platinum.models.story import Scene, Source, StageRun, StageStatus, Story
from platinum.pipeline.context import PipelineContext


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A minimal platinum project layout under tmp_path (config/, data/, secrets/)."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "tracks").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text("app:\n  log_level: INFO\n")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


@pytest.fixture
def config(tmp_project: Path) -> Config:
    return Config(root=tmp_project)


@pytest.fixture
def context(config: Config) -> PipelineContext:
    return PipelineContext(config=config, logger=logging.getLogger("test"))


@pytest.fixture
def source() -> Source:
    return Source(
        type="gutenberg",
        url="https://www.gutenberg.org/ebooks/1",
        title="The Tell-Tale Heart",
        author="Edgar Allan Poe",
        raw_text="TRUE! —nervous —very, very dreadfully nervous I had been...",
        fetched_at=datetime(2026, 4, 24, 12, 0, 0),
        license="PD-US",
    )


@pytest.fixture
def story(source: Source) -> Story:
    return Story(
        id="story_test_001",
        track="atmospheric_horror",
        source=source,
        scenes=[
            Scene(id="scene_001", index=0, narration_text="It was a dark and stormy night."),
            Scene(id="scene_002", index=1, narration_text="The old man lay sleeping."),
        ],
        stages=[
            StageRun(
                stage="source_fetcher",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 24, 12, 1, 0),
                completed_at=datetime(2026, 4, 24, 12, 1, 30),
                artifacts={"bytes": 12345},
            ),
        ],
    )
