"""Integration tests for ``platinum curate``.

These exercise the full Typer CLI over a real (tmp) project layout +
SQLite database. The interactive prompt reads from ``sys.stdin``, which
``CliRunner.invoke(input=...)`` populates with a scripted sequence.
The editor-open path is monkey-patched to a no-op so CI never spawns
a real editor.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from platinum.cli import app
from platinum.config import Config
from platinum.models.db import StageRunRow, sync_session
from platinum.models.story import (
    Source,
    StageRun,
    StageStatus,
    Story,
)

# ---------------------------------------------------------------------------
# Project fixture (same shape as the fetch integration test's cli_project)
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text("app:\n  log_level: INFO\n")
    (tmp_path / "config" / "tracks" / "atmospheric_horror.yaml").write_text(
        yaml.safe_dump({"track": {"id": "atmospheric_horror"}})
    )
    (tmp_path / "config" / "tracks" / "retro_scifi.yaml").write_text(
        yaml.safe_dump({"track": {"id": "retro_scifi"}})
    )
    monkeypatch.setattr("platinum.config._ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Pre-population helpers
# ---------------------------------------------------------------------------


def _seed_story(cfg: Config, story_id: str, *, track: str = "atmospheric_horror") -> Story:
    """Write an uncurated Story to disk + SQLite (mirrors what fetch does)."""
    src = Source(
        type="gutenberg",
        url=f"https://www.gutenberg.org/ebooks/{story_id}",
        title=f"Title for {story_id}",
        author="Edgar Allan Poe",
        raw_text="It was a dark and stormy night. " * 80,
        fetched_at=datetime(2026, 4, 24, 12, 0, 0),
        license="PD-US",
    )
    s = Story(
        id=story_id,
        track=track,
        source=src,
        stages=[
            StageRun(
                stage="source_fetcher",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 24),
                completed_at=datetime(2026, 4, 24),
            )
        ],
    )
    story_dir = cfg.story_dir(s.id)
    s.save(story_dir / "story.json")
    (story_dir / "source.txt").write_text(src.raw_text)
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_curate_walks_candidates_with_scripted_input(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approve the first, reject the second, skip the third — each
    decision should be reflected exactly in story.json + SQLite."""
    cfg = Config(root=cli_project)
    s1 = _seed_story(cfg, "story_2026_04_24_001")
    s2 = _seed_story(cfg, "story_2026_04_24_002")
    s3 = _seed_story(cfg, "story_2026_04_24_003")

    # Block the editor in case any test sends 'o'.
    monkeypatch.setattr(
        "platinum.pipeline.story_curator.subprocess.run", lambda *a, **kw: 0
    )

    result = CliRunner().invoke(app, ["curate"], input="a\nr\ns\n")
    assert result.exit_code == 0, result.output

    # On disk: story 1 has COMPLETE story_curator, 2 has SKIPPED, 3 unchanged.
    s1_loaded = Story.load(cfg.story_dir(s1.id) / "story.json")
    s2_loaded = Story.load(cfg.story_dir(s2.id) / "story.json")
    s3_loaded = Story.load(cfg.story_dir(s3.id) / "story.json")

    [r1] = [r for r in s1_loaded.stages if r.stage == "story_curator"]
    [r2] = [r for r in s2_loaded.stages if r.stage == "story_curator"]
    assert r1.status == StageStatus.COMPLETE
    assert r2.status == StageStatus.SKIPPED
    assert all(r.stage != "story_curator" for r in s3_loaded.stages)

    # Review gate echoes the decision in plain English on the JSON.
    assert s1_loaded.review_gates["curator"]["decision"] == "approved"
    assert s2_loaded.review_gates["curator"]["decision"] == "rejected"
    assert "curator" not in s3_loaded.review_gates

    # SQLite: 2 story_curator rows (approved + rejected); none for skipped.
    db_path = cfg.data_dir / "platinum.db"
    with sync_session(db_path) as session:
        curator_rows = (
            session.query(StageRunRow)
            .filter_by(stage="story_curator")
            .all()
        )
    by_story = {r.story_id: r.status for r in curator_rows}
    assert by_story == {s1.id: "complete", s2.id: "skipped"}

    # Output summarises the run.
    assert "approved=1" in result.output
    assert "rejected=1" in result.output
    assert "skipped=1" in result.output


def test_cli_curate_no_candidates_exits_zero_with_message(
    cli_project: Path,
) -> None:
    """Empty data/stories/ -> exit 0, friendly message, no DB writes."""
    result = CliRunner().invoke(app, ["curate"])
    assert result.exit_code == 0
    assert "No pending candidates" in result.output


def test_cli_curate_open_editor_then_approve(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``o`` invokes the editor exactly once, then ``a`` approves."""
    cfg = Config(root=cli_project)
    seeded = _seed_story(cfg, "story_2026_04_24_001")

    captured: list[list[str]] = []
    monkeypatch.setattr(
        "platinum.pipeline.story_curator.subprocess.run",
        lambda argv, *a, **kw: captured.append(list(argv)) or 0,
    )
    # Force a deterministic editor for the assertion.
    monkeypatch.setenv("EDITOR", "myeditor")

    result = CliRunner().invoke(app, ["curate"], input="o\na\n")
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0][0] == "myeditor"
    assert captured[0][-1].endswith("source.txt")

    loaded = Story.load(cfg.story_dir(seeded.id) / "story.json")
    [run] = [r for r in loaded.stages if r.stage == "story_curator"]
    assert run.status == StageStatus.COMPLETE


def test_cli_curate_track_filter_skips_other_tracks(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--track`` restricts the walk to matching stories only."""
    cfg = Config(root=cli_project)
    horror = _seed_story(cfg, "story_h_001", track="atmospheric_horror")
    sci_fi = _seed_story(cfg, "story_s_001", track="retro_scifi")

    monkeypatch.setattr(
        "platinum.pipeline.story_curator.subprocess.run", lambda *a, **kw: 0
    )

    result = CliRunner().invoke(
        app, ["curate", "--track", "atmospheric_horror"], input="a\n"
    )
    assert result.exit_code == 0, result.output

    h = Story.load(cfg.story_dir(horror.id) / "story.json")
    s = Story.load(cfg.story_dir(sci_fi.id) / "story.json")
    assert any(r.stage == "story_curator" for r in h.stages)
    assert all(r.stage != "story_curator" for r in s.stages)


def test_cli_curate_status_reflects_approval(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After approving, ``platinum status --story <id>`` shows COMPLETE."""
    cfg = Config(root=cli_project)
    seeded = _seed_story(cfg, "story_2026_04_24_001")

    monkeypatch.setattr(
        "platinum.pipeline.story_curator.subprocess.run", lambda *a, **kw: 0
    )

    runner = CliRunner()
    runner.invoke(app, ["curate"], input="a\n")
    status_result = runner.invoke(app, ["status", "--story", seeded.id])

    assert status_result.exit_code == 0
    # The status table prints "STORY_CURATOR" plus the status word.
    assert "story_curator" in status_result.output
    assert "COMPLETE" in status_result.output
