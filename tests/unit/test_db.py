"""SQLAlchemy schema + sync_from_story projection tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect

from platinum.models.db import (
    ApiUsageRow,
    Base,
    SceneRow,
    StageRunRow,
    StoryRow,
    create_all,
    sync_from_story,
    sync_session,
)
from platinum.models.story import StageRun, StageStatus, Story


def _db_path(tmp: Path) -> Path:
    return tmp / "platinum.db"


def test_create_all_creates_four_tables(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    create_all(db)
    with sync_session(db) as session:
        insp = inspect(session.connection())
        tables = set(insp.get_table_names())
    assert {"stories", "scenes", "stage_runs", "api_usage"} <= tables


def test_create_all_is_idempotent(tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    create_all(db)
    create_all(db)  # must not raise


def test_sync_from_story_inserts_rows(story: Story, tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    create_all(db)
    with sync_session(db) as s:
        sync_from_story(s, story)
    with sync_session(db) as s:
        stories = s.query(StoryRow).all()
        scenes = s.query(SceneRow).all()
        runs = s.query(StageRunRow).all()
    assert len(stories) == 1
    assert stories[0].id == story.id
    assert stories[0].track == "atmospheric_horror"
    assert len(scenes) == 2
    assert {sc.id for sc in scenes} == {"scene_001", "scene_002"}
    assert len(runs) == 1


def test_sync_from_story_replaces_children_on_rerun(story: Story, tmp_path: Path) -> None:
    """Calling sync_from_story twice should not duplicate scenes or stage_runs."""
    db = _db_path(tmp_path)
    create_all(db)
    with sync_session(db) as s:
        sync_from_story(s, story)
    # Add another stage run and sync again.
    story.stages.append(
        StageRun(stage="story_adapter", status=StageStatus.COMPLETE,
                 started_at=datetime.now(), completed_at=datetime.now())
    )
    with sync_session(db) as s:
        sync_from_story(s, story)
    with sync_session(db) as s:
        runs = s.query(StageRunRow).filter_by(story_id=story.id).all()
        scenes = s.query(SceneRow).filter_by(story_id=story.id).all()
    assert len(runs) == 2  # not 3 (old row + new 2) — wiped and re-inserted
    assert len(scenes) == 2


def test_status_derives_to_failed_on_latest_fail(story: Story, tmp_path: Path) -> None:
    db = _db_path(tmp_path)
    create_all(db)
    story.stages.append(
        StageRun(stage="story_adapter", status=StageStatus.FAILED,
                 started_at=datetime.now(), completed_at=datetime.now(), error="x")
    )
    with sync_session(db) as s:
        sync_from_story(s, story)
    with sync_session(db) as s:
        row = s.get(StoryRow, story.id)
    assert row is not None
    assert row.status == "failed"


def test_status_derives_to_published_when_publisher_complete(
    story: Story, tmp_path: Path
) -> None:
    db = _db_path(tmp_path)
    create_all(db)
    story.stages.append(
        StageRun(stage="publisher", status=StageStatus.COMPLETE,
                 started_at=datetime.now(), completed_at=datetime.now())
    )
    with sync_session(db) as s:
        sync_from_story(s, story)
    with sync_session(db) as s:
        row = s.get(StoryRow, story.id)
    assert row is not None
    assert row.status == "published"


def test_cost_usd_sums_from_api_usage(story: Story, tmp_path: Path) -> None:
    """stories.cost_usd should reflect the sum of api_usage.cost_usd rows
    for that story, re-computed on each sync."""
    db = _db_path(tmp_path)
    create_all(db)
    # Insert synthetic api_usage rows.
    with sync_session(db) as s:
        s.add_all([
            ApiUsageRow(story_id=story.id, provider="anthropic", model="claude-sonnet-4-5",
                        input_tokens=1000, output_tokens=500, cost_usd=0.25, ts=datetime.now()),
            ApiUsageRow(story_id=story.id, provider="anthropic", model="claude-sonnet-4-5",
                        input_tokens=2000, output_tokens=700, cost_usd=0.40, ts=datetime.now()),
        ])
    # Now sync the story; should roll the sum into stories.cost_usd.
    with sync_session(db) as s:
        sync_from_story(s, story)
    with sync_session(db) as s:
        row = s.get(StoryRow, story.id)
    assert row is not None
    assert abs(row.cost_usd - 0.65) < 1e-6
