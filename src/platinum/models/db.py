"""SQLAlchemy schema + session management for platinum.

Per-story ``story.json`` is the source of truth; SQLite is a derived index
for queries like "all horror stories in progress" or "total cost by track".
``sync_from_story`` projects a Story document into the stories/scenes/
stage_runs rows. ``api_usage`` is append-only, written from utilities that
make paid API calls (see ``utils/claude.py`` in Session 4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from platinum.models.story import StageStatus, Story

# ---------------------------------------------------------------------------
# Declarative base + tables
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class StoryRow(Base):
    __tablename__ = "stories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    track: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class SceneRow(Base):
    __tablename__ = "scenes"

    story_id: Mapped[str] = mapped_column(
        String, ForeignKey("stories.id", ondelete="CASCADE"), primary_key=True
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, default="pending")
    keyframe_score: Mapped[float] = mapped_column(Float, default=0.0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)


class StageRunRow(Base):
    __tablename__ = "stage_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    story_id: Mapped[str] = mapped_column(
        String, ForeignKey("stories.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ApiUsageRow(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    story_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("stories.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    ts: Mapped[datetime] = mapped_column(DateTime)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def _sync_url(db_path: Path) -> str:
    return f"sqlite:///{Path(db_path).as_posix()}"


def _async_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{Path(db_path).as_posix()}"


def create_all(db_path: Path) -> None:
    """Create every table declared on ``Base``. Idempotent."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(_sync_url(db_path), future=True)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def make_sync_session_factory(db_path: Path) -> sessionmaker[Session]:
    engine = create_engine(_sync_url(db_path), future=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def make_async_session_factory(db_path: Path) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(_async_url(db_path), future=True)
    return async_sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def sync_session(db_path: Path) -> Iterator[Session]:
    """One-shot sync session. Disposes its engine on exit so the SQLite
    file handle is released (matters on Windows where open handles block
    deletion of the DB file)."""
    engine = create_engine(_sync_url(db_path), future=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


@asynccontextmanager
async def async_session(db_path: Path) -> AsyncIterator[AsyncSession]:
    """One-shot async session. Disposes its engine on exit for the same
    reason as ``sync_session``."""
    engine = create_async_engine(_async_url(db_path), future=True)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Projection: Story JSON -> SQLite
# ---------------------------------------------------------------------------


def _derive_status(story: Story) -> str:
    """Rough pipeline status for the stories.status column.

    'pending' — no stages run
    'published' — publisher stage's latest run is COMPLETE
    'failed' — most recent stage run is FAILED (orchestrator halts on fail)
    'in_progress' — otherwise
    """
    if not story.stages:
        return "pending"
    pub = story.latest_stage_run("publisher")
    if pub and pub.status == StageStatus.COMPLETE:
        return "published"
    if story.stages[-1].status == StageStatus.FAILED:
        return "failed"
    return "in_progress"


def sync_from_story(
    session: Session,
    story: Story,
    *,
    now: datetime | None = None,
) -> None:
    """Upsert the stories/scenes/stage_runs rows for a single Story.

    Child tables (scenes, stage_runs) are wiped and re-inserted — child
    volumes are small per story (tens of scenes, low tens of stage runs).
    api_usage is not touched here; it is written by the utilities that make
    API calls and is summed back into ``stories.cost_usd`` below.
    """
    now = now or datetime.now()

    existing = session.get(StoryRow, story.id)
    total_cost = session.scalar(
        select(func.coalesce(func.sum(ApiUsageRow.cost_usd), 0.0))
        .where(ApiUsageRow.story_id == story.id)
    ) or 0.0

    pub = story.latest_stage_run("publisher")
    published_at = (
        pub.completed_at
        if pub and pub.status == StageStatus.COMPLETE
        else None
    )

    session.merge(
        StoryRow(
            id=story.id,
            track=story.track,
            status=_derive_status(story),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            published_at=published_at,
            duration_seconds=sum(s.video_duration_seconds for s in story.scenes),
            cost_usd=float(total_cost),
        )
    )

    session.execute(delete(SceneRow).where(SceneRow.story_id == story.id))
    for scene in story.scenes:
        best_score = max(scene.keyframe_scores) if scene.keyframe_scores else 0.0
        session.add(
            SceneRow(
                story_id=story.id,
                id=scene.id,
                index=scene.index,
                status=scene.review_status.value,
                keyframe_score=best_score,
                duration=scene.video_duration_seconds,
            )
        )

    session.execute(delete(StageRunRow).where(StageRunRow.story_id == story.id))
    for run in story.stages:
        session.add(
            StageRunRow(
                story_id=story.id,
                stage=run.stage,
                status=run.status.value,
                started_at=run.started_at,
                completed_at=run.completed_at,
                error=run.error,
            )
        )
