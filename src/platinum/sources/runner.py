"""Multi-source orchestration + story persistence.

The CLI ``platinum fetch`` command calls into this module. It is kept
separate from ``cli.py`` so unit tests can exercise the orchestration
without spinning up Typer.

Story IDs follow the convention ``story_YYYY_MM_DD_NNN`` — the date is
the wall-clock day at fetch time and ``NNN`` is the next free integer
after counting existing per-day directories. The scheme is single-process
safe (no concurrent fetches expected) and human-greppable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx

from platinum.config import Config
from platinum.models.db import create_all, sync_from_story, sync_session
from platinum.models.story import Source, StageRun, StageStatus, Story
from platinum.sources.registry import build_fetcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory injection
# ---------------------------------------------------------------------------


_USER_AGENT = (
    "Platinum/1.0 (cinematic short film pipeline; "
    "+https://github.com/AqilLakhani/platinum)"
)


def _default_client_factory() -> httpx.AsyncClient:
    """Real httpx client used at runtime. Tests monkey-patch this attribute
    to inject ``MockTransport``-backed clients.

    ``follow_redirects=True`` is required because Gutendex normalises
    ``/books`` to ``/books/`` via 301 and Wikisource may relocate API
    endpoints. The User-Agent includes a contact URL per Wikipedia's
    user-agent policy (Wikisource enforces it as 403 otherwise)."""
    return httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# Story id generation
# ---------------------------------------------------------------------------


def next_story_id(stories_dir: Path, when: datetime | None = None) -> str:
    """Compute the next ``story_YYYY_MM_DD_NNN`` id for the given day."""
    now = when or datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    prefix = f"story_{date_str}_"
    if stories_dir.exists():
        existing = sum(
            1 for d in stories_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        )
    else:
        existing = 0
    return f"{prefix}{existing + 1:03d}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def fetch_track_sources(
    track_cfg: dict,
    limit: int,
    *,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> list[Source]:
    """Drive every fetcher listed under ``track_cfg['sources']`` in order
    until ``limit`` Sources are collected. One client is built per fetcher
    so connection state never leaks across hosts."""
    factory = client_factory or _default_client_factory
    out: list[Source] = []

    for spec in track_cfg.get("sources") or []:
        if len(out) >= limit:
            break
        type_ = spec.get("type", "")
        client = factory()
        try:
            fetcher = build_fetcher(type_, client=client)
            if fetcher is None:
                logger.warning("Unknown source type %r — skipping", type_)
                continue
            try:
                sources = await fetcher.fetch(
                    spec.get("filters") or {},
                    limit=limit - len(out),
                )
            except Exception as exc:
                # One fetcher's failure must not abort the whole fetch run
                # — others may still succeed and the user can retry later.
                logger.exception("fetcher %r failed: %s", type_, exc)
                continue
            out.extend(sources)
        finally:
            await client.aclose()

    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_source_as_story(
    cfg: Config,
    source: Source,
    track: str,
    *,
    when: datetime | None = None,
) -> Story:
    """Wrap a fetched ``Source`` in a fresh ``Story`` and write to disk.

    Side-effects:
      * Creates ``data/stories/<id>/`` if missing.
      * Writes ``story.json`` (atomic via ``Story.save``) and a sibling
        ``source.txt`` mirroring ``source.raw_text`` for downstream tools.
      * Projects the new row into SQLite (creates the schema first if
        absent so this is safe to call before any other stage has run).

    Returns the persisted ``Story``.
    """
    cfg.stories_dir.mkdir(parents=True, exist_ok=True)
    story_id = next_story_id(cfg.stories_dir, when=when)
    story_dir = cfg.story_dir(story_id)

    now = when or datetime.now()
    story = Story(
        id=story_id,
        track=track,
        source=source,
        stages=[
            StageRun(
                stage="source_fetcher",
                status=StageStatus.COMPLETE,
                started_at=now,
                completed_at=now,
                artifacts={
                    "source_type": source.type,
                    "url": source.url,
                    "word_count": len(source.raw_text.split()),
                },
            )
        ],
    )

    story.save(story_dir / "story.json")
    (story_dir / "source.txt").write_text(source.raw_text, encoding="utf-8")

    db_path = cfg.data_dir / "platinum.db"
    create_all(db_path)
    with sync_session(db_path) as session:
        sync_from_story(session, story)

    return story
