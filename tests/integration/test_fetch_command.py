"""Integration tests for the ``platinum fetch`` CLI command.

Covers:
  * Multi-source orchestration via ``fetch_track_sources`` with all three
    fetchers behind one mocked ``httpx.AsyncClient``.
  * End-to-end CLI invocation (``CliRunner``) that writes Story JSONs +
    raw source.txt files + SQLite rows to a tmp project root.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from platinum.cli import app
from platinum.config import Config
from platinum.models.db import StageRunRow, StoryRow, sync_session
from platinum.models.story import Source, Story
from platinum.sources.runner import (
    fetch_track_sources,
    next_story_id,
    persist_source_as_story,
)

# ---------------------------------------------------------------------------
# Multi-host mock transport
# ---------------------------------------------------------------------------


def _multi_host_handler() -> Callable[[httpx.Request], httpx.Response]:
    """One handler that imitates Gutendex, Wikisource, and Reddit."""

    body_text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        + "word " * 800
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
    )
    long_wikitext = "{{header\n| author = Maupassant\n}}\n" + ("body " * 800)
    long_self = "word " * 1500

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        if "gutendex" in host:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": 1064,
                            "title": "Tell-Tale Heart",
                            "authors": [{"name": "Poe, Edgar Allan"}],
                            "languages": ["en"],
                            "copyright": False,
                            "formats": {
                                "text/plain; charset=utf-8":
                                    "https://www.gutenberg.org/cache/epub/1064/pg1064.txt",
                            },
                        },
                        {
                            "id": 1065,
                            "title": "Black Cat",
                            "authors": [{"name": "Poe, Edgar Allan"}],
                            "languages": ["en"],
                            "copyright": False,
                            "formats": {
                                "text/plain; charset=utf-8":
                                    "https://www.gutenberg.org/cache/epub/1065/pg1065.txt",
                            },
                        },
                    ]
                },
            )
        if "gutenberg.org" in host:
            return httpx.Response(200, text=body_text)
        if "wikisource" in host:
            params = request.url.params
            if params.get("action") == "query":
                return httpx.Response(
                    200,
                    json={
                        "query": {
                            "categorymembers": [
                                {"pageid": 1, "ns": 0, "title": "The Horla"},
                                {"pageid": 2, "ns": 0, "title": "The Yellow Wallpaper"},
                            ]
                        }
                    },
                )
            if params.get("action") == "parse":
                return httpx.Response(
                    200,
                    json={
                        "parse": {
                            "title": params.get("page", ""),
                            "wikitext": {"*": long_wikitext},
                        }
                    },
                )
        if "reddit.com" in host:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "children": [
                            {
                                "data": {
                                    "id": "rid1",
                                    "title": "Reddit Tale",
                                    "selftext": long_self,
                                    "score": 9999,
                                    "permalink": "/r/nosleep/comments/rid1/",
                                    "subreddit": "nosleep",
                                    "is_video": False,
                                    "author": "ghost",
                                }
                            },
                            {
                                "data": {
                                    "id": "rid2",
                                    "title": "Another Reddit Tale",
                                    "selftext": long_self,
                                    "score": 9999,
                                    "permalink": "/r/nosleep/comments/rid2/",
                                    "subreddit": "nosleep",
                                    "is_video": False,
                                    "author": "ghost",
                                }
                            },
                        ]
                    }
                },
            )
        return httpx.Response(404, text=f"unmocked: {host}{path}")

    return handler


def _mock_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_multi_host_handler()))


# ---------------------------------------------------------------------------
# Runner unit tests
# ---------------------------------------------------------------------------


def test_next_story_id_starts_at_one_when_dir_empty(tmp_path: Path) -> None:
    from datetime import datetime

    sid = next_story_id(tmp_path, when=datetime(2026, 4, 24))
    assert sid == "story_2026_04_24_001"


def test_next_story_id_increments_within_day(tmp_path: Path) -> None:
    from datetime import datetime

    when = datetime(2026, 4, 24)
    (tmp_path / "story_2026_04_24_001").mkdir()
    (tmp_path / "story_2026_04_24_002").mkdir()
    (tmp_path / "story_2026_04_23_001").mkdir()  # other day, ignored
    sid = next_story_id(tmp_path, when=when)
    assert sid == "story_2026_04_24_003"


async def test_fetch_track_sources_pulls_from_three_fetchers() -> None:
    track_cfg = {
        "id": "atmospheric_horror",
        "sources": [
            {
                "type": "gutenberg",
                "filters": {"authors": ["Edgar Allan Poe"], "min_words": 100},
            },
            {
                "type": "wikisource",
                "filters": {"categories": ["Horror_short_stories"], "min_words": 100},
            },
            {
                "type": "reddit",
                "filters": {
                    "subreddits": ["nosleep"],
                    "min_score": 100,
                    "min_words": 100,
                },
            },
        ],
    }
    sources = await fetch_track_sources(
        track_cfg, limit=6, client_factory=_mock_client_factory
    )
    assert len(sources) == 6
    types = [s.type for s in sources]
    # Order is YAML order: Gutenberg first (drains), then Wikisource, then Reddit.
    assert types == ["gutenberg"] * 2 + ["wikisource"] * 2 + ["reddit"] * 2


async def test_fetch_track_sources_stops_when_limit_reached() -> None:
    track_cfg = {
        "sources": [
            {"type": "gutenberg", "filters": {"authors": ["Poe"], "min_words": 100}},
            {
                "type": "reddit",
                "filters": {
                    "subreddits": ["nosleep"],
                    "min_score": 100,
                    "min_words": 100,
                },
            },
        ],
    }
    sources = await fetch_track_sources(
        track_cfg, limit=2, client_factory=_mock_client_factory
    )
    # Gutendex returns 2; reddit should not be called since limit met.
    assert all(s.type == "gutenberg" for s in sources)
    assert len(sources) == 2


async def test_fetch_track_sources_skips_unknown_type() -> None:
    track_cfg = {
        "sources": [
            {"type": "made_up", "filters": {}},
            {
                "type": "reddit",
                "filters": {
                    "subreddits": ["nosleep"],
                    "min_score": 100,
                    "min_words": 100,
                },
            },
        ],
    }
    sources = await fetch_track_sources(
        track_cfg, limit=10, client_factory=_mock_client_factory
    )
    assert all(s.type == "reddit" for s in sources)


def test_persist_source_as_story_writes_json_and_raw(
    tmp_project: Path, source: Source
) -> None:
    cfg = Config(root=tmp_project)
    story = persist_source_as_story(cfg, source, track="atmospheric_horror")
    story_dir = cfg.story_dir(story.id)
    assert (story_dir / "story.json").exists()
    assert (story_dir / "source.txt").exists()
    # source.txt mirrors raw_text exactly so downstream tools (Whisper, etc.)
    # can read it without parsing JSON.
    assert (story_dir / "source.txt").read_text(encoding="utf-8") == source.raw_text
    # story.json round-trips through Story.load.
    reloaded = Story.load(story_dir / "story.json")
    assert reloaded.source.title == source.title
    # source_fetcher StageRun is recorded as COMPLETE so downstream stages
    # can resume from this checkpoint without re-fetching.
    [run] = [r for r in reloaded.stages if r.stage == "source_fetcher"]
    assert run.status.value == "complete"


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tmp project root with config + atmospheric_horror track YAML."""
    (tmp_path / "config" / "tracks").mkdir(parents=True)
    (tmp_path / "secrets").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text("app:\n  log_level: INFO\n")
    track_yaml = {
        "track": {
            "id": "atmospheric_horror",
            "sources": [
                {
                    "type": "gutenberg",
                    "filters": {"authors": ["Edgar Allan Poe"], "min_words": 100},
                },
                {
                    "type": "wikisource",
                    "filters": {"categories": ["Horror_short_stories"], "min_words": 100},
                },
                {
                    "type": "reddit",
                    "filters": {
                        "subreddits": ["nosleep"],
                        "min_score": 100,
                        "min_words": 100,
                    },
                },
            ],
        }
    }
    (tmp_path / "config" / "tracks" / "atmospheric_horror.yaml").write_text(
        yaml.safe_dump(track_yaml), encoding="utf-8"
    )
    # Pin the Config root for this test process.
    monkeypatch.setattr("platinum.config._ROOT", tmp_path)
    return tmp_path


def test_cli_fetch_writes_n_stories_to_disk(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject the mocked client factory into the runner module.
    import platinum.sources.runner as runner_mod
    monkeypatch.setattr(runner_mod, "_default_client_factory", _mock_client_factory)

    result = CliRunner().invoke(
        app, ["fetch", "--track", "atmospheric_horror", "--limit", "5"]
    )
    assert result.exit_code == 0, result.output
    stories_dir = cli_project / "data" / "stories"
    story_jsons = list(stories_dir.glob("story_*/story.json"))
    assert len(story_jsons) == 5
    # Each has a sibling raw source.txt.
    for sj in story_jsons:
        assert (sj.parent / "source.txt").exists()
        assert (sj.parent / "source.txt").stat().st_size > 0


def test_cli_fetch_projects_to_sqlite(
    cli_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import platinum.sources.runner as runner_mod
    monkeypatch.setattr(runner_mod, "_default_client_factory", _mock_client_factory)

    result = CliRunner().invoke(
        app, ["fetch", "--track", "atmospheric_horror", "--limit", "3"]
    )
    assert result.exit_code == 0, result.output

    db_path = cli_project / "data" / "platinum.db"
    assert db_path.exists()
    with sync_session(db_path) as session:
        story_ids = {row.id for row in session.query(StoryRow).all()}
        run_stages = {row.stage for row in session.query(StageRunRow).all()}
    assert len(story_ids) == 3
    assert "source_fetcher" in run_stages


def test_cli_fetch_unknown_track_exits_nonzero(cli_project: Path) -> None:
    result = CliRunner().invoke(
        app, ["fetch", "--track", "does_not_exist", "--limit", "1"]
    )
    assert result.exit_code != 0
