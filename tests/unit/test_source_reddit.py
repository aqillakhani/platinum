"""Reddit fetcher tests (ported + adapted from gold/utils/reddit.py)."""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import httpx

from platinum.sources.reddit import RedditFetcher


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _post(
    *,
    post_id: str = "abc",
    title: str = "A scary story",
    selftext: str = "body " * 1500,
    score: int = 9999,
    is_video: bool = False,
    subreddit: str = "nosleep",
    author: str = "ghost",
) -> dict[str, Any]:
    return {
        "data": {
            "id": post_id,
            "title": title,
            "selftext": selftext,
            "score": score,
            "num_comments": 100,
            "permalink": f"/r/{subreddit}/comments/{post_id}/",
            "subreddit": subreddit,
            "is_video": is_video,
            "author": author,
            "created_utc": 1700000000,
        }
    }


def _subreddit_handler(
    pages: dict[str, list[dict[str, Any]]],
) -> Callable[[httpx.Request], httpx.Response]:
    """Map subreddit name → list of post payloads."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path  # /r/<sub>/top.json
        parts = path.strip("/").split("/")
        # ['r', '<sub>', 'top.json']
        if len(parts) < 3 or parts[0] != "r":
            return httpx.Response(404, text="bad path")
        sub = parts[1]
        children = pages.get(sub, [])
        return httpx.Response(
            200,
            content=json.dumps({"data": {"children": children}}).encode(),
        )

    return handler


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


async def test_fetch_returns_sources_for_subreddit() -> None:
    handler = _subreddit_handler(
        {
            "nosleep": [
                _post(post_id="aaa", title="Tale One"),
                _post(post_id="bbb", title="Tale Two"),
            ]
        }
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["nosleep"],
            "time_filter": "all",
            "min_score": 100,
            "min_words": 100,
            "max_words": 10000,
        },
        limit=10,
    )
    assert {s.title for s in sources} == {"Tale One", "Tale Two"}
    assert all(s.type == "reddit" for s in sources)
    one = next(s for s in sources if s.title == "Tale One")
    assert one.url == "https://reddit.com/r/nosleep/comments/aaa/"
    assert "body" in one.raw_text


async def test_fetch_filters_by_min_score() -> None:
    handler = _subreddit_handler(
        {
            "nosleep": [
                _post(post_id="lo", title="Low", score=100),
                _post(post_id="hi", title="High", score=10000),
            ]
        }
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["nosleep"],
            "min_score": 5000,
            "min_words": 100,
        },
        limit=10,
    )
    assert [s.title for s in sources] == ["High"]


async def test_fetch_filters_by_word_count() -> None:
    handler = _subreddit_handler(
        {
            "nosleep": [
                _post(post_id="s", title="Short", selftext="too short"),
                _post(post_id="b", title="TooLong", selftext="word " * 10000),
                _post(post_id="g", title="JustRight", selftext="word " * 1500),
            ]
        }
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["nosleep"],
            "min_score": 100,
            "min_words": 1000,
            "max_words": 8000,
        },
        limit=10,
    )
    assert [s.title for s in sources] == ["JustRight"]


async def test_fetch_skips_video_posts() -> None:
    handler = _subreddit_handler(
        {
            "nosleep": [
                _post(post_id="vid", title="VidPost", is_video=True),
                _post(post_id="txt", title="TextPost"),
            ]
        }
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["nosleep"],
            "min_score": 100,
            "min_words": 100,
        },
        limit=10,
    )
    assert [s.title for s in sources] == ["TextPost"]


async def test_fetch_skips_empty_selftext() -> None:
    """Link-only posts have no body."""
    handler = _subreddit_handler(
        {
            "nosleep": [
                _post(post_id="link", title="LinkOnly", selftext=""),
                _post(post_id="real", title="Real"),
            ]
        }
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["nosleep"],
            "min_score": 100,
            "min_words": 100,
        },
        limit=10,
    )
    assert [s.title for s in sources] == ["Real"]


async def test_fetch_balances_across_subreddits() -> None:
    handler = _subreddit_handler(
        {
            "nosleep": [_post(post_id=f"n{i}", title=f"NoSleep{i}") for i in range(5)],
            "shortscarystories": [
                _post(post_id=f"s{i}", title=f"Short{i}", subreddit="shortscarystories")
                for i in range(5)
            ],
        }
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["nosleep", "shortscarystories"],
            "min_score": 100,
            "min_words": 100,
        },
        limit=4,
    )
    assert len(sources) == 4


async def test_fetch_passes_time_filter_in_query() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json={"data": {"children": []}})

    fetcher = RedditFetcher(client=_client(handler))
    await fetcher.fetch(
        filters={"subreddits": ["nosleep"], "time_filter": "year", "min_score": 0},
        limit=5,
    )
    assert captured, "no requests captured"
    qs = parse_qs(urlparse(captured[0]).query)
    assert qs.get("t") == ["year"]


async def test_fetch_continues_when_subreddit_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        if "/r/broken/" in path:
            return httpx.Response(503, text="reddit down")
        if "/r/nosleep/" in path:
            return httpx.Response(
                200,
                content=json.dumps(
                    {"data": {"children": [_post(post_id="ok", title="OK")]}}
                ).encode(),
            )
        return httpx.Response(404)

    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "subreddits": ["broken", "nosleep"],
            "min_score": 100,
            "min_words": 100,
        },
        limit=10,
    )
    assert [s.title for s in sources] == ["OK"]


async def test_fetch_marks_license_to_signal_adaptation() -> None:
    """Reddit content is paraphrased downstream, never republished verbatim.
    The license field is the breadcrumb so adapter stages know to rewrite."""
    handler = _subreddit_handler({"nosleep": [_post(post_id="x", title="X")]})
    fetcher = RedditFetcher(client=_client(handler))
    [src] = await fetcher.fetch(
        filters={"subreddits": ["nosleep"], "min_score": 100, "min_words": 100},
        limit=1,
    )
    # Anything that includes "Reddit" + "Adapt" satisfies — exact wording can evolve.
    assert "reddit" in src.license.lower()
    assert "adapt" in src.license.lower()


async def test_fetch_respects_limit_per_subreddit_cap() -> None:
    """Don't bombard one subreddit with limit=100 if the upstream returned 1k posts."""
    handler = _subreddit_handler(
        {"nosleep": [_post(post_id=f"n{i}", title=f"N{i}") for i in range(50)]}
    )
    fetcher = RedditFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={"subreddits": ["nosleep"], "min_score": 100, "min_words": 100},
        limit=5,
    )
    assert len(sources) == 5
