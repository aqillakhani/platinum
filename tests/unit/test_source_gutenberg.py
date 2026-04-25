"""Gutendex / Project Gutenberg fetcher tests.

All HTTP is mocked via ``httpx.MockTransport`` — no live network in unit
tests. Live access is exercised in the CLI smoke test only.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from platinum.sources.gutenberg import (
    GutendexFetcher,
    _author_matches,
    _pick_text_url,
    _strip_boilerplate,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _book(
    book_id: int,
    title: str,
    authors: list[dict[str, Any]],
    languages: list[str] | None = None,
    text_url: str | None = None,
) -> dict[str, Any]:
    if text_url is None:
        text_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    return {
        "id": book_id,
        "title": title,
        "authors": authors,
        "languages": languages or ["en"],
        "subjects": [],
        "bookshelves": [],
        "copyright": False,
        "media_type": "Text",
        "formats": {
            "text/plain; charset=utf-8": text_url,
            "text/html": f"https://www.gutenberg.org/files/{book_id}/{book_id}-h.htm",
        },
        "download_count": 0,
    }


def _wrap_text(body: str, title: str = "TEST EBOOK") -> str:
    return (
        f"Project Gutenberg's {title}\n"
        "Some preface lines.\n\n"
        f"*** START OF THE PROJECT GUTENBERG EBOOK {title} ***\n\n"
        f"{body}\n\n"
        f"*** END OF THE PROJECT GUTENBERG EBOOK {title} ***\n"
        "Trailer with license boilerplate.\n"
    )


def _make_handler(
    routes: dict[str, dict[str, Any] | str | tuple[int, str]],
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler from a {url_path_or_full: payload} map.

    Payload may be a dict (JSON), a str (text/plain), or a (status, body) tuple.
    Match precedence: exact full URL, then path-suffix.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        # Exact URL match first
        if url in routes:
            payload = routes[url]
        else:
            # Path-suffix match (used for /books?search= patterns)
            match = next((p for p in routes if url.endswith(p) or p in url), None)
            if match is None:
                return httpx.Response(404, text=f"unmocked: {url}")
            payload = routes[match]
        if isinstance(payload, tuple):
            status, body = payload
            return httpx.Response(status, text=body)
        if isinstance(payload, dict):
            return httpx.Response(200, content=json.dumps(payload).encode())
        return httpx.Response(200, text=payload)

    return handler


def _client(routes: dict[str, Any]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_make_handler(routes)))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_strip_boilerplate_removes_header_and_footer() -> None:
    body = "The body of the story goes here, several sentences long."
    raw = _wrap_text(body)
    stripped = _strip_boilerplate(raw)
    assert body in stripped
    assert "Project Gutenberg's" not in stripped
    assert "Trailer" not in stripped


def test_strip_boilerplate_returns_input_when_markers_missing() -> None:
    plain = "No markers here. Just a body of text."
    assert _strip_boilerplate(plain) == plain


def test_strip_boilerplate_handles_alt_marker_phrasings() -> None:
    raw = (
        "preamble\n"
        "***START OF THIS PROJECT GUTENBERG EBOOK Test***\n"
        "BODY\n"
        "***END OF THIS PROJECT GUTENBERG EBOOK Test***\n"
        "trailer\n"
    )
    assert _strip_boilerplate(raw).strip() == "BODY"


def test_pick_text_url_prefers_utf8_then_ascii() -> None:
    formats = {
        "text/html": "https://x/h",
        "text/plain; charset=us-ascii": "https://x/ascii.txt",
        "text/plain; charset=utf-8": "https://x/utf8.txt",
    }
    assert _pick_text_url(formats) == "https://x/utf8.txt"


def test_pick_text_url_falls_back_to_ascii() -> None:
    formats = {
        "text/html": "https://x/h",
        "text/plain; charset=us-ascii": "https://x/ascii.txt",
    }
    assert _pick_text_url(formats) == "https://x/ascii.txt"


def test_pick_text_url_returns_none_when_no_text_format() -> None:
    formats = {"text/html": "https://x/h", "application/epub+zip": "https://x/epub"}
    assert _pick_text_url(formats) is None


def test_author_matches_handles_inverted_name() -> None:
    assert _author_matches(
        "Edgar Allan Poe",
        [{"name": "Poe, Edgar Allan", "birth_year": 1809, "death_year": 1849}],
    )


def test_author_matches_initials() -> None:
    assert _author_matches(
        "H. P. Lovecraft",
        [{"name": "Lovecraft, H. P. (Howard Phillips)"}],
    )


def test_author_matches_rejects_unrelated() -> None:
    assert not _author_matches(
        "Edgar Allan Poe",
        [{"name": "Twain, Mark"}],
    )


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


async def test_fetch_returns_sources_for_matching_authors() -> None:
    body_a = "TALE A " * 400  # ~800 words
    body_b = "TALE B " * 400
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [
                _book(1, "The Tell-Tale Heart",
                      [{"name": "Poe, Edgar Allan", "death_year": 1849}]),
                _book(2, "The Black Cat",
                      [{"name": "Poe, Edgar Allan", "death_year": 1849}]),
            ]
        },
        "https://www.gutenberg.org/cache/epub/1/pg1.txt": _wrap_text(body_a, "TELL TALE HEART"),
        "https://www.gutenberg.org/cache/epub/2/pg2.txt": _wrap_text(body_b, "BLACK CAT"),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "languages": ["en"], "min_words": 100},
        limit=10,
    )
    assert len(sources) == 2
    assert {s.title for s in sources} == {"The Tell-Tale Heart", "The Black Cat"}
    assert all(s.type == "gutenberg" for s in sources)
    assert all(s.license == "PD-US" for s in sources)
    assert all("TALE" in s.raw_text for s in sources)
    # Header/footer scrubbed
    assert all("Project Gutenberg's" not in s.raw_text for s in sources)


async def test_fetch_filters_by_min_words() -> None:
    short_body = "tiny story"  # 2 words
    long_body = "word " * 2000  # 2000 words
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [
                _book(1, "Short",
                      [{"name": "Poe, Edgar Allan"}]),
                _book(2, "Long",
                      [{"name": "Poe, Edgar Allan"}]),
            ]
        },
        "https://www.gutenberg.org/cache/epub/1/pg1.txt": _wrap_text(short_body),
        "https://www.gutenberg.org/cache/epub/2/pg2.txt": _wrap_text(long_body),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "min_words": 100, "max_words": 5000},
        limit=10,
    )
    assert [s.title for s in sources] == ["Long"]


async def test_fetch_filters_by_max_words() -> None:
    body = "word " * 6000
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [_book(1, "TooLong", [{"name": "Poe, Edgar Allan"}])]
        },
        "https://www.gutenberg.org/cache/epub/1/pg1.txt": _wrap_text(body),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "min_words": 100, "max_words": 1000},
        limit=10,
    )
    assert sources == []


async def test_fetch_skips_books_with_no_text_format() -> None:
    no_text_book = {
        "id": 99,
        "title": "Audio Only",
        "authors": [{"name": "Poe, Edgar Allan"}],
        "languages": ["en"],
        "copyright": False,
        "formats": {"audio/mpeg": "https://x/mp3"},
    }
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [
                no_text_book,
                _book(1, "Real", [{"name": "Poe, Edgar Allan"}]),
            ]
        },
        "https://www.gutenberg.org/cache/epub/1/pg1.txt": _wrap_text("body " * 200),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "min_words": 100},
        limit=10,
    )
    assert [s.title for s in sources] == ["Real"]


async def test_fetch_continues_when_text_fetch_errors() -> None:
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [
                _book(1, "Broken", [{"name": "Poe, Edgar Allan"}]),
                _book(2, "OK", [{"name": "Poe, Edgar Allan"}]),
            ]
        },
        "https://www.gutenberg.org/cache/epub/1/pg1.txt": (500, "server error"),
        "https://www.gutenberg.org/cache/epub/2/pg2.txt": _wrap_text("body " * 200),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "min_words": 100},
        limit=10,
    )
    assert [s.title for s in sources] == ["OK"]


async def test_fetch_respects_limit_across_authors() -> None:
    body = "word " * 200
    routes: dict[str, Any] = {}
    # Two authors, each returns 3 books, fetch limit 4 → 4 total returned.
    for author_idx, surname in enumerate(["Poe", "Lovecraft"], start=1):
        author_name = (
            f"{surname}, Edgar Allan" if surname == "Poe"
            else f"{surname}, H. P. (Howard Phillips)"
        )
        results = []
        for j in range(3):
            book_id = author_idx * 100 + j
            results.append(_book(book_id, f"{surname}_{j}", [{"name": author_name}]))
            routes[f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"] = (
                _wrap_text(body)
            )
        routes[f"/books?search={surname}&languages=en"] = {"results": results}

    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={
            "authors": ["Edgar Allan Poe", "H. P. Lovecraft"],
            "min_words": 100,
        },
        limit=4,
    )
    assert len(sources) == 4


async def test_fetch_skips_authors_unrelated_to_query_results() -> None:
    """Gutendex search by surname can return books by other people with the
    same surname (e.g. searching 'Poe' returns books by 'Poe, Other'). The
    fetcher must filter by the configured author list, not blindly trust the
    search results."""
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [
                _book(1, "Mismatch",
                      [{"name": "Poe, John Doe"}]),
                _book(2, "Match",
                      [{"name": "Poe, Edgar Allan"}]),
            ]
        },
        "https://www.gutenberg.org/cache/epub/2/pg2.txt": _wrap_text("body " * 200),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    sources = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "min_words": 100},
        limit=10,
    )
    assert [s.title for s in sources] == ["Match"]


async def test_fetch_records_canonical_book_url() -> None:
    routes = {
        "/books?search=Poe&languages=en": {
            "results": [_book(1064, "Tell-Tale", [{"name": "Poe, Edgar Allan"}])]
        },
        "https://www.gutenberg.org/cache/epub/1064/pg1064.txt": _wrap_text("body " * 200),
    }
    fetcher = GutendexFetcher(client=_client(routes))
    [src] = await fetcher.fetch(
        filters={"authors": ["Edgar Allan Poe"], "min_words": 100},
        limit=1,
    )
    assert src.url == "https://www.gutenberg.org/ebooks/1064"
    assert src.author == "Poe, Edgar Allan"


@pytest.mark.parametrize("languages", [["en"], ["en", "fr"]])
async def test_fetch_propagates_languages_to_query(languages: list[str]) -> None:
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        if "/books" in str(request.url):
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = GutendexFetcher(client=client)
    await fetcher.fetch(
        filters={"authors": ["Poe"], "languages": languages},
        limit=5,
    )
    expected_plain = ",".join(languages)
    expected_pct = "%2C".join(languages)
    assert any(
        f"languages={expected_plain}" in u or f"languages={expected_pct}" in u
        for u in captured_urls
    ), captured_urls
