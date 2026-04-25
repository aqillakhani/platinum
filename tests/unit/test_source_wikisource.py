"""Wikisource fetcher tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from platinum.sources.wikisource import (
    WikisourceFetcher,
    _clean_wikitext,
    _extract_author,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_clean_wikitext_strips_header_template() -> None:
    raw = (
        "{{header\n"
        "| title = The Horla\n"
        "| author = Guy de Maupassant\n"
        "| year = 1887\n"
        "}}\n"
        "\n"
        "The body of the story is here. It is a wonderful tale.\n"
    )
    cleaned = _clean_wikitext(raw)
    assert "header" not in cleaned
    assert "Maupassant" not in cleaned
    assert "The body of the story is here." in cleaned


def test_clean_wikitext_strips_refs_and_html() -> None:
    raw = "Story line.<ref>citation</ref> Another.<br/> End."
    cleaned = _clean_wikitext(raw)
    assert "<ref>" not in cleaned
    assert "<br" not in cleaned
    assert "citation" not in cleaned
    assert "Story line." in cleaned
    assert "Another." in cleaned
    assert "End." in cleaned


def test_clean_wikitext_resolves_wikilinks() -> None:
    raw = "Visit [[Some Page|the link]] and [[Other Page]] now."
    cleaned = _clean_wikitext(raw)
    assert cleaned == "Visit the link and Other Page now."


def test_clean_wikitext_handles_nested_templates() -> None:
    raw = "Before {{outer | inner = {{inner | x = y}} | rest}} After."
    cleaned = _clean_wikitext(raw)
    assert "{{" not in cleaned
    assert "}}" not in cleaned
    assert cleaned == "Before  After."


def test_extract_author_from_header() -> None:
    raw = (
        "{{header\n"
        "| title = X\n"
        "| author = Edgar Allan Poe\n"
        "}}\nbody"
    )
    assert _extract_author(raw) == "Edgar Allan Poe"


def test_extract_author_returns_none_when_absent() -> None:
    raw = "{{header\n| title = X\n}}\nbody"
    assert _extract_author(raw) is None


# ---------------------------------------------------------------------------
# Mock-transport helper
# ---------------------------------------------------------------------------


def _mediawiki_handler(
    *,
    category_pages: dict[str, list[dict[str, Any]]],
    page_wikitext: dict[str, str],
    fail_pages: set[str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a MockTransport handler that imitates the MediaWiki API."""
    fail_pages = fail_pages or set()

    def handler(request: httpx.Request) -> httpx.Response:
        params = parse_qs(urlparse(str(request.url)).query)
        action = params.get("action", [""])[0]
        if action == "query":
            cmtitle = params.get("cmtitle", [""])[0]
            assert params.get("list", [""])[0] == "categorymembers", params
            members = category_pages.get(cmtitle, [])
            return httpx.Response(
                200,
                content=json.dumps({"query": {"categorymembers": members}}).encode(),
            )
        if action == "parse":
            page = params.get("page", [""])[0]
            if page in fail_pages:
                return httpx.Response(500, text="boom")
            wikitext = page_wikitext.get(page)
            if wikitext is None:
                return httpx.Response(
                    200,
                    content=json.dumps({"error": {"code": "missingtitle"}}).encode(),
                )
            return httpx.Response(
                200,
                content=json.dumps(
                    {"parse": {"title": page, "wikitext": {"*": wikitext}}}
                ).encode(),
            )
        return httpx.Response(404, text=f"unmocked action: {action}")

    return handler


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


async def test_fetch_returns_sources_for_category() -> None:
    body = "Story body. " * 200
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": 1, "ns": 0, "title": "The Horla"},
                {"pageid": 2, "ns": 0, "title": "Some Other Tale"},
            ]
        },
        page_wikitext={
            "The Horla": (
                "{{header\n| title = The Horla\n| author = Guy de Maupassant\n}}\n"
                + body
            ),
            "Some Other Tale": (
                "{{header\n| title = Some Other Tale\n| author = Anonymous\n}}\n"
                + body
            ),
        },
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={"categories": ["Horror_short_stories"], "min_words": 100},
        limit=10,
    )
    assert {s.title for s in sources} == {"The Horla", "Some Other Tale"}
    assert all(s.type == "wikisource" for s in sources)
    horla = next(s for s in sources if s.title == "The Horla")
    assert horla.author == "Guy de Maupassant"
    assert "header" not in horla.raw_text
    assert "Story body." in horla.raw_text


async def test_fetch_filters_by_word_count() -> None:
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": 1, "ns": 0, "title": "Short"},
                {"pageid": 2, "ns": 0, "title": "Long"},
            ]
        },
        page_wikitext={
            "Short": "{{header\n| title = Short\n}}\nTiny.",
            "Long": "{{header\n| title = Long\n}}\n" + ("word " * 800),
        },
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={"categories": ["Horror_short_stories"], "min_words": 100, "max_words": 5000},
        limit=10,
    )
    assert [s.title for s in sources] == ["Long"]


async def test_fetch_skips_pages_outside_main_namespace() -> None:
    """Wikisource categories include sub-pages, talk pages, etc. Only ns=0
    are actual stories."""
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": 10, "ns": 14, "title": "Category:Subcategory"},
                {"pageid": 11, "ns": 1, "title": "Talk:Something"},
                {"pageid": 12, "ns": 0, "title": "Real Story"},
            ]
        },
        page_wikitext={"Real Story": "body " * 200},
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={"categories": ["Horror_short_stories"], "min_words": 100},
        limit=10,
    )
    assert [s.title for s in sources] == ["Real Story"]


async def test_fetch_continues_when_page_fetch_errors() -> None:
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": 1, "ns": 0, "title": "Broken"},
                {"pageid": 2, "ns": 0, "title": "OK"},
            ]
        },
        page_wikitext={"OK": "body " * 200},
        fail_pages={"Broken"},
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={"categories": ["Horror_short_stories"], "min_words": 100},
        limit=10,
    )
    assert [s.title for s in sources] == ["OK"]


async def test_fetch_respects_limit_across_categories() -> None:
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": i, "ns": 0, "title": f"Horror{i}"} for i in range(1, 4)
            ],
            "Category:Weird_fiction": [
                {"pageid": i + 100, "ns": 0, "title": f"Weird{i}"} for i in range(1, 4)
            ],
        },
        page_wikitext={
            **{f"Horror{i}": "body " * 200 for i in range(1, 4)},
            **{f"Weird{i}": "body " * 200 for i in range(1, 4)},
        },
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "categories": ["Horror_short_stories", "Weird_fiction"],
            "min_words": 100,
        },
        limit=4,
    )
    assert len(sources) == 4


async def test_fetch_url_points_to_canonical_wikisource_page() -> None:
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": 1, "ns": 0, "title": "The Horla"}
            ]
        },
        page_wikitext={"The Horla": "{{header\n| author = Maupassant\n}}\nbody " * 50},
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    [src] = await fetcher.fetch(
        filters={"categories": ["Horror_short_stories"], "min_words": 50},
        limit=1,
    )
    assert src.url == "https://en.wikisource.org/wiki/The_Horla"
    assert src.license == "PD-Old"


async def test_fetch_dedupes_pages_listed_in_multiple_categories() -> None:
    handler = _mediawiki_handler(
        category_pages={
            "Category:Horror_short_stories": [
                {"pageid": 1, "ns": 0, "title": "The Horla"}
            ],
            "Category:Weird_fiction": [
                {"pageid": 1, "ns": 0, "title": "The Horla"}
            ],
        },
        page_wikitext={"The Horla": "body " * 200},
    )
    fetcher = WikisourceFetcher(client=_client(handler))
    sources = await fetcher.fetch(
        filters={
            "categories": ["Horror_short_stories", "Weird_fiction"],
            "min_words": 100,
        },
        limit=10,
    )
    assert len(sources) == 1
