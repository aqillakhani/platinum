"""Wikisource fetcher via the MediaWiki action API.

We pull category members (`list=categorymembers`), then for each page
request its raw wikitext (`action=parse&prop=wikitext`) and run a small
cleanup pass to remove templates, refs, HTML tags, and resolve wikilinks.
The story_adapter (Session 4) handles deeper polishing.

Filters expected (from per-track YAML):
    categories:   list[str]   — Wikisource category names without the
                                 ``Category:`` prefix (e.g. ``Horror_short_stories``)
    min_words:    int         — body word-count floor
    max_words:    int         — body word-count ceiling
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx

from platinum.models.story import Source
from platinum.sources.base import SourceFetcher


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wikitext cleanup
# ---------------------------------------------------------------------------


_REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref>|<ref\b[^/>]*/>", re.DOTALL | re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_WIKILINK_PIPE_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_AUTHOR_RE = re.compile(
    r"\{\{header[^}]*?\bauthor\s*=\s*([^|}\n]+)",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def _strip_templates(text: str) -> str:
    """Iteratively remove ``{{...}}`` templates, innermost-first, so nested
    templates collapse cleanly. Bounded by string length to guarantee
    termination."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return text


def _clean_wikitext(raw: str) -> str:
    text = _strip_templates(raw)
    text = _REF_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _WIKILINK_PIPE_RE.sub(r"\2", text)
    text = _WIKILINK_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub("\n", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_author(raw: str) -> Optional[str]:
    m = _AUTHOR_RE.search(raw)
    if not m:
        return None
    val = m.group(1).strip()
    return val or None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class WikisourceFetcher(SourceFetcher):
    """Async Wikisource MediaWiki API client returning ``Source`` objects."""

    type = "wikisource"

    API_URL: ClassVar[str] = "https://en.wikisource.org/w/api.php"
    PAGE_URL: ClassVar[str] = "https://en.wikisource.org/wiki/{title}"
    USER_AGENT: ClassVar[str] = "Platinum/1.0 (cinematic short film pipeline)"
    CATEGORY_LIMIT: ClassVar[int] = 50

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client

    async def fetch(self, filters: dict[str, Any], limit: int) -> list[Source]:
        categories: list[str] = list(filters.get("categories") or [])
        min_words = int(filters.get("min_words", 0))
        max_words = int(filters.get("max_words", 10**18))

        if not categories:
            logger.warning("wikisource fetch called without categories — returning []")
            return []

        results: list[Source] = []
        seen_titles: set[str] = set()

        client = self._client or httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.USER_AGENT},
        )
        owns_client = self._client is None
        try:
            for cat in categories:
                if len(results) >= limit:
                    break
                cat_title = cat if cat.startswith("Category:") else f"Category:{cat}"
                members = await self._list_category(client, cat_title)
                for member in members:
                    if len(results) >= limit:
                        break
                    if member.get("ns") != 0:
                        continue  # skip Talk:, Category:, etc.
                    title = member.get("title", "")
                    if not title or title in seen_titles:
                        continue
                    raw = await self._fetch_wikitext(client, title)
                    if raw is None:
                        continue
                    body = _clean_wikitext(raw)
                    word_count = len(body.split())
                    if not (min_words <= word_count <= max_words):
                        continue
                    results.append(
                        Source(
                            type=self.type,
                            url=self.PAGE_URL.format(title=title.replace(" ", "_")),
                            title=title,
                            author=_extract_author(raw),
                            raw_text=body,
                            fetched_at=datetime.now(timezone.utc),
                            license="PD-Old",
                        )
                    )
                    seen_titles.add(title)
        finally:
            if owns_client:
                await client.aclose()

        logger.info(
            "Wikisource: %d/%d sources for categories=%s",
            len(results), limit, categories,
        )
        return results

    async def _list_category(
        self, client: httpx.AsyncClient, cat_title: str
    ) -> list[dict[str, Any]]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmnamespace": "0",
            "cmlimit": str(self.CATEGORY_LIMIT),
            "format": "json",
        }
        try:
            resp = await client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Wikisource category %s failed: %s", cat_title, exc)
            return []
        return data.get("query", {}).get("categorymembers", []) or []

    async def _fetch_wikitext(
        self, client: httpx.AsyncClient, page_title: str
    ) -> Optional[str]:
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "format": "json",
        }
        try:
            resp = await client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Wikisource page fetch failed for %r: %s", page_title, exc)
            return None
        if "error" in data:
            logger.info(
                "Wikisource missing/error for %r: %s",
                page_title, data["error"].get("code"),
            )
            return None
        return data.get("parse", {}).get("wikitext", {}).get("*")
