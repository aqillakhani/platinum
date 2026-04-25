"""Project Gutenberg fetcher via the Gutendex REST proxy.

Gutendex (https://gutendex.com) gives us a clean JSON metadata layer over
Project Gutenberg's catalog. We then fetch each book's plain-text body
straight from gutenberg.org and strip the standard PG header/footer.

Filters expected (from per-track YAML, all optional except ``authors``):
    authors:              list[str]   — full names; matched token-wise
    languages:            list[str]   — ISO 639-1 codes (default ["en"])
    min_words:            int         — body word-count floor
    max_words:            int         — body word-count ceiling
    license:              str         — informational; we always emit "PD-US"
                                        because Gutendex's ``copyright: false``
                                        flag is the actual safety gate.
    year_published_max:   int         — informational (Gutendex does not expose
                                        publication year; trust copyright flag).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx

from platinum.models.story import Source
from platinum.sources.base import SourceFetcher

logger = logging.getLogger(__name__)


_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _strip_boilerplate(raw: str) -> str:
    """Return the body of a Gutenberg .txt file with the standard
    *** START / *** END boilerplate removed. Returns ``raw`` unchanged if
    the markers are missing (rare older editions)."""
    start = _START_RE.search(raw)
    if start is None:
        return raw
    body = raw[start.end():]
    end = _END_RE.search(body)
    if end is None:
        return body.strip()
    return body[: end.start()].strip()


def _pick_text_url(formats: dict[str, str]) -> str | None:
    """Pick the best plain-text format URL from a Gutendex ``formats`` map."""
    for mime in (
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    ):
        if mime in formats:
            return formats[mime]
    for mime, url in formats.items():
        if mime.startswith("text/plain"):
            return url
    return None


def _author_matches(query: str, gutendex_authors: list[dict[str, Any]]) -> bool:
    """True iff every alphabetic token of ``query`` appears in some Gutendex
    author's name. Matches "Edgar Allan Poe" against "Poe, Edgar Allan" and
    "H. P. Lovecraft" against "Lovecraft, H. P. (Howard Phillips)"."""
    q = {t.lower() for t in _TOKEN_RE.findall(query)}
    if not q:
        return True
    for a in gutendex_authors:
        tokens = {t.lower() for t in _TOKEN_RE.findall(a.get("name", ""))}
        if q.issubset(tokens):
            return True
    return False


class GutendexFetcher(SourceFetcher):
    """Async Gutendex client that returns ready-to-use ``Source`` objects."""

    type = "gutenberg"

    BASE_URL: ClassVar[str] = "https://gutendex.com"
    EBOOK_URL: ClassVar[str] = "https://www.gutenberg.org/ebooks/{id}"
    USER_AGENT: ClassVar[str] = "Platinum/1.0 (cinematic short film pipeline)"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        # Inject a pre-built client (with MockTransport) for tests; otherwise
        # build a fresh one per call so timeouts and headers are predictable.
        self._client = client

    async def fetch(self, filters: dict[str, Any], limit: int) -> list[Source]:
        authors: list[str] = list(filters.get("authors") or [])
        languages: list[str] = list(filters.get("languages") or ["en"])
        min_words = int(filters.get("min_words", 0))
        max_words = int(filters.get("max_words", 10**18))

        # If no author filter, do one undirected search to keep the contract.
        author_queries: list[str] = authors if authors else [""]

        results: list[Source] = []
        seen_ids: set[int] = set()

        client = self._client or httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.USER_AGENT},
        )
        owns_client = self._client is None
        try:
            for author in author_queries:
                if len(results) >= limit:
                    break
                # Search by surname (last whitespace token) — Gutendex's
                # full-text search is generous enough that this catches all
                # works while keeping the per-query response small.
                search_term = author.split()[-1] if author else ""
                params = {"search": search_term, "languages": ",".join(languages)}
                try:
                    resp = await client.get(f"{self.BASE_URL}/books", params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Gutendex search failed for '%s': %s", search_term, exc
                    )
                    continue

                for book in data.get("results", []):
                    if len(results) >= limit:
                        break
                    book_id = book.get("id")
                    if book_id in seen_ids:
                        continue
                    if author and not _author_matches(author, book.get("authors", [])):
                        continue
                    text_url = _pick_text_url(book.get("formats", {}))
                    if text_url is None:
                        continue
                    try:
                        text_resp = await client.get(text_url)
                        text_resp.raise_for_status()
                        raw = text_resp.text
                    except httpx.HTTPError as exc:
                        logger.warning(
                            "Gutendex text fetch failed for book %s: %s",
                            book_id, exc,
                        )
                        continue

                    body = _strip_boilerplate(raw)
                    word_count = len(body.split())
                    if not (min_words <= word_count <= max_words):
                        continue

                    primary_author = (
                        book["authors"][0]["name"] if book.get("authors") else None
                    )
                    results.append(
                        Source(
                            type=self.type,
                            url=self.EBOOK_URL.format(id=book_id),
                            title=book.get("title", ""),
                            author=primary_author,
                            raw_text=body,
                            fetched_at=datetime.now(UTC),
                            license="PD-US",
                        )
                    )
                    seen_ids.add(book_id)
        finally:
            if owns_client:
                await client.aclose()

        logger.info(
            "Gutendex: %d/%d sources for authors=%s",
            len(results), limit, authors or "[any]",
        )
        return results
