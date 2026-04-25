"""Reddit fetcher — public ``.json`` endpoint, no auth required.

Ported from ``gold/utils/reddit.py`` and adapted to the ``SourceFetcher``
interface. Public Reddit JSON has no API key requirement but rate-limits
hard if you hammer it; we send a custom ``User-Agent`` and rely on the
retry decorator at the orchestrator level.

Reddit content is **never republished verbatim**. The Source's
``license`` field is set to ``Reddit-CC-BY-NC-Adapt`` as a breadcrumb so
the story_adapter stage (Session 4) knows to paraphrase rather than
quote.

Filters expected (from per-track YAML):
    subreddits:           list[str]   — without the ``r/`` prefix
    time_filter:          str         — hour/day/week/month/year/all (default ``all``)
    min_score:            int         — Reddit upvote floor
    min_words:            int         — selftext word-count floor
    max_words:            int         — selftext word-count ceiling
    adaptation_required:  bool        — informational; ignored here
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

import httpx

from platinum.models.story import Source
from platinum.sources.base import SourceFetcher


logger = logging.getLogger(__name__)


class RedditFetcher(SourceFetcher):
    """Async Reddit JSON client returning ``Source`` objects."""

    type = "reddit"

    BASE_URL: ClassVar[str] = "https://www.reddit.com"
    PERMALINK_URL: ClassVar[str] = "https://reddit.com{permalink}"
    USER_AGENT: ClassVar[str] = "Platinum/1.0 (cinematic short film pipeline)"
    LICENSE: ClassVar[str] = "Reddit-CC-BY-NC-Adapt"
    PER_REQUEST_LIMIT: ClassVar[int] = 25

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client

    async def fetch(self, filters: dict[str, Any], limit: int) -> list[Source]:
        subreddits: list[str] = list(filters.get("subreddits") or [])
        time_filter: str = str(filters.get("time_filter", "all"))
        min_score = int(filters.get("min_score", 0))
        min_words = int(filters.get("min_words", 0))
        max_words = int(filters.get("max_words", 10**18))

        if not subreddits:
            logger.warning("reddit fetch called without subreddits — returning []")
            return []

        results: list[Source] = []
        seen_ids: set[str] = set()

        client = self._client or httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": self.USER_AGENT},
        )
        owns_client = self._client is None
        try:
            for sub in subreddits:
                if len(results) >= limit:
                    break
                posts = await self._top_posts(client, sub, time_filter)
                for child in posts:
                    if len(results) >= limit:
                        break
                    post = child.get("data", {})
                    post_id = post.get("id", "")
                    if not post_id or post_id in seen_ids:
                        continue
                    if post.get("is_video"):
                        continue
                    selftext = post.get("selftext") or ""
                    if not selftext.strip():
                        continue
                    if int(post.get("score", 0)) < min_score:
                        continue
                    word_count = len(selftext.split())
                    if not (min_words <= word_count <= max_words):
                        continue
                    permalink = post.get("permalink", "")
                    results.append(
                        Source(
                            type=self.type,
                            url=self.PERMALINK_URL.format(permalink=permalink),
                            title=post.get("title", ""),
                            author=(
                                f"u/{post['author']}" if post.get("author") else None
                            ),
                            raw_text=selftext,
                            fetched_at=datetime.now(timezone.utc),
                            license=self.LICENSE,
                        )
                    )
                    seen_ids.add(post_id)
        finally:
            if owns_client:
                await client.aclose()

        logger.info(
            "Reddit: %d/%d sources for subreddits=%s",
            len(results), limit, subreddits,
        )
        return results

    async def _top_posts(
        self,
        client: httpx.AsyncClient,
        subreddit: str,
        time_filter: str,
    ) -> list[dict[str, Any]]:
        url = f"{self.BASE_URL}/r/{subreddit}/top.json"
        params = {"t": time_filter, "limit": str(self.PER_REQUEST_LIMIT)}
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Reddit fetch failed for r/%s: %s", subreddit, exc)
            return []
        return data.get("data", {}).get("children", []) or []
