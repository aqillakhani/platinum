"""Source-fetcher registry — maps the YAML ``type`` string to a class."""

from __future__ import annotations

import httpx

from platinum.sources.base import SourceFetcher
from platinum.sources.gutenberg import GutendexFetcher
from platinum.sources.reddit import RedditFetcher
from platinum.sources.wikisource import WikisourceFetcher

REGISTRY: dict[str, type[SourceFetcher]] = {
    GutendexFetcher.type: GutendexFetcher,
    WikisourceFetcher.type: WikisourceFetcher,
    RedditFetcher.type: RedditFetcher,
}


def build_fetcher(
    type_: str, *, client: httpx.AsyncClient | None = None
) -> SourceFetcher | None:
    """Return a fetcher instance for ``type_`` or ``None`` if unknown.

    Unknown types are logged-and-skipped at the runner level; raising would
    take a single misconfigured track entry and break the whole fetch run.
    """
    cls = REGISTRY.get(type_)
    if cls is None:
        return None
    return cls(client=client)
