"""SourceFetcher ABC contract tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from platinum.models.story import Source
from platinum.sources.base import SourceFetcher


class _OkFetcher(SourceFetcher):
    type = "ok"

    async def fetch(self, filters: dict[str, Any], limit: int) -> list[Source]:
        return [
            Source(
                type=self.type,
                url="https://example.com/1",
                title="Example",
                author="Test",
                raw_text="x" * 200,
                fetched_at=datetime(2026, 4, 24, 12),
                license="PD-US",
            )
        ]


def test_concrete_fetcher_must_declare_type() -> None:
    with pytest.raises(TypeError, match="no 'type' class attribute"):

        class _NoType(SourceFetcher):  # noqa: F841
            async def fetch(self, filters: dict[str, Any], limit: int) -> list[Source]:
                return []


def test_fetcher_cannot_be_instantiated_without_fetch() -> None:
    class _Bare(SourceFetcher):
        type = "bare"

    with pytest.raises(TypeError):
        _Bare()  # type: ignore[abstract]


async def test_fetch_returns_list_of_sources() -> None:
    fetcher = _OkFetcher()
    sources = await fetcher.fetch(filters={}, limit=1)
    assert len(sources) == 1
    assert sources[0].type == "ok"
    assert sources[0].title == "Example"


async def test_fetcher_type_propagates_into_source_type() -> None:
    fetcher = _OkFetcher()
    sources = await fetcher.fetch(filters={}, limit=1)
    assert sources[0].type == fetcher.type
