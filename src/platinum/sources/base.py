"""SourceFetcher abstract base class.

Each fetcher pulls candidate stories from one external corpus
(Project Gutenberg, Wikisource, Reddit, ...) and returns them as
``Source`` dataclasses ready to embed in a ``Story``. Per-track YAML
declares which fetchers run and supplies their filter dictionaries; the
shape of ``filters`` is fetcher-specific and validated by each subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from platinum.models.story import Source


class SourceFetcher(ABC):
    """Abstract source fetcher. Subclasses set ``type`` and implement ``fetch``."""

    type: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Intermediate ABCs (with abstract methods left over) are exempt.
        if not getattr(cls, "__abstractmethods__", frozenset()):
            if not cls.type:
                raise TypeError(
                    f"{cls.__name__} is a concrete SourceFetcher but has no 'type' "
                    f"class attribute"
                )

    @abstractmethod
    async def fetch(self, filters: dict[str, Any], limit: int) -> list[Source]:
        """Return up to ``limit`` candidate Sources matching ``filters``.

        Implementations may return fewer than ``limit`` if the upstream corpus
        runs out, but must never return more.
        """
