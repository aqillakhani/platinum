"""Exponential backoff retry decorator.

Vendored from gold/src/gold/utils/retry.py (platinum stays self-contained).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator for async functions with exponential backoff + jitter."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            "Failed after %d retries: %s — %s", max_retries, func.__name__, exc
                        )
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)
                    total_delay = delay + jitter
                    logger.warning(
                        "Retry %d/%d for %s in %.1fs — %s",
                        attempt + 1, max_retries, func.__name__, total_delay, exc,
                    )
                    await asyncio.sleep(total_delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
