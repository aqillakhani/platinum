"""Fixture recorder/replayer for offline LLM tests.

Replay mode: read JSON from disk, return its `response` field.
Record mode: call the live backend, save (request, response) to disk.

Path scheme: tests/fixtures/anthropic/<stage>/<test_name>__<attempt>.json
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

LiveCall = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class FixtureMissingError(FileNotFoundError):
    """Raised in replay mode when no fixture matches the request."""


class FixtureRecorder:
    """Per-call recorder. One instance per (stage, test, attempt).

    In replay mode, returns the saved response. In record mode, invokes
    `live` and writes the (request, response) pair to disk on the fly.
    """

    def __init__(
        self,
        *,
        path: Path,
        mode: Literal["replay", "record"] = "replay",
        live: LiveCall | None = None,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.live = live

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "replay":
            if not self.path.exists():
                raise FixtureMissingError(
                    f"No fixture at {self.path}. Re-run with PLATINUM_RECORD_FIXTURES=1 "
                    "to capture against the live API."
                )
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data["response"]

        # record mode
        if self.live is None:
            raise RuntimeError("FixtureRecorder(mode='record') requires a `live` callable.")
        response = await self.live(request)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"request": request, "response": response}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return response
