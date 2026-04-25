"""ComfyUI client.

Production: HttpComfyClient -- async httpx wrapper around ComfyUI's REST API.
Tests:      FakeComfyClient  -- deterministic, copies prebaked fixture PNGs to
            the requested output_path.

`generate_image` takes an *already-injected* workflow dict. Workflow JSON
schema knowledge lives in utils/workflow.py; this module only handles
transport (or fake transport).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def workflow_signature(workflow: dict[str, Any]) -> str:
    """SHA256 of canonical JSON; used by FakeComfyClient response keying."""
    canonical = json.dumps(workflow, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@runtime_checkable
class ComfyClient(Protocol):
    """Talk to a ComfyUI server. Workflow injection is the caller's job."""

    async def generate_image(
        self,
        *,
        workflow: dict[str, Any],
        output_path: Path,
    ) -> Path: ...

    async def upload_image(self, image_path: Path) -> str: ...

    async def health_check(self) -> bool: ...


@dataclass
class FakeComfyClient:
    """Deterministic ComfyClient for tests.

    `responses` maps a workflow signature to a list of fixture PNG paths.
    Each generate_image call rotates through the list; once exhausted, the
    last entry is reused. Unknown signatures raise KeyError.

    `calls` records every generate_image call as a dict with
    {"workflow_signature": str, "output_path": Path} so tests can assert
    on call count, ordering, and the workflow each candidate received.
    """

    responses: dict[str, list[Path]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursors: dict[str, int] = field(default_factory=dict)

    async def generate_image(
        self,
        *,
        workflow: dict[str, Any],
        output_path: Path,
    ) -> Path:
        sig = workflow_signature(workflow)
        if sig not in self.responses:
            raise KeyError(
                f"FakeComfyClient has no response configured for workflow signature {sig[:12]}..."
            )
        sources = self.responses[sig]
        if not sources:
            raise ValueError(f"FakeComfyClient.responses[{sig[:12]}...] is empty")
        cursor = self._cursors.get(sig, 0)
        src = sources[min(cursor, len(sources) - 1)]
        self._cursors[sig] = cursor + 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, output_path)
        self.calls.append({"workflow_signature": sig, "output_path": output_path})
        return output_path

    async def upload_image(self, image_path: Path) -> str:
        return Path(image_path).name

    async def health_check(self) -> bool:
        return True
