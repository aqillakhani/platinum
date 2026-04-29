"""ComfyUI client.

Production: HttpComfyClient -- async httpx wrapper around ComfyUI's REST API.
Tests:      FakeComfyClient  -- deterministic, copies prebaked fixture PNGs to
            the requested output_path.

`generate_image` takes an *already-injected* workflow dict. Workflow JSON
schema knowledge lives in utils/workflow.py; this module only handles
transport (or fake transport).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


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
    {"workflow_signature": str, "output_path": Path, "workflow": dict}
    so tests can assert on call count, ordering, and the workflow each
    candidate received. The "workflow" entry is a deepcopy snapshot so
    later mutations cannot retroactively change asserted values.
    """

    responses: dict[str, list[Path]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursors: dict[str, int] = field(default_factory=dict)

    @property
    def submitted_workflows(self) -> list[dict[str, Any]]:
        """Per-call workflow snapshots in submission order."""
        return [c["workflow"] for c in self.calls]

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
        self.calls.append({
            "workflow_signature": sig,
            "output_path": output_path,
            "workflow": copy.deepcopy(workflow),
        })
        return output_path

    async def upload_image(self, image_path: Path) -> str:
        return Path(image_path).name

    async def health_check(self) -> bool:
        return True


class HttpComfyClient:
    """Async httpx-based ComfyUI client.

    `transport` plumbs through to httpx.AsyncClient; tests pass MockTransport
    for unit-level wire-shape verification without a Fake.

    `poll_interval` defaults to 2.0s for production; tests pass 0.0 to get
    instant polling.
    """

    def __init__(
        self,
        host: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_polls: int = 600,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.transport = transport
        self.max_polls = max_polls

    def _client(self, *, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.host,
            timeout=timeout or self.timeout,
            transport=self.transport,
        )

    async def health_check(self) -> bool:
        try:
            async with self._client(timeout=10.0) as client:
                resp = await client.get("/system_stats")
                return resp.status_code == 200
        except Exception:
            return False

    async def upload_image(self, image_path: Path) -> str:
        path = Path(image_path)
        async with self._client(timeout=60.0) as client:
            with path.open("rb") as fh:
                resp = await client.post(
                    "/upload/image",
                    files={"image": (path.name, fh, "image/png")},
                    data={"overwrite": "true"},
                )
                resp.raise_for_status()
                payload = resp.json()
                return payload.get("name", path.name)

    async def generate_image(
        self,
        *,
        workflow: dict[str, Any],
        output_path: Path,
    ) -> Path:
        prompt_id = await self._submit(workflow)
        result = await self._poll(prompt_id)
        await self._download(result, output_path)
        return output_path

    async def _submit(self, workflow: dict[str, Any]) -> str:
        async with self._client(timeout=30.0) as client:
            # Strip the platinum-internal `_meta` block before posting; ComfyUI
            # treats every top-level key as a node and rejects keys with no
            # `class_type` ("invalid prompt: missing_node_type Node 'ID #_meta'").
            prompt_workflow = {k: v for k, v in workflow.items() if k != "_meta"}
            payload = {"prompt": prompt_workflow, "client_id": str(uuid.uuid4())}
            resp = await client.post("/prompt", json=payload)
            resp.raise_for_status()
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"No prompt_id in /prompt response: {data!r}")
            logger.info("submitted ComfyUI workflow, prompt_id=%s", prompt_id)
            return prompt_id

    async def _poll(self, prompt_id: str) -> dict[str, Any]:
        async with self._client(timeout=30.0) as client:
            for _attempt in range(self.max_polls):
                resp = await client.get(f"/history/{prompt_id}")
                if resp.status_code == 200:
                    body = resp.json()
                    if prompt_id in body:
                        result = body[prompt_id]
                        status = result.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI workflow error: {result!r}")
                        if status.get("completed"):
                            return result
                if self.poll_interval > 0:
                    await asyncio.sleep(self.poll_interval)
            raise RuntimeError(
                f"Timed out polling /history/{prompt_id} after {self.max_polls} attempts"
            )

    async def _download(self, result: dict[str, Any], output_path: Path) -> None:
        outputs = result.get("outputs", {})
        for _node_id, node_output in outputs.items():
            files = (
                node_output.get("images")
                or node_output.get("gifs")
                or node_output.get("videos")
                or []
            )
            if not files:
                continue
            file_info = files[0]
            params = {
                "filename": file_info.get("filename", ""),
                "subfolder": file_info.get("subfolder", ""),
                "type": file_info.get("type", "output"),
            }
            async with self._client() as client:
                resp = await client.get("/view", params=params)
                resp.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(resp.content)
                logger.info(
                    "downloaded ComfyUI output to %s (%d bytes)",
                    output_path,
                    len(resp.content),
                )
                return
        raise RuntimeError(f"No output files found in ComfyUI result. Keys: {list(outputs.keys())}")
