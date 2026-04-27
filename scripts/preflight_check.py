"""Pre-flight sanity check for vast.ai box readiness.

Runs after `vast_setup.sh` provisions the box, before any GPU sampling.
Five checks, fail-fast on first failure -- exit nonzero with clear stderr.

Usage:
    python scripts/preflight_check.py [--workflow path/to/json]
"""
from __future__ import annotations

import argparse
import hashlib  # noqa: F401 (used in Task 17 for workflow signature)
import json
import os
import sys
import time
from pathlib import Path

import httpx

REQUIRED_ROLES = frozenset({
    "positive_prompt", "negative_prompt", "empty_latent", "sampler",
    "save_image", "model_sampling_flux", "flux_guidance",
})


def _check_hf_token(token: str) -> tuple[bool, str]:
    """GET 1KB range from gated FLUX.1-dev safetensors.

    Whoami can succeed while a token lacks gated download access; this is
    the real test. 200 or 206 -> token works; 401/403 -> missing access.
    """
    url = ("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/"
           "flux1-dev.safetensors")
    headers = {"Authorization": f"Bearer {token}", "Range": "bytes=0-1023"}
    try:
        with httpx.Client(follow_redirects=True, timeout=10) as client:
            resp = client.get(url, headers=headers)
        if resp.status_code in (200, 206):
            return True, f"HF token OK (got {resp.status_code} {len(resp.content)}B)"
        return False, (
            f"HF token resolve failed: {resp.status_code} {resp.text[:200]}"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"HF token resolve raised: {exc!r}"


def _check_workflow_json(path: Path) -> tuple[bool, str]:
    """Load JSON; verify _meta.role tags reference valid nodes."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"workflow JSON load failed: {exc!r}"
    roles = data.get("_meta", {}).get("role", {})
    missing = REQUIRED_ROLES - set(roles)
    if missing:
        return False, f"workflow missing roles: {sorted(missing)}"
    for role, node_id in roles.items():
        if node_id not in data:
            return False, f"role '{role}' -> node '{node_id}' not in workflow"
    return True, f"workflow JSON OK ({len(data)} nodes, {len(roles)} roles)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow", type=Path,
        default=Path("config/workflows/flux_dev_keyframe.json"),
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not in env", file=sys.stderr)
        return 2

    checks = [
        ("HF token resolve",      lambda: _check_hf_token(hf_token)),
        ("Workflow JSON",         lambda: _check_workflow_json(args.workflow)),
    ]
    for label, fn in checks:
        t0 = time.monotonic()
        passed, msg = fn()
        dt = time.monotonic() - t0
        marker = "OK " if passed else "FAIL"
        print(f"  [{marker}] {label}: {msg}  ({dt*1000:.0f}ms)", flush=True)
        if not passed:
            return 1

    print("preflight OK (partial: HF token + workflow only)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
