"""Pre-flight sanity check for vast.ai box readiness.

Runs after `vast_setup.sh` provisions the box, before any GPU sampling.
Five checks, fail-fast on first failure -- exit nonzero with clear stderr.

Usage:
    python scripts/preflight_check.py [--workflow path/to/json]
"""
from __future__ import annotations

import argparse
import hashlib
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


def _check_comfyui_alive(host: str) -> tuple[bool, str]:
    """GET /system_stats to verify ComfyUI is responding."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{host}/system_stats")
        if resp.status_code != 200:
            return False, f"ComfyUI {host}: {resp.status_code}"
        stats = resp.json()
        gpu_name = (stats.get("devices") or [{}])[0].get("name", "?")
        return True, f"ComfyUI alive at {host} (GPU={gpu_name})"
    except Exception as exc:  # noqa: BLE001
        return False, f"ComfyUI unreachable at {host}: {exc!r}"


def _check_score_server_alive(host: str) -> tuple[bool, str]:
    """GET /health to verify score-server is responding."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{host}/health")
        if resp.status_code != 200:
            return False, f"score-server {host}: {resp.status_code}"
        return True, f"score-server alive at {host}"
    except Exception as exc:  # noqa: BLE001
        return False, f"score-server unreachable at {host}: {exc!r}"


def _workflow_signature(path: Path) -> str:
    """sha256 of canonical-form JSON (sorted keys); short hex prefix."""
    data = json.loads(Path(path).read_text())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:12]


WAN_REQUIRED_ROLES = frozenset({"image_in", "prompt", "seed", "video_out"})

# Filenames + minimum sizes for the Wan 2.2 weights downloaded by
# vast_setup.sh. HIGH/LOW/VAE come from Kijai/WanVideo_comfy single-file
# repackages (bf16 safetensors). UMT5 comes from Wan-AI/Wan2.2-I2V-A14B
# (the original .pth). Real on-disk sizes are roughly 28.6/28.6/1.4/11.4 GB.
# Min sizes are deliberately loose (100MB floor) -- they catch "file
# wasn't downloaded at all / is a 0-byte stub" rather than partial
# truncations; ComfyUI's loader will fail loudly on a truly broken file.
WAN_WEIGHT_FILES = (
    ("diffusion_models", "Wan2_2-I2V-A14B-HIGH_bf16.safetensors", 100_000_000),
    ("diffusion_models", "Wan2_2-I2V-A14B-LOW_bf16.safetensors", 100_000_000),
    # Wan 2.1 VAE: the Wan 2.2 14B I2V experts inherit the 2.1 VAE encoder.
    # All four 2.2 14B reference workflows in the WanVideoWrapper example
    # tarball load this file, not Wan2_2_VAE (which is for the 2.2 5B family).
    ("vae", "Wan2_1_VAE_bf16.safetensors", 100_000_000),
    ("text_encoders", "umt5_xxl.pth", 100_000_000),
)


def _check_wan_workflow_json(path: Path) -> tuple[bool, str]:
    """Load Wan workflow JSON; verify _meta.role tags reference valid nodes."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"wan workflow JSON load failed: {exc!r}"
    roles = data.get("_meta", {}).get("role", {})
    missing = WAN_REQUIRED_ROLES - set(roles)
    if missing:
        return False, f"wan workflow missing roles: {sorted(missing)}"
    for role, node_id in roles.items():
        if role in WAN_REQUIRED_ROLES and node_id not in data:
            return False, f"wan role '{role}' -> node '{node_id}' not in workflow"
    return True, f"wan workflow OK (roles: {sorted(WAN_REQUIRED_ROLES)})"


def _check_wan_weights(models_dir: Path) -> tuple[bool, str]:
    """Verify Wan 2.2 weight files exist and are at least minimal size."""
    missing: list[str] = []
    too_small: list[str] = []
    for subdir, filename, min_size in WAN_WEIGHT_FILES:
        p = Path(models_dir) / subdir / filename
        if not p.exists():
            missing.append(str(p))
            continue
        if p.stat().st_size < min_size:
            too_small.append(f"{p} ({p.stat().st_size}B < {min_size}B)")
    if missing:
        return False, f"wan weights missing: {missing}"
    if too_small:
        return False, f"wan weights too small (download incomplete?): {too_small}"
    return True, "wan weights OK (4 files present, sizes plausible)"


def _check_wan_extension_importable() -> tuple[bool, str]:
    """Confirm the ComfyUI-WanVideoWrapper extension is on disk.

    On the box this is just a path check -- we don't actually run its
    node registration here.
    """
    extension_path = Path("/workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper")
    if not extension_path.exists():
        return False, f"WanVideoWrapper not at {extension_path}"
    init_py = extension_path / "__init__.py"
    if not init_py.exists():
        return False, f"WanVideoWrapper __init__.py missing at {init_py}"
    return True, f"WanVideoWrapper present at {extension_path}"


def _detect_workflow_mode(path: Path) -> str:
    """Return 'wan' if the workflow's _meta.role advertises Wan-only roles,
    else 'flux'. On any load failure, default to 'flux' so the Flux validator
    runs and surfaces the load error with its existing message."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return "flux"
    roles = set(data.get("_meta", {}).get("role", {}).keys())
    return "wan" if "video_out" in roles or "image_in" in roles else "flux"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow", type=Path,
        default=Path("config/workflows/flux_dev_keyframe.json"),
    )
    parser.add_argument(
        "--wan-models-dir", type=Path,
        default=Path("/workspace/ComfyUI/models"),
        help="ComfyUI models root (used only for Wan workflows).",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    comfyui_host = os.environ.get("PLATINUM_COMFYUI_HOST", "")
    aesthetics_host = os.environ.get("PLATINUM_AESTHETICS_HOST", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not in env", file=sys.stderr)
        return 2
    if not comfyui_host:
        print("ERROR: PLATINUM_COMFYUI_HOST not in env", file=sys.stderr)
        return 2
    if not aesthetics_host:
        print("ERROR: PLATINUM_AESTHETICS_HOST not in env", file=sys.stderr)
        return 2

    mode = _detect_workflow_mode(args.workflow)
    if mode == "wan":
        checks = [
            ("HF token resolve",      lambda: _check_hf_token(hf_token)),
            ("Wan workflow JSON",     lambda: _check_wan_workflow_json(args.workflow)),
            ("ComfyUI alive",         lambda: _check_comfyui_alive(comfyui_host)),
            ("Score-server alive",    lambda: _check_score_server_alive(aesthetics_host)),
            ("Wan weights",           lambda: _check_wan_weights(args.wan_models_dir)),
            ("Wan extension",         lambda: _check_wan_extension_importable()),
        ]
    else:
        checks = [
            ("HF token resolve",      lambda: _check_hf_token(hf_token)),
            ("Workflow JSON",         lambda: _check_workflow_json(args.workflow)),
            ("ComfyUI alive",         lambda: _check_comfyui_alive(comfyui_host)),
            ("Score-server alive",    lambda: _check_score_server_alive(aesthetics_host)),
        ]
    for label, fn in checks:
        t0 = time.monotonic()
        passed, msg = fn()
        dt = time.monotonic() - t0
        marker = "OK " if passed else "FAIL"
        print(f"  [{marker}] {label}: {msg}  ({dt*1000:.0f}ms)", flush=True)
        if not passed:
            return 1

    sig = _workflow_signature(args.workflow)
    print(f"  [INFO] workflow signature: {sig}  (mode={mode})", flush=True)
    print("preflight OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
