"""ComfyUI workflow JSON loader + injector.

Workflows are static JSON files under config/workflows/. Each file has a
top-level `_meta.role` block mapping role names ("positive_prompt",
"sampler", ...) to node IDs, so `inject` can mutate the right node-input
fields without depending on node-id numbering staying stable.

`inject` is a pure function: deepcopy the input, mutate the copy, return.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REQUIRED_ROLES = (
    "positive_prompt",
    "negative_prompt",
    "empty_latent",
    "sampler",
    "save_image",
)


def load_workflow(name: str, *, config_dir: Path) -> dict[str, Any]:
    """Load `<config_dir>/workflows/<name>.json`.

    Raises FileNotFoundError if the named file is missing.
    """
    path = Path(config_dir) / "workflows" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_role(workflow: dict[str, Any], role: str) -> str:
    roles = workflow.get("_meta", {}).get("role", {})
    if role not in roles:
        raise KeyError(f"workflow _meta.role missing required role: {role!r}")
    return roles[role]


def inject(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    output_prefix: str = "flux_dev",
) -> dict[str, Any]:
    """Return a new workflow dict with the variable fields swapped in.

    Required _meta.role entries: positive_prompt, negative_prompt,
    empty_latent, sampler, save_image. Raises KeyError if any are missing.

    Optional _meta.role entries: model_sampling_flux.
    """
    out = copy.deepcopy(workflow)
    pos_id = _resolve_role(out, "positive_prompt")
    neg_id = _resolve_role(out, "negative_prompt")
    latent_id = _resolve_role(out, "empty_latent")
    sampler_id = _resolve_role(out, "sampler")
    save_id = _resolve_role(out, "save_image")
    out[pos_id]["inputs"]["text"] = prompt
    out[neg_id]["inputs"]["text"] = negative_prompt
    out[latent_id]["inputs"]["width"] = width
    out[latent_id]["inputs"]["height"] = height
    out[sampler_id]["inputs"]["seed"] = seed
    out[save_id]["inputs"]["filename_prefix"] = output_prefix
    # Optional: ModelSamplingFlux carries its own width/height; keep aligned with empty_latent.
    msf_roles = out.get("_meta", {}).get("role", {})
    if "model_sampling_flux" in msf_roles:
        msf_id = msf_roles["model_sampling_flux"]
        out[msf_id]["inputs"]["width"] = width
        out[msf_id]["inputs"]["height"] = height
    return out
