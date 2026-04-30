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


def _resolve_role_ids(workflow: dict[str, Any], role: str) -> list[str]:
    """Resolve a role to a list of node IDs.

    Single-string roles wrap to a one-element list; list-valued roles pass
    through. Used by `inject_video` for the `seed` role, which on the Wan
    2.2 14B I2V workflow points at two samplers (HIGH + LOW) that must
    share a coupled seed for determinism across the MoE handoff.
    """
    roles = workflow.get("_meta", {}).get("role", {})
    if role not in roles:
        raise KeyError(f"workflow _meta.role missing required role: {role!r}")
    val = roles[role]
    return [val] if isinstance(val, str) else list(val)


def _apply_ref(
    workflow: dict[str, Any],
    *,
    ref_path: str | None,
    loader_role: str,
    apply_role: str,
    bypass_field: str,
) -> None:
    """Wire a face/pose/depth reference into the workflow, mutating in place.

    Looks up the LoadImage node id via `loader_role` and the apply node id
    (IPAdapterFaceIDApply or ControlNetApplyAdvanced) via `apply_role`.
    If either role is unregistered in `_meta.role`, this is a no-op so
    pre-S7.1.B workflows still inject cleanly.

    `ref_path is None` -> set apply node's `bypass_field` to 0.0
        (weight=0 for IP-Adapter; strength=0 for ControlNet) -- the apply
        node still runs but contributes nothing, equivalent to "skip this
        conditioning for this candidate."
    `ref_path is str` -> validate the file exists on disk (raises
        FileNotFoundError early rather than letting ComfyUI fail
        mid-sample) and write the path into the LoadImage node's `image`
        input.
    """
    roles = workflow.get("_meta", {}).get("role", {})
    if loader_role not in roles or apply_role not in roles:
        return
    apply_id = roles[apply_role]
    if ref_path is None:
        workflow[apply_id]["inputs"][bypass_field] = 0.0
        return
    if not Path(ref_path).exists():
        raise FileNotFoundError(
            f"{loader_role} path does not exist: {ref_path}"
        )
    loader_id = roles[loader_role]
    workflow[loader_id]["inputs"]["image"] = ref_path


def inject(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    output_prefix: str = "flux_dev",
    face_ref_path: str | None = None,
    depth_ref_path: str | None = None,
    pose_ref_path: str | None = None,
) -> dict[str, Any]:
    """Return a new workflow dict with the variable fields swapped in.

    Required _meta.role entries: positive_prompt, negative_prompt,
    empty_latent, sampler, save_image. Raises KeyError if any are missing.

    Optional _meta.role entries: model_sampling_flux, ipadapter_apply,
    controlnet_depth_apply, controlnet_pose_apply (with their
    corresponding *_loader / *_ref_image entries).

    Reference path kwargs (S7.1.B1.4):
      face_ref_path, depth_ref_path, pose_ref_path: str | None.
        None -> bypass the corresponding apply node (weight/strength=0).
        str  -> validated to exist on disk; written into the LoadImage
                node's `image` input. Raises FileNotFoundError if missing.
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
    # Reference conditioning: face (IPAdapter) + depth + pose (ControlNet).
    _apply_ref(
        out, ref_path=face_ref_path,
        loader_role="face_ref_image",
        apply_role="ipadapter_apply",
        bypass_field="weight",
    )
    _apply_ref(
        out, ref_path=depth_ref_path,
        loader_role="depth_ref_image",
        apply_role="controlnet_depth_apply",
        bypass_field="strength",
    )
    _apply_ref(
        out, ref_path=pose_ref_path,
        loader_role="pose_ref_image",
        apply_role="controlnet_pose_apply",
        bypass_field="strength",
    )
    return out


REQUIRED_VIDEO_ROLES = (
    "image_in",
    "prompt",
    "seed",
    "video_out",
)


def inject_video(
    workflow: dict[str, Any],
    *,
    image_in: str,
    prompt: str,
    seed: int,
    output_prefix: str,
    width: int | None = None,
    height: int | None = None,
    frame_count: int | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    """Return a new workflow dict with Wan 2.2 I2V variable fields swapped in.

    Required _meta.role entries: image_in, prompt, seed, video_out.
      seed may be either a single node ID (str) or a list of IDs; the list
      form is used by Wan 2.2 14B I2V's two-sampler MoE chain (HIGH + LOW
      must share a coupled seed for determinism across the samples handoff).
      The prompt-target node's `positive_prompt` input is overwritten
      (matches WanVideoTextEncode's schema).

    Optional _meta.role entries (each mutated only if present in roles):
      width, height -- typically WanVideoImageToVideoEncode's widget inputs.
      frame_count   -- the same encoder's `num_frames` input (frames per clip).
      fps           -- VHS_VideoCombine node's `frame_rate` input.

    image_in is the server-side filename returned by ComfyClient.upload_image
    (NOT a local path); the LoadImage node references files by name relative
    to ComfyUI's `input/` directory.
    """
    out = copy.deepcopy(workflow)
    image_id = _resolve_role(out, "image_in")
    prompt_id = _resolve_role(out, "prompt")
    video_id = _resolve_role(out, "video_out")
    out[image_id]["inputs"]["image"] = image_in
    out[prompt_id]["inputs"]["positive_prompt"] = prompt
    # seed is a list-capable role: same value to every sampler in the chain
    # (Wan 2.2 14B I2V uses two WanVideoSampler nodes that must share a seed).
    for sid in _resolve_role_ids(out, "seed"):
        out[sid]["inputs"]["seed"] = seed
    out[video_id]["inputs"]["filename_prefix"] = output_prefix

    roles = out.get("_meta", {}).get("role", {})
    if width is not None and "width" in roles:
        out[roles["width"]]["inputs"]["width"] = width
    if height is not None and "height" in roles:
        out[roles["height"]]["inputs"]["height"] = height
    if frame_count is not None and "frame_count" in roles:
        out[roles["frame_count"]]["inputs"]["num_frames"] = frame_count
    if fps is not None and "fps" in roles:
        out[roles["fps"]]["inputs"]["frame_rate"] = fps

    return out
