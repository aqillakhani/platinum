"""Bright-probe smoke driver: feeds a single hand-authored prompt through
generate_for_scene against a synthetic Scene, prints per-candidate stats.

Bypasses Story / StageRun / orchestrator -- testing only the gen + score +
select layer with arbitrary prompts. Used by the S6.2 runbook's bright-
probe step + future sessions when ad-hoc prompts need quick validation.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


def _scene_index_for_label(label: str) -> int:
    """Deterministic mapping label -> scene_index in [0, 10000)."""
    digest = sha256(label.encode()).digest()
    return int.from_bytes(digest[:2], "big") % 10000


@dataclass
class _SyntheticScene:
    """Minimal Scene-shaped object for generate_for_scene's duck-typed interface."""

    index: int
    visual_prompt: str
    negative_prompt: str = ""
    keyframe_path: Path | None = None
    keyframe_candidates: list[Path] = field(default_factory=list)
    keyframe_scores: list[float] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)


async def run(
    *,
    prompt: str,
    label: str,
    negative: str,
    track: str,
    output_dir: Path,
) -> None:
    from platinum.config import Config
    from platinum.pipeline.keyframe_generator import generate_for_scene
    from platinum.utils.aesthetics import RemoteAestheticScorer
    from platinum.utils.comfyui import HttpComfyClient

    cfg = Config()
    track_cfg = cfg.track(track)
    track_visual = dict(track_cfg.get("visual", {}))
    quality_gates = dict(track_cfg.get("quality_gates", {}))

    scene_index = _scene_index_for_label(label)
    scene = _SyntheticScene(
        index=scene_index, visual_prompt=prompt, negative_prompt=negative
    )
    scene_dir = output_dir / label
    scene_dir.mkdir(parents=True, exist_ok=True)

    comfy = HttpComfyClient(host=cfg.settings["comfyui"]["host"])
    scorer = RemoteAestheticScorer(host=cfg.settings["aesthetics"]["host"])
    try:
        report = await generate_for_scene(
            scene,
            track_visual=track_visual,
            quality_gates=quality_gates,
            comfy=comfy,
            scorer=scorer,
            output_dir=scene_dir,
            config_dir=cfg.config_dir,
        )
    finally:
        await scorer.aclose()

    print(f"label={label} scene_index={scene_index}")
    print(
        f"  {'cand':>4} {'meanRGB':>7} {'score':>5} "
        f"{'bright':>6} {'anat':>4} {'sel':>3} {'fb':>2}"
    )
    for i, _path in enumerate(report.candidates):
        sel = "yes" if i == report.selected_index else "-"
        fb = "yes" if (i == report.selected_index and report.selected_via_fallback) else "-"
        print(
            f"  {i:>4} {'?':>7} {report.scores[i]:>5.2f} "
            f"{('ok' if report.brightness_passed[i] else 'FAIL'):>6} "
            f"{('ok' if report.anatomy_passed[i] else 'FAIL'):>4} {sel:>3} {fb:>2}"
        )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Bright-probe smoke for keyframe quality validation."
    )
    parser.add_argument("--prompt", required=True, help="visual_prompt for the synthetic scene")
    parser.add_argument("--label", required=True, help="output dir name + scene_index seed")
    parser.add_argument("--negative", default="", help="optional negative_prompt")
    parser.add_argument(
        "--track",
        default="atmospheric_horror",
        help="track for visual + quality_gates config",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/smoke"),
        help="parent dir for label/candidate_*.png",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            prompt=args.prompt,
            label=args.label,
            negative=args.negative,
            track=args.track,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    _main()
