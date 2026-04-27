"""Reset keyframe state for specified scenes in a story.

Needed because the orchestrator skips scenes whose keyframe_path is set.
After S6.2 Phase 2's dirty state, scenes 1/8/16 carry stale keyframe_path
values pointing at the disproven hypothesis's outputs. To re-run those
scenes against the rebuilt workflow, clear the path or append a failed
StageRun -- this script is the cleaner option.

Usage:
    python scripts/reset_scene_keyframes.py STORY_ID --scenes "1,8,16"
                                            [--delete-files]
                                            [--story-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def reset_scenes(
    *,
    story_id: str,
    scenes: list[int],
    story_dir: Path,
    delete_files: bool,
    reset_stage: bool = False,
) -> int:
    story_path = Path(story_dir) / story_id / "story.json"
    if not story_path.exists():
        print(f"ERROR: {story_path} not found", file=sys.stderr)
        return 2
    data = json.loads(story_path.read_text())
    keyframes_dir = Path(story_dir) / story_id / "keyframes"
    target_set = set(scenes)
    reset_count = 0
    for scene in data.get("scenes", []):
        if scene.get("index") not in target_set:
            continue
        scene["keyframe_path"] = None
        scene["keyframe_candidates"] = []
        scene["keyframe_scores"] = []
        validation = scene.get("validation", {})
        validation.pop("keyframe_anatomy", None)
        validation.pop("keyframe_selected_via_fallback", None)
        scene["validation"] = validation
        reset_count += 1
        if delete_files:
            scene_subdir = keyframes_dir / f"scene_{scene['index']:03d}"
            if scene_subdir.exists():
                for png in scene_subdir.glob("candidate_*.png"):
                    png.unlink()
    if reset_stage:
        before = len(data.get("stages", []))
        data["stages"] = [
            s for s in data.get("stages", [])
            if s.get("stage") != "keyframe_generator"
        ]
        stripped = before - len(data["stages"])
        print(f"stripped {stripped} keyframe_generator StageRun(s)", flush=True)
    tmp = story_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(story_path)
    print(f"reset {reset_count} scenes in {story_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("story_id")
    parser.add_argument("--scenes", required=True,
                        help="comma-separated scene indices (e.g. 1,8,16)")
    parser.add_argument("--delete-files", action="store_true",
                        help="also delete candidate_*.png from disk")
    parser.add_argument("--reset-stage", action="store_true",
                        help="Also strip keyframe_generator StageRuns from "
                             "story.json so the orchestrator re-enters "
                             "generate() instead of short-circuiting on "
                             "Stage.is_complete().")
    parser.add_argument("--story-dir", type=Path, default=Path("data/stories"))
    args = parser.parse_args()

    scenes = sorted({int(s.strip()) for s in args.scenes.split(",") if s.strip()})
    return reset_scenes(
        story_id=args.story_id,
        scenes=scenes,
        story_dir=args.story_dir,
        delete_files=args.delete_files,
        reset_stage=args.reset_stage,
    )


if __name__ == "__main__":
    sys.exit(main())
