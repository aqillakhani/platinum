"""PoseDepthMapStage: per-scene pose + depth preprocessor outputs.

For each scene whose composition_notes is non-empty, this stage produces
two preprocessor outputs that condition keyframe rendering:

  1. A low-res Flux prerender (8 steps, 512x896) of the composition_notes
     prompt. Cheap to generate; serves as the geometric anchor.
  2. DWPose preprocessor output -> scene_NNN/_pose.png. Captures the
     intended subject blocking, gestures, and hand positions.
  3. DepthAnythingV2 preprocessor output -> scene_NNN/_depth.png.
     Captures the intended foreground/background spatial layout.

The Stage writes scene.pose_ref_path and scene.depth_ref_path to the
new outputs. keyframe_generator (Layer B5 keyframe wiring, deferred)
feeds those paths into inject(pose_ref_path=, depth_ref_path=) so the
ControlNet apply nodes can condition the final render.

is_complete() returns True iff every scene with composition_notes has
BOTH pose_ref_path AND depth_ref_path set to a path that exists on
disk -- partial state is "resume" rather than "skip", so an
interrupted run picks up cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage


class PoseDepthMapStage(Stage):
    """Stage skeleton for per-scene pose+depth map generation.

    B5.3 fills in run() with the prerender + preprocessor pass.
    """

    name: ClassVar[str] = "pose_depth_maps"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        # B5.3 wires this; skeleton stub for now.
        raise NotImplementedError(
            "PoseDepthMapStage.run is wired in S7.1.B5.3"
        )

    def _scenes_needing_refs(self, story: Story) -> list[Any]:
        """Scenes whose composition_notes is non-empty -- those are the
        scenes that need pose+depth preprocessor outputs. Scenes with no
        composition_notes (transitional shots; dialogue-only) skip the
        whole stage.
        """
        return [
            scene
            for scene in story.scenes
            if scene.composition_notes
        ]

    def is_complete(self, story: Story) -> bool:
        """True iff every composition_notes scene has BOTH refs on disk.

        - No composition_notes scenes -> True (nothing to do).
        - At least one such scene missing either ref path -> False.
        - At least one ref path that doesn't exist on disk -> False.
        - Otherwise -> True (resume can skip).

        Path lookup accepts absolute paths AND repo-relative paths
        under data/stories/<story.id>/ for backwards compatibility.
        """
        targets = self._scenes_needing_refs(story)
        if not targets:
            return True
        for scene in targets:
            for path_str in (scene.pose_ref_path, scene.depth_ref_path):
                if not path_str:
                    return False
                if Path(path_str).exists():
                    continue
                rel = Path("data/stories") / story.id / path_str
                if not rel.exists():
                    return False
        return True
