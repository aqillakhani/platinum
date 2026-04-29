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

import copy
from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage

# Negative prompt for the prerender pass. The prerender exists only as
# a geometric anchor for DWPose + DepthAnythingV2; visual quality is
# moot. Generic noise filters keep the layout legible to the
# preprocessors.
_PRERENDER_NEGATIVE_PROMPT = "cartoon, anime, plastic, blurry, low quality"

# Low-res prerender dims. 512x896 is a 9:16 portrait at half-res of the
# 768x1344 final. DWPose + DepthAnythingV2 are scale-invariant; the
# prerender just needs the layout right.
_PRERENDER_WIDTH = 512
_PRERENDER_HEIGHT = 896
_PRERENDER_STEPS = 8


class PoseDepthMapStage(Stage):
    """Stage skeleton for per-scene pose+depth map generation.

    B5.3 fills in run() with the prerender + preprocessor pass.
    """

    name: ClassVar[str] = "pose_depth_maps"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        """Walk composition_notes scenes; produce pose + depth refs each.

        Per scene:
          1. _prerender: Flux call with composition_notes as prompt at
             512x896, 8 steps. Output: <scene_dir>/_prerender.png.
          2. _preprocess: load pose_depth_map.json, point its LoadImage
             at the prerender, submit to comfy. Two SaveImage outputs
             (pose + depth) emitted to <scene_dir>/_pose.png and
             _depth.png.
          3. Set scene.pose_ref_path and scene.depth_ref_path.

        Returns artifacts dict with the indices of scenes that were
        prepared this run (excludes scenes that didn't need refs).
        """
        targets = self._scenes_needing_refs(story)
        if not targets:
            return {"prepared_scenes": []}

        prepared: list[int] = []
        for scene in targets:
            # B5.4 resume: skip scenes whose pose+depth refs are both set
            # AND point at files that exist on disk. The whole stage is
            # idempotent over re-runs as long as the prior outputs are
            # still on disk; any missing or unset ref triggers a re-pass.
            if (
                scene.pose_ref_path
                and scene.depth_ref_path
                and Path(scene.pose_ref_path).exists()
                and Path(scene.depth_ref_path).exists()
            ):
                continue
            prerender_path = await self._prerender(scene, story, ctx)
            pose_path, depth_path = await self._preprocess(
                prerender_path, scene, story, ctx
            )
            scene.pose_ref_path = str(pose_path)
            scene.depth_ref_path = str(depth_path)
            prepared.append(scene.index)
        return {"prepared_scenes": prepared}

    async def _prerender(
        self,
        scene: Scene,
        story: Story,
        ctx: PipelineContext,
    ) -> Path:
        """Generate a fast Flux prerender of composition_notes.

        Reuses inject() against flux_dev_keyframe.json with the bare
        defaults (face/depth/pose ref kwargs unset -> apply weights/strengths
        forced to 0 -- B1.4) and overrides the sampler's `steps` to 8 for
        speed. The prerender is geometric scaffolding for the preprocessors,
        not a final-quality image.
        """
        from platinum.utils.comfyui import HttpComfyClient
        from platinum.utils.workflow import inject, load_workflow

        test_overrides = ctx.config.settings.get("test", {})
        comfy = test_overrides.get("comfy_client") or HttpComfyClient(
            host=ctx.config.settings.get("comfyui", {}).get(
                "host", "http://localhost:8188"
            ),
        )

        out_dir = (
            ctx.config.story_dir(story.id)
            / "keyframes"
            / f"scene_{scene.index:03d}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        wf_template = load_workflow(
            "flux_dev_keyframe", config_dir=ctx.config.config_dir
        )
        prompt = scene.composition_notes or ""
        wf = inject(
            wf_template,
            prompt=prompt,
            negative_prompt=_PRERENDER_NEGATIVE_PROMPT,
            seed=scene.index * 1000 + 999,
            width=_PRERENDER_WIDTH,
            height=_PRERENDER_HEIGHT,
            output_prefix=f"scene_{scene.index:03d}_prerender",
        )
        sampler_id = wf["_meta"]["role"]["sampler"]
        wf[sampler_id]["inputs"]["steps"] = _PRERENDER_STEPS

        prerender_path = out_dir / "_prerender.png"
        await comfy.generate_image(workflow=wf, output_path=prerender_path)
        return prerender_path

    async def _preprocess(
        self,
        prerender_path: Path,
        scene: Scene,
        story: Story,
        ctx: PipelineContext,
    ) -> tuple[Path, Path]:
        """Run pose_depth_map.json against the prerender; return (pose, depth).

        Submits the same workflow signature twice. FakeComfyClient.responses
        rotates through its list (test path) so [pose_fixture, depth_fixture]
        yields each in order. Production HttpComfyClient currently picks the
        first SaveImage output; multi-output download support is a follow-on
        before the verify run.
        """
        from platinum.utils.comfyui import HttpComfyClient
        from platinum.utils.workflow import load_workflow

        test_overrides = ctx.config.settings.get("test", {})
        comfy = test_overrides.get("comfy_client") or HttpComfyClient(
            host=ctx.config.settings.get("comfyui", {}).get(
                "host", "http://localhost:8188"
            ),
        )

        out_dir = (
            ctx.config.story_dir(story.id)
            / "keyframes"
            / f"scene_{scene.index:03d}"
        )

        wf_template = load_workflow(
            "pose_depth_map", config_dir=ctx.config.config_dir
        )
        wf = copy.deepcopy(wf_template)
        image_id = wf["_meta"]["role"]["image_input"]
        wf[image_id]["inputs"]["image"] = str(prerender_path)

        pose_path = out_dir / "_pose.png"
        depth_path = out_dir / "_depth.png"
        await comfy.generate_image(workflow=wf, output_path=pose_path)
        await comfy.generate_image(workflow=wf, output_path=depth_path)
        return pose_path, depth_path

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
