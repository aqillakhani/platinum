"""CharacterReferenceStage: per-character reference portrait generation.

After visual_prompts assigns scene.character_refs (B2.3), each story has a
set of recurring character names that need an IP-Adapter face reference.
This stage walks that set, generates 3 candidate studio-portrait refs per
character (no IP-Adapter / no ControlNet on the generation -- bare Flux
Dev), and persists the candidates to disk for review.

The stage does NOT auto-pick a ref. Story.characters stays empty until
the user picks a ref via the review UI (B6), which calls
apply_select_character_reference (B6.1) to mutate
Story.characters[name] = picked_path.

is_complete() returns True iff every discovered character has a path in
Story.characters AND that path exists on disk -- partial state is "in
progress" and the stage will resume rather than re-generating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from platinum.models.story import Scene, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage

# Sentinel scene index for character ref generation. Real scenes use
# 1..N (per their position in the story); 999 is reserved for synthetic
# ref scenes so the seed block doesn't collide with anything else.
_REF_SCENE_INDEX = 999

_REF_VISUAL_PROMPT_TEMPLATE = (
    "studio reference portrait of {character}, neutral expression, "
    "three-quarter face turn, even mid-tone studio lighting, "
    "neutral grey background, photorealistic painterly style"
)
_REF_NEGATIVE_PROMPT = (
    "cartoon, anime, plastic, multiple faces, blurred, dark, low quality"
)


class CharacterReferenceStage(Stage):
    """Stage skeleton for per-character ref generation. B4.3 fills in run()."""

    name: ClassVar[str] = "character_references"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        # B4.3 / B4.4 implement candidate generation. Skeleton stub for now.
        raise NotImplementedError(
            "CharacterReferenceStage.run is wired in S7.1.B4.3 / B4.4"
        )

    def _discover_characters(self, story: Story) -> set[str]:
        """Union of all scene.character_refs across the story.

        Empty list / missing field both produce no contribution. The
        result is the closed set of names this stage needs to materialise
        ref portraits for.
        """
        return {
            name
            for scene in story.scenes
            for name in (scene.character_refs or [])
        }

    async def _generate_character_refs(
        self,
        character: str,
        story: Story,
        ctx: PipelineContext,
    ) -> list[Path]:
        """Generate 3 candidate ref portraits for `character`.

        Reuses keyframe_generator.generate_for_scene against a synthetic
        Scene whose visual_prompt is a generic studio-portrait template.
        IP-Adapter / ControlNet conditioning is bypassed via inject()'s
        default `*_ref_path=None` (which sets weight/strength=0 -- B1.4),
        so the generation is bare Flux Dev. Outputs land at
        <story>/references/<character>/candidate_{0,1,2}.png.

        Pulls comfy / scorer / hands_factory from
        ctx.config.settings["test"] so unit tests can inject fakes; falls
        back to HttpComfyClient + RemoteAestheticScorer in production.
        """
        from platinum.pipeline.keyframe_generator import generate_for_scene
        from platinum.utils.aesthetics import RemoteAestheticScorer
        from platinum.utils.comfyui import HttpComfyClient

        test_overrides = ctx.config.settings.get("test", {})
        comfy = test_overrides.get("comfy_client") or HttpComfyClient(
            host=ctx.config.settings.get("comfyui", {}).get(
                "host", "http://localhost:8188"
            ),
        )
        scorer = test_overrides.get("aesthetic_scorer") or RemoteAestheticScorer(
            host=ctx.config.settings.get("aesthetics", {}).get("host", ""),
        )
        mp_hands_factory = test_overrides.get("mp_hands_factory")

        out_dir = ctx.config.story_dir(story.id) / "references" / character
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic = Scene(
            id=f"_ref_{character}",
            index=_REF_SCENE_INDEX,
            narration_text="",
            visual_prompt=_REF_VISUAL_PROMPT_TEMPLATE.format(character=character),
            negative_prompt=_REF_NEGATIVE_PROMPT,
        )

        track_cfg = ctx.config.track(story.track)
        track_visual = dict(track_cfg.get("visual", {}))
        # Force content_gate off for ref generation -- the gate compares against
        # scene narration which doesn't apply to synthetic studio portraits.
        quality_gates = dict(track_cfg.get("quality_gates", {}))
        quality_gates["content_gate"] = "off"

        report = await generate_for_scene(
            synthetic,
            track_visual=track_visual,
            quality_gates=quality_gates,
            comfy=comfy,
            scorer=scorer,
            output_dir=out_dir,
            config_dir=ctx.config.config_dir,
            n_candidates=3,
            width=768,
            height=1344,
            mp_hands_factory=mp_hands_factory,
        )
        return list(report.candidates)

    def is_complete(self, story: Story) -> bool:
        """True iff every discovered character has an existing ref on disk.

        - No discovered characters -> True (nothing to do).
        - At least one discovered name not present in Story.characters -> False.
        - Every name has a path, but a path does not exist on disk -> False.
        - Otherwise -> True (resume can skip this stage).

        Path resolution accepts both repo-relative paths (e.g.
        "data/stories/<id>/references/<name>/candidate_2.png") and
        absolute paths.
        """
        discovered = self._discover_characters(story)
        if not discovered:
            return True
        for name in discovered:
            picked = story.characters.get(name)
            if not picked:
                return False
            if not Path(picked).exists():
                # Try repo-relative under data/stories/<id>/ for backward compat
                rel = Path("data/stories") / story.id / picked
                if not rel.exists():
                    return False
        return True
