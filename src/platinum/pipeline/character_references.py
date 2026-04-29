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

from platinum.models.story import Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage


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
