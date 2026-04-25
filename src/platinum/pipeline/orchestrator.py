"""Checkpoint-aware pipeline orchestrator.

Composes an ordered list of Stages and runs them against a Story. Stages
that are already complete are skipped (resume-on-restart). Each attempt
appends a StageRun to ``story.stages`` (append-log semantics); on failure
the orchestrator halts *that* story, checkpoints current state, and
re-raises so the caller can decide how to proceed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from platinum.models.db import create_all, sync_from_story, sync_session
from platinum.models.story import StageRun, StageStatus, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.stage import Stage

# Canonical stage order defined by PRD + plan §6 (17 stages plus added 8.5).
# Used by ``platinum status`` to display the pipeline definition independent
# of a concrete run.
CANONICAL_STAGE_NAMES: list[str] = [
    "source_fetcher",
    "story_curator",
    "story_adapter",
    "scene_breakdown",
    "visual_prompts",
    "keyframe_generator",
    "keyframe_review",
    "video_generator",
    "upscaler",
    "voice_generator",
    "music_selector",
    "sfx_layer",
    "subtitle_generator",
    "assembly",
    "color_grade",
    "final_review",
    "thumbnail",
    "publisher",
]


@dataclass
class Orchestrator:
    """Runs a list of stages against a Story, with checkpoint/resume."""

    stages: list[Stage] = field(default_factory=list)

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    async def run(self, story: Story, ctx: PipelineContext) -> Story:
        """Execute stages in order. Skip any whose ``is_complete`` already
        returns True; on first failure, checkpoint + re-raise."""
        # Ensure the SQLite schema exists before we try to project anything.
        create_all(ctx.db_path)

        for stage in self.stages:
            if stage.is_complete(story):
                ctx.logger.info(
                    "[%s] %s: skipped (already complete)", story.id, stage.name
                )
                continue

            run = StageRun(
                stage=stage.name,
                status=StageStatus.RUNNING,
                started_at=datetime.now(),
            )
            story.stages.append(run)
            ctx.logger.info("[%s] %s: running", story.id, stage.name)

            try:
                artifacts = await stage.run(story, ctx)
            except Exception as exc:
                run.status = StageStatus.FAILED
                run.completed_at = datetime.now()
                run.error = f"{type(exc).__name__}: {exc}"
                ctx.logger.error(
                    "[%s] %s: FAILED — %s", story.id, stage.name, run.error
                )
                self._checkpoint(stage, story, ctx)
                raise

            run.status = StageStatus.COMPLETE
            run.completed_at = datetime.now()
            run.artifacts = dict(artifacts) if artifacts else {}
            ctx.logger.info("[%s] %s: complete", story.id, stage.name)
            self._checkpoint(stage, story, ctx)

        return story

    def _checkpoint(self, stage: Stage, story: Story, ctx: PipelineContext) -> None:
        """Persist Story JSON + project to SQLite."""
        stage.checkpoint(story, ctx)
        with sync_session(ctx.db_path) as session:
            sync_from_story(session, story)
