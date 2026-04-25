"""Stage abstract base class.

Every pipeline stage is a subclass with a unique ``name`` and an async
``run(story, ctx)``. The orchestrator is responsible for recording
StageRun entries and projecting state to SQLite — a Stage's contract is
narrowly "advance the Story by doing my work."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from platinum.models.story import StageStatus, Story
from platinum.pipeline.context import PipelineContext


class Stage(ABC):
    """Abstract pipeline stage. Subclasses set ``name`` and implement ``run``."""

    name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Allow ABCs (intermediate subclasses) to skip this; concrete stages
        # must declare a non-empty name.
        if not getattr(cls, "__abstractmethods__", frozenset()):
            if not cls.name:
                raise TypeError(
                    f"{cls.__name__} is a concrete Stage but has no 'name' class attribute"
                )

    @abstractmethod
    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        """Advance the Story. Return an artifacts dict (paths, metrics) that
        the orchestrator will attach to the StageRun log. Raise on failure."""

    def is_complete(self, story: Story) -> bool:
        """True iff the most recent StageRun for this stage is COMPLETE."""
        latest = story.latest_stage_run(self.name)
        return latest is not None and latest.status == StageStatus.COMPLETE

    def checkpoint(self, story: Story, ctx: PipelineContext) -> None:
        """Persist Story state after this stage. Default: atomic save of
        ``story.json``. Override to perform stage-specific persistence."""
        story.save(ctx.story_path(story))
