"""Orchestrator resume-and-checkpoint tests."""

from __future__ import annotations

from typing import Any

import pytest

from platinum.models.db import StageRunRow, sync_session
from platinum.models.story import StageStatus, Story
from platinum.pipeline.context import PipelineContext
from platinum.pipeline.orchestrator import CANONICAL_STAGE_NAMES, Orchestrator
from platinum.pipeline.stage import Stage


class _StageA(Stage):
    name = "stage_a"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        return {"a": 1}


class _StageBFail(Stage):
    name = "stage_b"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        raise RuntimeError("boom")


class _StageBOk(Stage):
    name = "stage_b"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        return {"b": 2}


class _StageC(Stage):
    name = "stage_c"

    async def run(self, story: Story, ctx: PipelineContext) -> dict[str, Any]:
        return {"c": 3}


def test_canonical_has_eighteen_stages() -> None:
    assert len(CANONICAL_STAGE_NAMES) == 18
    # PRD stages 1 and 18 as sanity anchors.
    assert CANONICAL_STAGE_NAMES[0] == "source_fetcher"
    assert CANONICAL_STAGE_NAMES[-1] == "publisher"


def test_stage_without_name_rejected() -> None:
    with pytest.raises(TypeError, match="no 'name' class attribute"):

        class _NoName(Stage):  # type: ignore[unused-variable]
            async def run(self, story, ctx):
                return {}


async def test_orchestrator_runs_all_stages_in_order(
    story: Story, context: PipelineContext
) -> None:
    # Start with an empty stages list to remove the fixture's source_fetcher run
    # which would make StageA look like a different stage.
    story.stages = []
    orch = Orchestrator(stages=[_StageA(), _StageBOk(), _StageC()])
    await orch.run(story, context)
    assert [r.stage for r in story.stages] == ["stage_a", "stage_b", "stage_c"]
    assert all(r.status == StageStatus.COMPLETE for r in story.stages)
    # Artifacts captured from each stage's return value.
    assert story.latest_stage_run("stage_a").artifacts == {"a": 1}


async def test_orchestrator_halts_and_checkpoints_on_failure(
    story: Story, context: PipelineContext
) -> None:
    story.stages = []
    orch = Orchestrator(stages=[_StageA(), _StageBFail(), _StageC()])
    with pytest.raises(RuntimeError, match="boom"):
        await orch.run(story, context)

    # A recorded COMPLETE, B recorded FAILED with the error captured, C never ran.
    assert story.latest_stage_run("stage_a").status == StageStatus.COMPLETE
    b = story.latest_stage_run("stage_b")
    assert b is not None and b.status == StageStatus.FAILED
    assert "boom" in (b.error or "")
    assert story.latest_stage_run("stage_c") is None

    # Checkpoint wrote story.json through the failure.
    reloaded = Story.load(context.story_path(story))
    assert reloaded.latest_stage_run("stage_b").status == StageStatus.FAILED


async def test_orchestrator_resumes_from_mid_pipeline_state(
    story: Story, context: PipelineContext
) -> None:
    story.stages = []
    # First attempt: stage B fails.
    first = Orchestrator(stages=[_StageA(), _StageBFail(), _StageC()])
    with pytest.raises(RuntimeError):
        await first.run(story, context)

    # Reload from disk (simulates a fresh process resuming after a crash).
    resumed = Story.load(context.story_path(story))

    # Second attempt: B now succeeds. A must be skipped; C runs fresh.
    second = Orchestrator(stages=[_StageA(), _StageBOk(), _StageC()])
    await second.run(resumed, context)

    a_runs = [r for r in resumed.stages if r.stage == "stage_a"]
    b_runs = [r for r in resumed.stages if r.stage == "stage_b"]
    c_runs = [r for r in resumed.stages if r.stage == "stage_c"]
    assert len(a_runs) == 1, "stage_a should have been skipped on resume"
    assert len(b_runs) == 2, "stage_b should have one FAILED and one COMPLETE run"
    assert b_runs[0].status == StageStatus.FAILED
    assert b_runs[1].status == StageStatus.COMPLETE
    assert len(c_runs) == 1 and c_runs[0].status == StageStatus.COMPLETE


async def test_orchestrator_projects_to_sqlite_after_each_stage(
    story: Story, context: PipelineContext
) -> None:
    story.stages = []
    orch = Orchestrator(stages=[_StageA(), _StageBOk()])
    await orch.run(story, context)
    with sync_session(context.db_path) as s:
        rows = s.query(StageRunRow).filter_by(story_id=story.id).all()
    assert {r.stage for r in rows} == {"stage_a", "stage_b"}
    assert all(r.status == StageStatus.COMPLETE.value for r in rows)
