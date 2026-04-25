"""Typer CLI entry point.

Most commands are stubs until later sessions implement them. ``status``
is real from Session 1; ``fetch`` is real from Session 2.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from platinum.config import Config
from platinum.models.db import StageRunRow, StoryRow, create_all, sync_session
from platinum.models.story import StageStatus
from platinum.pipeline.orchestrator import CANONICAL_STAGE_NAMES
from platinum.pipeline.story_curator import (
    curate as _curate_run,
)
from platinum.pipeline.story_curator import (
    make_interactive_decide,
    persist_decision,
)
from platinum.sources.runner import fetch_track_sources, persist_source_as_story

app = typer.Typer(
    name="platinum",
    help="Cinematic AI short-film pipeline.",
    no_args_is_help=True,
)
console = Console()


_NOT_YET_IMPLEMENTED = (
    "Command '{name}' is not implemented yet — scheduled for {session}. "
    "See the implementation plan for details."
)


def _stub(name: str, session: str) -> None:
    console.print(f"[yellow]{_NOT_YET_IMPLEMENTED.format(name=name, session=session)}[/yellow]")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Real commands
# ---------------------------------------------------------------------------


@app.command()
def status(
    story: str | None = typer.Option(
        None, "--story", "-s", help="Show actual state for this story id (resolved from SQLite)."
    ),
) -> None:
    """Show pipeline status.

    With no args, prints the canonical 18-stage pipeline definition with
    every stage marked PENDING. With ``--story <id>``, resolves the most
    recent StageRun per stage from SQLite.
    """
    if story is None:
        _print_canonical()
        return

    cfg = Config()
    create_all(cfg.data_dir / "platinum.db")
    with sync_session(cfg.data_dir / "platinum.db") as session:
        story_row = session.get(StoryRow, story)
        if story_row is None:
            console.print(f"[red]Story not found:[/red] {story}")
            raise typer.Exit(code=1)

        # Latest StageRun per stage for this story
        runs = (
            session.query(StageRunRow)
            .filter(StageRunRow.story_id == story)
            .order_by(StageRunRow.id)
            .all()
        )
    latest: dict[str, StageRunRow] = {}
    for r in runs:
        latest[r.stage] = r  # later rows overwrite earlier -> latest wins

    _print_story_status(story_row, latest)


# ---------------------------------------------------------------------------
# Stubs — one command per future session
# ---------------------------------------------------------------------------


@app.command()
def fetch(
    track: str = typer.Option(..., "--track", "-t", help="Track id (e.g. atmospheric_horror)."),
    limit: int = typer.Option(10, "--limit", "-n", help="How many candidates to fetch."),
) -> None:
    """Fetch candidate source stories for a track.

    Drives every source listed under the track YAML's ``sources`` block in
    order until ``--limit`` candidates are gathered, then writes each as a
    Story JSON under ``data/stories/<id>/``.
    """
    cfg = Config()
    try:
        track_cfg = cfg.track(track)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if not track_cfg.get("sources"):
        console.print(
            f"[red]Track '{track}' has no 'sources' block configured.[/red]"
        )
        raise typer.Exit(code=1)

    sources = asyncio.run(fetch_track_sources(track_cfg, limit=limit))
    if not sources:
        console.print(
            f"[yellow]No candidates returned for track '{track}'. "
            f"Check filters and network access.[/yellow]"
        )
        raise typer.Exit(code=0)

    table = Table(title=f"Fetched candidates — track={track}", show_lines=False)
    table.add_column("Story id", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Title", style="white")
    table.add_column("Author", style="white")
    table.add_column("Words", style="green", justify="right")

    for src in sources:
        story = persist_source_as_story(cfg, src, track=track)
        table.add_row(
            story.id,
            src.type,
            (src.title or "")[:60],
            (src.author or "—")[:30],
            str(len(src.raw_text.split())),
        )

    console.print(table)
    console.print(
        f"[green]Wrote {len(sources)} story candidate(s) to "
        f"{cfg.stories_dir}[/green]"
    )


@app.command()
def curate(
    track: str | None = typer.Option(
        None, "--track", "-t", help="Restrict to one track id."
    ),
) -> None:
    """Interactive curator -- walk fetched candidates, approve/reject/skip.

    Approve appends a COMPLETE ``story_curator`` StageRun (which
    unblocks Stage 3, ``adapt``); reject appends a SKIPPED StageRun
    (rejection is editorial, not failure); skip leaves the story
    untouched so the next ``platinum curate`` run will re-surface it.
    Pressing ``o`` opens the source text in ``$EDITOR`` (notepad on
    Windows / nano elsewhere by default) and re-prompts.
    """
    cfg = Config()
    decide = make_interactive_decide(cfg, console)

    def _save(story) -> None:
        persist_decision(cfg, story)

    summary = _curate_run(cfg, decide=decide, save=_save, track=track)

    if (summary.approved + summary.rejected + summary.skipped) == 0:
        console.print(
            "[yellow]No pending candidates. Run "
            "'platinum fetch --track <id>' first.[/yellow]"
        )
        return

    console.print(
        f"[green]Curation complete.[/green] "
        f"approved={summary.approved} "
        f"rejected={summary.rejected} "
        f"skipped={summary.skipped}"
    )


@app.command()
def adapt(
    story: str | None = typer.Option(
        None, "--story", "-s", help="Adapt only this story id."
    ),
    track: str | None = typer.Option(
        None, "--track", "-t", help="Restrict to one track id."
    ),
) -> None:
    """Adapt curator-approved stories: narration -> scenes -> visual prompts.

    Walks `data/stories/*/story.json` and runs the three Session-4 Stages
    (story_adapter, scene_breakdown, visual_prompts) on every story whose
    curator decision was approve and whose visual_prompts stage is not
    yet COMPLETE. Resume-safe: stages already COMPLETE are skipped.
    """
    import logging

    from platinum.models.story import StageStatus, Story
    from platinum.pipeline.context import PipelineContext
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage
    from platinum.pipeline.story_adapter import StoryAdapterStage
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    cfg = Config()
    eligible: list[Story] = []
    for story_dir in sorted(p for p in cfg.stories_dir.iterdir() if p.is_dir()):
        story_json = story_dir / "story.json"
        if not story_json.exists():
            continue
        try:
            s = Story.load(story_json)
        except Exception as exc:
            console.print(f"[yellow]Skipping unreadable story at {story_dir.name}: {exc}[/yellow]")
            continue

        if story is not None and s.id != story:
            continue
        if track is not None and s.track != track:
            continue

        curator = s.latest_stage_run("story_curator")
        if curator is None or curator.status != StageStatus.COMPLETE:
            continue

        vp = s.latest_stage_run("visual_prompts")
        if vp is not None and vp.status == StageStatus.COMPLETE:
            continue

        eligible.append(s)

    if not eligible:
        console.print("[yellow]No eligible stories to adapt.[/yellow]")
        return

    ctx = PipelineContext(config=cfg, logger=logging.getLogger("platinum.adapt"))
    orchestrator = Orchestrator(stages=[
        StoryAdapterStage(), SceneBreakdownStage(), VisualPromptsStage(),
    ])

    for s in eligible:
        console.print(f"[cyan]Adapting {s.id} (track={s.track})...[/cyan]")
        try:
            asyncio.run(orchestrator.run(s, ctx))
        except Exception as exc:
            console.print(f"[red]{s.id} failed: {exc}[/red]")
            raise

    console.print(f"[green]Adapted {len(eligible)} story candidate(s).[/green]")


@app.command()
def render(
    story: str = typer.Argument(..., help="Story id to render."),
) -> None:
    """Run the render pipeline (keyframes -> video -> upscale -> voice -> mix -> grade)."""
    _stub("render", "Sessions 6-14 (render pipeline)")


@app.command()
def review(
    target: str = typer.Argument(..., help="What to review: 'keyframes' or 'final'."),
    story: str = typer.Argument(..., help="Story id."),
) -> None:
    """Launch a review UI gate (Flask). 'keyframes' after stage 6, 'final' after stage 14."""
    if target not in {"keyframes", "final"}:
        console.print(f"[red]Unknown review target:[/red] {target} (use 'keyframes' or 'final')")
        raise typer.Exit(code=1)
    _stub("review", "Sessions 7 / 15 (review UIs)")


@app.command()
def publish(
    story: str = typer.Argument(..., help="Story id to publish."),
) -> None:
    """Publish a finished story to YouTube."""
    _stub("publish", "Session 16 (publisher)")


@app.command("report-costs")
def report_costs(
    story: str | None = typer.Option(None, "--story", "-s", help="Filter to one story."),
) -> None:
    """Sum Anthropic + cloud GPU costs across stories or for one story."""
    _stub("report-costs", "Session 4 (cost tracking) / Session 17 (report)")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _print_canonical() -> None:
    table = Table(
        title="Platinum pipeline — canonical stage definition",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="yellow")
    for i, name in enumerate(CANONICAL_STAGE_NAMES, start=1):
        table.add_row(str(i), name, StageStatus.PENDING.value.upper())
    console.print(table)


def _print_story_status(
    story_row: StoryRow, latest: dict[str, StageRunRow]
) -> None:
    table = Table(
        title=f"Story {story_row.id} — {story_row.track} ({story_row.status})",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Completed at", style="green")
    table.add_column("Error", style="red")
    for i, name in enumerate(CANONICAL_STAGE_NAMES, start=1):
        run = latest.get(name)
        if run is None:
            status_label = StageStatus.PENDING.value.upper()
            completed = ""
            error = ""
        else:
            status_label = run.status.upper()
            completed = run.completed_at.isoformat(timespec="seconds") if run.completed_at else ""
            error = (run.error or "")[:60]
        table.add_row(str(i), name, status_label, completed, error)
    console.print(table)


if __name__ == "__main__":
    app()
