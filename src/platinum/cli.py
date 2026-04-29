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
        raise typer.Exit(code=1) from exc

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
    rerun_rejected: bool = typer.Option(
        False, "--rerun-rejected",
        help="Re-run visual_prompts for REJECTED scenes using review_feedback (S7).",
    ),
) -> None:
    """Adapt curator-approved stories: narration -> scenes -> visual prompts.

    Walks `data/stories/*/story.json` and runs the three Session-4 Stages
    (story_adapter, scene_breakdown, visual_prompts) on every story whose
    curator decision was approve and whose visual_prompts stage is not
    yet COMPLETE. Resume-safe: stages already COMPLETE are skipped.
    """
    import logging

    from platinum.models.story import ReviewStatus, StageStatus, Story
    from platinum.pipeline.context import PipelineContext
    from platinum.pipeline.orchestrator import Orchestrator
    from platinum.pipeline.scene_breakdown import SceneBreakdownStage
    from platinum.pipeline.story_adapter import StoryAdapterStage
    from platinum.pipeline.visual_prompts import VisualPromptsStage

    cfg = Config()

    if rerun_rejected:
        eligible_rejected: list[tuple[Story, set[int]]] = []
        for story_dir in sorted(p for p in cfg.stories_dir.iterdir() if p.is_dir()):
            story_json = story_dir / "story.json"
            if not story_json.exists():
                continue
            try:
                s = Story.load(story_json)
            except Exception as exc:
                console.print(f"[yellow]Skipping unreadable {story_dir.name}: {exc}[/yellow]")
                continue
            if story is not None and s.id != story:
                continue
            if track is not None and s.track != track:
                continue
            vp = s.latest_stage_run("visual_prompts")
            if vp is None or vp.status != StageStatus.COMPLETE:
                continue
            rejected_indices = {sc.index for sc in s.scenes
                                if sc.review_status == ReviewStatus.REJECTED}
            if not rejected_indices:
                continue
            eligible_rejected.append((s, rejected_indices))

        if not eligible_rejected:
            console.print("[yellow]No rejected scenes found.[/yellow]")
            raise typer.Exit(code=0)

        ctx = PipelineContext(config=cfg, logger=logging.getLogger("platinum.adapt"))
        for s, rejected_indices in eligible_rejected:
            deviation_feedback = [
                {
                    "index": sc.index,
                    "current_prompt": sc.visual_prompt or "(cleared)",
                    "feedback": sc.review_feedback or "",
                }
                for sc in s.scenes
                if sc.index in rejected_indices
            ]
            cfg.settings.setdefault("runtime", {})["scene_filter"] = rejected_indices
            cfg.settings.setdefault("runtime", {})["deviation_feedback"] = deviation_feedback
            console.print(
                f"[cyan]Re-prompting {s.id} (scenes={sorted(rejected_indices)})...[/cyan]"
            )
            stage = VisualPromptsStage()
            try:
                asyncio.run(stage.run(s, ctx))
            except Exception as exc:
                console.print(f"[red]{s.id} failed: {exc}[/red]")
                raise
            # Save the mutated story back
            s.save(cfg.stories_dir / s.id / "story.json")

        # Clear the runtime knobs after the run so they don't affect later state
        cfg.settings.get("runtime", {}).pop("scene_filter", None)
        cfg.settings.get("runtime", {}).pop("deviation_feedback", None)

        console.print(f"[green]Re-prompted {len(eligible_rejected)} story candidate(s).[/green]")
        return

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
def keyframes(
    story: str = typer.Argument(..., help="Story id (must have visual_prompts COMPLETE)."),
    scenes: str | None = typer.Option(
        None, "--scenes",
        help="Comma-sep scene indices to run (smoke subset). Default: all.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the planned scene set + comfy/aesthetics hosts; do not call.",
    ),
    rerun_regen_requested: bool = typer.Option(
        False, "--rerun-regen-requested",
        help="Auto-build --scenes filter from REGENERATE-status scenes (S7 review loop).",
    ),
    no_content_gate: bool = typer.Option(
        False, "--no-content-gate",
        help="Disable Claude vision content gate for this run (S7.1.A4.6).",
    ),
) -> None:
    """Generate keyframes for a curator-approved + adapted story.

    Requires visual_prompts COMPLETE. Reads PLATINUM_COMFYUI_HOST and
    PLATINUM_AESTHETICS_HOST from .env (or settings.yaml fallback).
    Use --scenes for smoke runs.
    """
    import logging

    from platinum.models.story import ReviewStatus, StageStatus, Story
    from platinum.pipeline.context import PipelineContext
    from platinum.pipeline.keyframe_generator import KeyframeGeneratorStage
    from platinum.pipeline.orchestrator import Orchestrator

    cfg = Config()
    story_path = cfg.stories_dir / story / "story.json"
    if not story_path.exists():
        console.print(
            f"[red]Story not found:[/red] {story} (looked in {story_path})"
        )
        raise typer.Exit(code=1)

    s = Story.load(story_path)
    vp = s.latest_stage_run("visual_prompts")
    if vp is None or vp.status != StageStatus.COMPLETE:
        console.print(
            f"[red]Story {story} has no completed visual_prompts; "
            f"run 'platinum adapt --story {story}' first.[/red]"
        )
        raise typer.Exit(code=1)

    # Parse --scenes "1,8,16" -> {1, 8, 16}. Values are matched against
    # `scene.index` (which the scene_breakdown stage emits as 1-indexed),
    # not against the array offset.
    scene_filter: set[int] | None = None

    if rerun_regen_requested:
        if scenes is not None:
            raise typer.BadParameter(
                "--rerun-regen-requested is mutually exclusive with --scenes",
                param_hint="--rerun-regen-requested",
            )
        regen_indices = sorted({
            sc.index for sc in s.scenes
            if sc.review_status == ReviewStatus.REGENERATE
        })
        if not regen_indices:
            console.print(
                "[yellow]No scenes flagged for regeneration. Run "
                "'platinum review keyframes <id>' first.[/yellow]"
            )
            raise typer.Exit(code=0)
        scene_filter = set(regen_indices)
    elif scenes is not None:
        try:
            scene_filter = {int(x.strip()) for x in scenes.split(",") if x.strip()}
        except ValueError as exc:
            raise typer.BadParameter(
                f"--scenes must be comma-separated integers (got: {scenes!r})",
                param_hint="--scenes",
            ) from exc
        valid_indices = {scene.index for scene in s.scenes}
        unknown = scene_filter - valid_indices
        if unknown:
            available = sorted(valid_indices)
            raise typer.BadParameter(
                f"--scenes references unknown scene index(es) "
                f"{sorted(unknown)}; available: {available}",
                param_hint="--scenes",
            )

    comfy_host = cfg.settings.get("comfyui", {}).get("host", "")
    aest_host = cfg.settings.get("aesthetics", {}).get("host", "")

    if dry_run:
        planned = sorted(scene_filter) if scene_filter else list(range(len(s.scenes)))
        console.print(
            f"[cyan]Would generate keyframes for scenes {planned} of story {story}[/cyan]"
        )
        console.print(f"  comfy   = {comfy_host}")
        console.print(f"  scorer  = {aest_host}")
        raise typer.Exit(code=0)

    cfg.settings.setdefault("runtime", {})["scene_filter"] = scene_filter
    if no_content_gate:
        cfg.settings.setdefault("runtime", {})["no_content_gate"] = True
    ctx = PipelineContext(config=cfg, logger=logging.getLogger("platinum.keyframes"))
    orchestrator = Orchestrator(stages=[KeyframeGeneratorStage()])

    try:
        asyncio.run(orchestrator.run(s, ctx))
    except Exception as exc:
        console.print(f"[red]keyframes failed for {story}: {exc}[/red]")
        raise

    if rerun_regen_requested and scene_filter is not None:
        for sc in s.scenes:
            if sc.index in scene_filter and sc.keyframe_path is not None:
                sc.review_status = ReviewStatus.PENDING
        s.save(story_path)

    console.print(f"[green]Keyframes complete for {story}.[/green]")


@app.command()
def render(
    story: str = typer.Argument(..., help="Story id to render."),
) -> None:
    """Run the render pipeline (keyframes -> video -> upscale -> voice -> mix -> grade)."""
    _stub("render", "Sessions 6-14 (render pipeline)")


# ---------------------------------------------------------------------------
# review sub-app (S7 keyframes; S15 will add `final`)
# ---------------------------------------------------------------------------

review_app = typer.Typer(
    name="review",
    help="Launch a review UI gate (Flask). 'keyframes' after stage 6.",
    no_args_is_help=True,
)
app.add_typer(review_app, name="review")


@review_app.command("keyframes")
def review_keyframes(
    story: str = typer.Argument(..., help="Story id."),
    port: int = typer.Option(5001, "--port", "-p", help="Flask binding port."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Skip webbrowser.open()."
    ),
    threshold: float = typer.Option(
        6.0, "--threshold", "-t",
        help="Default value for the batch-approve threshold input.",
    ),
) -> None:
    """Launch the keyframe review UI for a story.

    Local 127.0.0.1 only. Loads the story from data/stories/<id>/story.json
    and serves a Flask app with per-scene Approve / Reject / Regenerate
    actions.
    """
    import webbrowser

    from platinum.review_ui.app import create_app

    cfg = Config()
    story_path = cfg.stories_dir / story / "story.json"
    if not story_path.exists():
        console.print(
            f"[red]Story not found:[/red] {story} (looked in {story_path})"
        )
        raise typer.Exit(code=1)

    flask_app = create_app(story_id=story, data_root=cfg.stories_dir)
    flask_app.config["DEFAULT_THRESHOLD"] = threshold

    url = f"http://127.0.0.1:{port}/"
    if not no_browser:
        webbrowser.open(url)

    console.print(f"[green]Review UI listening on {url}[/green] (Ctrl+C to stop)")
    flask_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


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
