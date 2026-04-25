"""Story curator — Session 3.

Drives an interactive walk over fetched candidate stories. The user
approves, rejects, or skips each one; approvals append a COMPLETE
StageRun under stage ``story_curator`` (which is what unblocks Session 4's
``adapt`` command), rejections append a SKIPPED StageRun (rejection is an
editorial choice, not an error), and skips leave the story untouched so
the next ``platinum curate`` run will surface it again.

The module is split into:

  * Pure transformations (``apply_decision``) that mutate a Story without
    touching disk or DB — easy to unit-test.
  * I/O helpers (``persist_decision``, ``open_in_editor``, ``load_pending_candidates``)
    that bridge the data model to the filesystem and SQLite index.
  * A ``curate`` driver that composes the above with caller-supplied
    ``decide`` and ``save`` callables. The CLI wires the interactive
    prompt; tests inject a scripted callable.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from platinum.config import Config
from platinum.models.db import create_all, sync_from_story, sync_session
from platinum.models.story import StageRun, StageStatus, Story

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------


class Decision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"


@dataclass(frozen=True)
class CurateSummary:
    approved: int = 0
    rejected: int = 0
    skipped: int = 0


# ---------------------------------------------------------------------------
# Pure transformation
# ---------------------------------------------------------------------------


_DECISION_TO_STATUS: dict[Decision, StageStatus] = {
    Decision.APPROVE: StageStatus.COMPLETE,
    Decision.REJECT: StageStatus.SKIPPED,
}

_DECISION_TO_GATE_LABEL: dict[Decision, str] = {
    Decision.APPROVE: "approved",
    Decision.REJECT: "rejected",
}


def apply_decision(
    story: Story,
    decision: Decision,
    *,
    when: datetime | None = None,
) -> Story:
    """Mutate ``story`` to record a curator decision.

    APPROVE/REJECT append a ``story_curator`` StageRun and write a
    ``review_gates["curator"]`` entry. SKIP is a no-op so the story
    remains curate-eligible the next time the CLI is invoked.
    """
    if decision is Decision.SKIP:
        return story

    now = when or datetime.now()
    label = _DECISION_TO_GATE_LABEL[decision]
    story.stages.append(
        StageRun(
            stage="story_curator",
            status=_DECISION_TO_STATUS[decision],
            started_at=now,
            completed_at=now,
            artifacts={"decision": label},
        )
    )
    story.review_gates["curator"] = {
        "decision": label,
        "decided_at": now.isoformat(),
        "reviewer": "user",
    }
    return story


# ---------------------------------------------------------------------------
# Pending-candidate discovery
# ---------------------------------------------------------------------------


def load_pending_candidates(
    cfg: Config,
    *,
    track: str | None = None,
) -> list[Story]:
    """Load every Story under ``cfg.stories_dir`` (no filter yet)."""
    if not cfg.stories_dir.exists():
        return []
    out: list[Story] = []
    for child in sorted(cfg.stories_dir.iterdir()):
        story_json = child / "story.json"
        if not story_json.exists():
            continue
        try:
            story = Story.load(story_json)
        except Exception as exc:
            # Corrupt story.json or schema mismatch — log and skip rather
            # than abort the whole curation pass.
            logger.warning("Could not load %s: %s", story_json, exc)
            continue
        if track is not None and story.track != track:
            continue
        if story.latest_stage_run("story_curator") is not None:
            continue
        out.append(story)
    return out


# ---------------------------------------------------------------------------
# Editor integration
# ---------------------------------------------------------------------------


def _os_name() -> str:
    """Indirection so tests can patch the platform without forging os.name."""
    return os.name


def _default_editor_argv() -> list[str]:
    return ["notepad"] if _os_name() == "nt" else ["nano"]


def open_in_editor(
    path: Path,
    *,
    runner: Callable[[list[str]], Any] | None = None,
) -> None:
    """Open ``path`` in ``$EDITOR`` (or the platform default).

    ``EDITOR`` is parsed with ``shlex.split`` so values like ``code -w``
    work. The runner returns whatever ``subprocess.run`` returns; the
    return value is ignored — a non-zero editor exit isn't fatal because
    the user may have closed without saving.

    ``runner`` defaults to ``None`` (resolved to ``subprocess.run`` at
    call time) so tests can monkey-patch
    ``platinum.pipeline.story_curator.subprocess.run`` and have the
    patch take effect.
    """
    editor_env = os.environ.get("EDITOR", "").strip()
    argv = (shlex.split(editor_env) if editor_env else _default_editor_argv()) + [
        str(path)
    ]
    actual_runner = runner if runner is not None else subprocess.run
    actual_runner(argv)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


_PREVIEW_CHARS = 500


def _preview(raw: str, limit: int = _PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate to ``limit`` chars."""
    flat = " ".join(raw.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


def render_story_card(story: Story, console: Console) -> None:
    """Print a candidate as a rich Panel with a metadata table + preview."""
    src = story.source
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    word_count = len(src.raw_text.split())
    table.add_row("Story id", story.id)
    table.add_row("Track", story.track)
    table.add_row("Source", src.type)
    table.add_row("Title", src.title)
    table.add_row("Author", src.author or "-")
    table.add_row("Words", str(word_count))
    table.add_row("License", src.license)
    table.add_row("URL", src.url)
    table.add_row("Preview", _preview(src.raw_text))
    console.print(Panel(table, title=src.title, border_style="cyan"))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_decision(cfg: Config, story: Story) -> None:
    """Atomically write the Story JSON and project into SQLite.

    Both writes always run; if the JSON write fails the DB row is not
    touched, and if the DB write fails the JSON is still on disk and the
    in-memory Story matches it. Either case can be reconciled by
    re-running ``platinum curate`` (the JSON is the source of truth).
    """
    story_dir = cfg.story_dir(story.id)
    story.save(story_dir / "story.json")

    db_path = cfg.data_dir / "platinum.db"
    create_all(db_path)
    with sync_session(db_path) as session:
        sync_from_story(session, story)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_DECISION_TO_FIELD: dict[Decision, str] = {
    Decision.APPROVE: "approved",
    Decision.REJECT: "rejected",
    Decision.SKIP: "skipped",
}


_PROMPT_TEXT = (
    "Decision [a=approve, r=reject, s=skip, o=open in editor]: "
)
_CHOICE_TO_DECISION: dict[str, Decision] = {
    "a": Decision.APPROVE,
    "r": Decision.REJECT,
    "s": Decision.SKIP,
}


def make_interactive_decide(
    cfg: Config,
    console: Console,
    *,
    prompt_fn: Callable[[str], str] = input,
    editor_open: Callable[[Path], None] = open_in_editor,
) -> Callable[[Story], Decision]:
    """Build the ``decide`` callable used by the CLI.

    The returned function renders the story card, prompts for one of
    a/r/s/o, loops on invalid input, and on ``o`` opens the source.txt
    in the user's editor before re-prompting. Both the input and the
    editor-open are injectable so unit tests can drive the loop without
    touching stdin or spawning a real editor.
    """

    def decide(story: Story) -> Decision:
        render_story_card(story, console)
        while True:
            choice = prompt_fn(_PROMPT_TEXT).strip().lower()
            mapped = _CHOICE_TO_DECISION.get(choice)
            if mapped is not None:
                return mapped
            if choice == "o":
                editor_open(cfg.story_dir(story.id) / "source.txt")
                continue
            console.print(
                f"[yellow]Unknown choice {choice!r} - use a/r/s/o.[/yellow]"
            )

    return decide


def curate(
    cfg: Config,
    *,
    decide: Callable[[Story], Decision],
    save: Callable[[Story], None],
    track: str | None = None,
) -> CurateSummary:
    """Walk every pending candidate, applying the user's decision to each.

    ``decide`` and ``save`` are injected so unit tests can drive the loop
    without a real terminal. The CLI wires ``decide`` to an interactive
    prompt and ``save`` to ``persist_decision``.

    Returns a ``CurateSummary`` of decision counts. Iteration stops when
    every pending candidate has been visited; skipped stories remain
    eligible for the next ``platinum curate`` run.
    """
    counts: dict[str, int] = {"approved": 0, "rejected": 0, "skipped": 0}
    for story in load_pending_candidates(cfg, track=track):
        decision = decide(story)
        apply_decision(story, decision)
        if decision is not Decision.SKIP:
            save(story)
        counts[_DECISION_TO_FIELD[decision]] += 1
    return CurateSummary(**counts)
