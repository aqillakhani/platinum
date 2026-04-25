"""Unit tests for the story curator (Session 3).

These tests exercise the pure-Python core in
``platinum.pipeline.story_curator`` without going through Typer or stdin.
The CLI integration tests live under ``tests/integration``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from platinum.config import Config
from platinum.models.db import StageRunRow, sync_session
from platinum.models.story import (
    Source,
    StageRun,
    StageStatus,
    Story,
)


# ---------------------------------------------------------------------------
# apply_decision — pure transformation on the Story dataclass
# ---------------------------------------------------------------------------


def test_apply_decision_approve_appends_complete_stagerun(story: Story) -> None:
    from platinum.pipeline.story_curator import Decision, apply_decision

    when = datetime(2026, 4, 25, 10, 0, 0)
    apply_decision(story, Decision.APPROVE, when=when)

    runs = [r for r in story.stages if r.stage == "story_curator"]
    assert len(runs) == 1
    run = runs[0]
    assert run.status == StageStatus.COMPLETE
    assert run.started_at == when
    assert run.completed_at == when
    assert run.artifacts == {"decision": "approved"}
    assert story.review_gates["curator"] == {
        "decision": "approved",
        "decided_at": when.isoformat(),
        "reviewer": "user",
    }


def test_apply_decision_reject_appends_skipped_stagerun(story: Story) -> None:
    """Rejection is editorial, not failure -> SKIPPED, not FAILED."""
    from platinum.pipeline.story_curator import Decision, apply_decision

    when = datetime(2026, 4, 25, 10, 0, 0)
    apply_decision(story, Decision.REJECT, when=when)

    [run] = [r for r in story.stages if r.stage == "story_curator"]
    assert run.status == StageStatus.SKIPPED
    assert run.artifacts == {"decision": "rejected"}
    assert story.review_gates["curator"]["decision"] == "rejected"


def test_apply_decision_skip_is_a_noop(story: Story) -> None:
    """Skip means 'I'll decide later' — leaves the Story curate-eligible."""
    from platinum.pipeline.story_curator import Decision, apply_decision

    stages_before = list(story.stages)
    gates_before = dict(story.review_gates)

    apply_decision(story, Decision.SKIP)

    assert story.stages == stages_before
    assert story.review_gates == gates_before


# ---------------------------------------------------------------------------
# load_pending_candidates — disk discovery
# ---------------------------------------------------------------------------


def _write_story(cfg: Config, story: Story) -> Path:
    """Persist a Story under the project's data/stories/ tree (test helper)."""
    story_dir = cfg.stories_dir / story.id
    story_dir.mkdir(parents=True, exist_ok=True)
    story.save(story_dir / "story.json")
    return story_dir


def test_load_pending_candidates_finds_uncurated(
    tmp_project: Path, story: Story
) -> None:
    """A story that has only a source_fetcher run is curate-eligible."""
    from platinum.pipeline.story_curator import load_pending_candidates

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)

    candidates = load_pending_candidates(cfg)

    assert [c.id for c in candidates] == [story.id]


def test_load_pending_candidates_excludes_approved(
    tmp_project: Path, story: Story
) -> None:
    """COMPLETE story_curator run -> already approved, do not re-surface."""
    from platinum.pipeline.story_curator import (
        Decision,
        apply_decision,
        load_pending_candidates,
    )

    cfg = Config(root=tmp_project)
    apply_decision(story, Decision.APPROVE)
    _write_story(cfg, story)

    assert load_pending_candidates(cfg) == []


def test_load_pending_candidates_excludes_rejected(
    tmp_project: Path, story: Story
) -> None:
    """SKIPPED story_curator run -> rejected, do not re-surface."""
    from platinum.pipeline.story_curator import (
        Decision,
        apply_decision,
        load_pending_candidates,
    )

    cfg = Config(root=tmp_project)
    apply_decision(story, Decision.REJECT)
    _write_story(cfg, story)

    assert load_pending_candidates(cfg) == []


def _make_uncurated(track: str, story_id: str, source: Source) -> Story:
    return Story(
        id=story_id,
        track=track,
        source=source,
        stages=[
            StageRun(
                stage="source_fetcher",
                status=StageStatus.COMPLETE,
                started_at=datetime(2026, 4, 24),
                completed_at=datetime(2026, 4, 24),
            )
        ],
    )


def test_load_pending_candidates_filters_by_track(
    tmp_project: Path, source: Source
) -> None:
    """``track=`` arg surfaces only matching candidates."""
    from platinum.pipeline.story_curator import load_pending_candidates

    cfg = Config(root=tmp_project)
    horror = _make_uncurated("atmospheric_horror", "story_h_001", source)
    sci_fi = _make_uncurated("retro_scifi", "story_s_001", source)
    _write_story(cfg, horror)
    _write_story(cfg, sci_fi)

    result = load_pending_candidates(cfg, track="atmospheric_horror")

    assert [c.id for c in result] == ["story_h_001"]


def test_load_pending_candidates_skips_unreadable_dirs(
    tmp_project: Path, story: Story, caplog: pytest.LogCaptureFixture
) -> None:
    """A dir with no story.json or with corrupt JSON is logged and skipped."""
    from platinum.pipeline.story_curator import load_pending_candidates

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)
    # Sibling directory with no story.json -> silently skipped.
    (cfg.stories_dir / "bogus_dir").mkdir()
    # Sibling directory with corrupt story.json -> logged warning, skipped.
    bad = cfg.stories_dir / "story_corrupt"
    bad.mkdir()
    (bad / "story.json").write_text("{not valid json")

    with caplog.at_level("WARNING", logger="platinum.pipeline.story_curator"):
        result = load_pending_candidates(cfg)

    assert [c.id for c in result] == [story.id]
    assert any("story_corrupt" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# open_in_editor
# ---------------------------------------------------------------------------


def test_open_in_editor_uses_env_editor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EDITOR`` is parsed via shlex so 'code -w' becomes ['code','-w',path]."""
    from platinum.pipeline.story_curator import open_in_editor

    monkeypatch.setenv("EDITOR", "code -w")
    captured: list[list[str]] = []

    def fake_runner(argv: list[str]) -> int:
        captured.append(list(argv))
        return 0

    target = tmp_path / "source.txt"
    target.write_text("text")

    open_in_editor(target, runner=fake_runner)

    assert captured == [["code", "-w", str(target)]]


def test_open_in_editor_falls_back_to_notepad_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from platinum.pipeline.story_curator import open_in_editor

    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr("platinum.pipeline.story_curator._os_name", lambda: "nt")
    captured: list[list[str]] = []

    open_in_editor(tmp_path / "src.txt", runner=lambda argv: captured.append(list(argv)) or 0)

    assert captured[0][0] == "notepad"


def test_open_in_editor_falls_back_to_nano_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from platinum.pipeline.story_curator import open_in_editor

    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setattr("platinum.pipeline.story_curator._os_name", lambda: "posix")
    captured: list[list[str]] = []

    open_in_editor(tmp_path / "src.txt", runner=lambda argv: captured.append(list(argv)) or 0)

    assert captured[0][0] == "nano"


# ---------------------------------------------------------------------------
# persist_decision — JSON + SQLite projection
# ---------------------------------------------------------------------------


def test_persist_decision_writes_json_and_projects_to_db(
    tmp_project: Path, story: Story
) -> None:
    """After approve+persist, story.json reflects the StageRun and the
    SQLite stage_runs table has a story_curator row."""
    from platinum.pipeline.story_curator import (
        Decision,
        apply_decision,
        persist_decision,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)  # baseline json on disk

    apply_decision(story, Decision.APPROVE)
    persist_decision(cfg, story)

    reloaded = Story.load(cfg.story_dir(story.id) / "story.json")
    [run] = [r for r in reloaded.stages if r.stage == "story_curator"]
    assert run.status == StageStatus.COMPLETE

    db_path = cfg.data_dir / "platinum.db"
    with sync_session(db_path) as session:
        rows = session.query(StageRunRow).filter_by(story_id=story.id).all()
    assert "story_curator" in {r.stage for r in rows}


# ---------------------------------------------------------------------------
# curate driver
# ---------------------------------------------------------------------------


def test_curate_runs_decide_for_each_pending_story(
    tmp_project: Path, source: Source
) -> None:
    """Each pending story is shown to ``decide``; non-skip results are saved."""
    from platinum.pipeline.story_curator import (
        CurateSummary,
        Decision,
        curate,
    )

    cfg = Config(root=tmp_project)
    for sid in ("story_001", "story_002", "story_003"):
        _write_story(cfg, _make_uncurated("atmospheric_horror", sid, source))

    scripted = iter([Decision.APPROVE, Decision.REJECT, Decision.SKIP])
    seen: list[str] = []
    saved: list[str] = []

    def decide(story: Story) -> Decision:
        seen.append(story.id)
        return next(scripted)

    def save(story: Story) -> None:
        saved.append(story.id)

    summary = curate(cfg, decide=decide, save=save)

    assert summary == CurateSummary(approved=1, rejected=1, skipped=1)
    assert seen == ["story_001", "story_002", "story_003"]
    # SKIP must not be saved.
    assert saved == ["story_001", "story_002"]


def test_curate_returns_zero_summary_when_empty(tmp_project: Path) -> None:
    """No candidates -> decide is never called and summary is all zeros."""
    from platinum.pipeline.story_curator import (
        CurateSummary,
        Decision,
        curate,
    )

    cfg = Config(root=tmp_project)
    decide_calls: list[Story] = []

    summary = curate(
        cfg,
        decide=lambda s: (decide_calls.append(s) or Decision.APPROVE),
        save=lambda s: None,
    )

    assert summary == CurateSummary()
    assert decide_calls == []


def test_curate_skip_keeps_story_eligible(
    tmp_project: Path, story: Story
) -> None:
    """A skipped story should be re-surfaced on the next ``curate`` run."""
    from platinum.pipeline.story_curator import (
        Decision,
        curate,
        load_pending_candidates,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)

    curate(cfg, decide=lambda s: Decision.SKIP, save=lambda s: None)

    still_pending = load_pending_candidates(cfg)
    assert [c.id for c in still_pending] == [story.id]


def test_curate_passes_track_to_loader(
    tmp_project: Path, source: Source
) -> None:
    """``track`` filter is honoured — only matching candidates are walked."""
    from platinum.pipeline.story_curator import (
        Decision,
        curate,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, _make_uncurated("atmospheric_horror", "story_h", source))
    _write_story(cfg, _make_uncurated("retro_scifi", "story_s", source))

    seen: list[str] = []
    curate(
        cfg,
        track="atmospheric_horror",
        decide=lambda s: (seen.append(s.id) or Decision.SKIP),
        save=lambda s: None,
    )

    assert seen == ["story_h"]


# ---------------------------------------------------------------------------
# render_story_card — display
# ---------------------------------------------------------------------------


def test_render_story_card_includes_key_fields(story: Story) -> None:
    """The card shows id, title, author, source type, word count, preview."""
    from rich.console import Console

    from platinum.pipeline.story_curator import render_story_card

    console = Console(record=True, width=120)
    render_story_card(story, console)
    out = console.export_text()

    word_count = len(story.source.raw_text.split())
    assert story.id in out
    assert story.source.title in out
    assert story.source.author in out
    assert story.source.type in out
    assert str(word_count) in out
    # Preview includes some of the raw text.
    assert "nervous" in out.lower()


# ---------------------------------------------------------------------------
# make_interactive_decide — terminal-driven shell over the pure core
# ---------------------------------------------------------------------------


@pytest.fixture
def quiet_console():
    """A Console that swallows output so tests don't dirty the captured log."""
    import os
    from rich.console import Console

    f = open(os.devnull, "w")
    try:
        yield Console(file=f, force_terminal=False)
    finally:
        f.close()


def test_interactive_decide_approve(
    tmp_project: Path, story: Story, quiet_console
) -> None:
    from platinum.pipeline.story_curator import make_interactive_decide

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)
    inputs = iter(["a"])
    decide = make_interactive_decide(
        cfg, quiet_console, prompt_fn=lambda _: next(inputs)
    )
    from platinum.pipeline.story_curator import Decision

    assert decide(story) is Decision.APPROVE


def test_interactive_decide_reject(
    tmp_project: Path, story: Story, quiet_console
) -> None:
    from platinum.pipeline.story_curator import (
        Decision,
        make_interactive_decide,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)
    decide = make_interactive_decide(
        cfg, quiet_console, prompt_fn=lambda _: "R"  # also tests case folding
    )
    assert decide(story) is Decision.REJECT


def test_interactive_decide_skip(
    tmp_project: Path, story: Story, quiet_console
) -> None:
    from platinum.pipeline.story_curator import (
        Decision,
        make_interactive_decide,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)
    decide = make_interactive_decide(
        cfg, quiet_console, prompt_fn=lambda _: "s"
    )
    assert decide(story) is Decision.SKIP


def test_interactive_decide_invalid_then_valid(
    tmp_project: Path, story: Story, quiet_console
) -> None:
    """Garbage input loops until a valid choice arrives."""
    from platinum.pipeline.story_curator import (
        Decision,
        make_interactive_decide,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)
    inputs = iter(["", "x", "approve please", "a"])
    decide = make_interactive_decide(
        cfg, quiet_console, prompt_fn=lambda _: next(inputs)
    )
    assert decide(story) is Decision.APPROVE


def test_interactive_decide_open_editor_then_approve(
    tmp_project: Path, story: Story, quiet_console
) -> None:
    """``o`` opens the source file in the editor and re-prompts."""
    from platinum.pipeline.story_curator import (
        Decision,
        make_interactive_decide,
    )

    cfg = Config(root=tmp_project)
    _write_story(cfg, story)
    # Make sure source.txt exists alongside story.json so the editor has
    # something to open.
    (cfg.story_dir(story.id) / "source.txt").write_text(story.source.raw_text)

    opened: list[Path] = []
    inputs = iter(["o", "a"])
    decide = make_interactive_decide(
        cfg,
        quiet_console,
        prompt_fn=lambda _: next(inputs),
        editor_open=lambda p: opened.append(p),
    )

    assert decide(story) is Decision.APPROVE
    assert opened == [cfg.story_dir(story.id) / "source.txt"]
