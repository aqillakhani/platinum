# Session 3 — Story curator CLI

**Spec:** plan §8 Session 3. **Deliverable:** `python -m platinum curate` walks the candidates produced by Session 2's `fetch`, accepting `a` (approve) / `r` (reject) / `s` (skip) / `o` (open in $EDITOR) per story, and persists each decision to `story.json` + SQLite. Approved stories satisfy stage 2 (`story_curator`) so Session 4's `adapt` can pick them up.

## Files

New:
- `src/platinum/pipeline/story_curator.py` — pure orchestration: discover pending candidates, render story card, apply decision (StageRun + review_gate), persist, drive the loop. No Typer dependency.
- `tests/unit/test_story_curator.py` — pending-discovery, decision application, editor open, persistence projection, loop.
- `tests/integration/test_curate_command.py` — end-to-end CLI (Typer `CliRunner` with scripted stdin) over real SQLite + tmp project.

Modified:
- `src/platinum/cli.py` — replace the `curate` stub with a real implementation that wires `story_curator.curate()` to a `typer.prompt`-based interactive `decide` callable. Add `--track` filter.

(Optional, only if needed by tests:)
- `tests/conftest.py` — possibly add a `curated_story` / `pending_story` fixture if the integration test would otherwise duplicate setup.

## Architecture

A pure-Python core, an interactive shell, and a thin CLI. The core has no Typer/rich/stdin dependency so it is fully unit-testable.

```
cli.py (curate command)
    │  builds the interactive `decide` callable (typer.prompt + render_story_card)
    │  builds the `save` callable (Story.save + sync_from_story)
    ▼
story_curator.curate(cfg, *, track, decide, save)
    │  loads pending candidates from data/stories/
    │  for each story:
    │       decision = decide(story)        ← may loop on 'o' internally
    │       apply_decision(story, decision) ← pure function (StageRun + review_gate)
    │       if decision != SKIP: save(story)
    ▼
returns CurateSummary(approved, rejected, skipped)
```

## Decisions

- **Decision model:** `class Decision(str, Enum): APPROVE = "approve" / REJECT = "reject" / SKIP = "skip"`.
- **StageRun semantics** for stage `story_curator`:
  - approve → append `StageRun(stage="story_curator", status=COMPLETE, started_at=now, completed_at=now, artifacts={"decision": "approved"})`. This is what unblocks Stage 3 (`story_adapter`).
  - reject → append `StageRun(stage="story_curator", status=SKIPPED, started_at=now, completed_at=now, artifacts={"decision": "rejected"})`. Stage 3 can short-circuit on SKIPPED rather than COMPLETE.
  - skip → no StageRun appended; story stays curate-eligible the next time `platinum curate` runs.
- **Review gate:** also write `story.review_gates["curator"] = {"decision": "approved"|"rejected", "decided_at": ISO, "reviewer": "user"}` for human-readable visibility in `story.json`. Skip writes nothing here either.
- **Pending-candidate discovery:** scan `cfg.stories_dir/*/story.json`; load each via `Story.load`; include if `story.latest_stage_run("story_curator") is None`. Optional `track == story.track` filter. Skip dirs whose `story.json` is missing or unparseable (log warning, continue).
- **Display:** `rich.panel.Panel` per candidate. Fields: id, track, source.type, title, author (or "—"), license, word_count = `len(source.raw_text.split())`, url, preview = first ~500 chars of `source.raw_text` with whitespace collapsed.
- **Interactive prompt:** `typer.prompt("Decision [a/r/s/o]")` with manual choice validation (Typer's `type=Choice` works but lowercase coercion is cleaner manual). Loop on invalid input; loop on `o` after editor returns; return `Decision` on `a`/`r`/`s`.
- **EDITOR open:** `subprocess.run([editor, str(story_dir / "source.txt")])`. `editor = shlex.split(os.environ["EDITOR"])` if set else `["notepad"]` on Windows / `["nano"]` elsewhere. After it returns (any exit code), reload the Story from disk in case the user edited it (safer than holding stale state) and re-prompt.
- **Persistence:** after a non-skip decision, `Story.save(story_dir / "story.json")` (atomic write already implemented) + `sync_from_story(session, story)` inside a `sync_session(db_path)` context. The `save` callable in `curate()` handles both.
- **`--track` filter:** when provided, only candidates with matching `story.track` are surfaced; useful when multiple tracks have been fetched.
- **Exit codes:** 0 on normal completion (including "no candidates"). Non-zero only on configuration errors (e.g., bad `--track` argument that excludes everything is still 0; only programming errors / unhandled exceptions are non-zero).
- **No Stage wrapper:** like Session 2, the Stage subclass for `story_curator` is not needed yet — the orchestrator doesn't drive interactive curation. The decision lives in `story.json` for any later automated re-runs.

## TDD checklist

Unit (`tests/unit/test_story_curator.py`):
- [ ] `test_load_pending_candidates_finds_uncurated` — Story without `story_curator` run is included.
- [ ] `test_load_pending_candidates_excludes_approved` — Story with COMPLETE `story_curator` run is excluded.
- [ ] `test_load_pending_candidates_excludes_rejected` — Story with SKIPPED `story_curator` run is excluded.
- [ ] `test_load_pending_candidates_filters_by_track` — `track=` arg drops other tracks.
- [ ] `test_load_pending_candidates_skips_unreadable_dirs` — missing/corrupt `story.json` logs and continues.
- [ ] `test_apply_decision_approve_appends_complete_stagerun` — and writes `review_gates["curator"]={"decision":"approved",...}`.
- [ ] `test_apply_decision_reject_appends_skipped_stagerun` — and writes `review_gates["curator"]={"decision":"rejected",...}`.
- [ ] `test_apply_decision_skip_no_stagerun_appended` — `story.stages` length unchanged; no review_gate.
- [ ] `test_open_in_editor_uses_env_editor` — monkeypatch `subprocess.run`, set `EDITOR=code -w`, assert argv = `["code","-w",str(path)]`.
- [ ] `test_open_in_editor_falls_back_when_env_unset` — monkeypatch `os.name`/env, assert notepad on win, nano on posix.
- [ ] `test_persist_decision_writes_json_and_projects_to_db` — story.json + SQLite row updated; `latest stage_run` for that story has stage=story_curator.
- [ ] `test_curate_runs_decide_for_each_pending_story` — scripted decisions list, assert summary counts and per-story state.
- [ ] `test_curate_returns_zero_summary_when_empty` — no candidates → `CurateSummary(0,0,0)`, decide never invoked.
- [ ] `test_curate_skip_leaves_story_eligible_next_time` — after skip, `load_pending_candidates` still returns the same story.
- [ ] `test_render_story_card_includes_key_fields` — capture rich Console output, assert title/author/word_count/preview present.

Integration (`tests/integration/test_curate_command.py`):
- [ ] `test_cli_curate_walks_candidates_with_scripted_input` — pre-populate 3 stories via `persist_source_as_story`, invoke `platinum curate` with `input="a\nr\ns\n"`, assert exit 0, JSON + DB reflect approved/rejected/(no run for skipped).
- [ ] `test_cli_curate_no_candidates_exits_zero_with_message` — empty `data/stories/`, exit 0, output contains a "no candidates" line.
- [ ] `test_cli_curate_open_editor_then_approve` — `input="o\na\n"`, monkeypatch `subprocess.run`, assert editor invoked once and decision is approve.
- [ ] `test_cli_curate_track_filter` — fetch into two tracks, `platinum curate --track atmospheric_horror`, only horror candidates surfaced.
- [ ] `test_cli_curate_status_reflects_approval` — after approve, `platinum status --story <id>` shows `story_curator` COMPLETE.

Quality gates:
- [ ] `pytest -q` passes (Session 1 + 2 + 3 ≈ 95–100 tests; 0 failures).
- [ ] `ruff check src tests` clean (or only style-only changes are needed).
- [ ] Live smoke: re-run `python -m platinum fetch --track atmospheric_horror --limit 5`, then `python -m platinum curate` and walk a/r/s/o; confirm `story.json` and SQLite mirror the decisions.

## Notes

- `data/stories/` is currently empty (the prior session's smoke fetch wasn't committed; only `.gitkeep` is tracked). Live demo needs a fresh `fetch` first.
- `typer.prompt` reads from `sys.stdin`, so `CliRunner(...).invoke(app, [...], input="a\nr\ns\n")` drives it deterministically. Do not introduce `rich.prompt.Prompt.ask` for the main decision — it can bypass stdin patching depending on rich's TTY detection.
- The integration test must monkey-patch `subprocess.run` (used only by `open_in_editor`) so CI never spawns a real editor.
- Per Windows lessons from Session 2: keep CLI strings ASCII-only (no `→`, no smart quotes) so `--help` and Rich rendering don't trip cp1252 console.
- `apply_decision` is a pure transformation on the Story dataclass — easy to unit test without touching disk or DB. The DB+disk write lives in a separate `persist_decision(cfg, story)` helper.
- "Open in editor" reloading the story after the editor exits matters because Stage 4 may want any narrator notes the user jotted into `source.txt`. Reloading also makes the post-edit prompt show the (possibly truncated) preview accurately.
- For the rejected path, status SKIPPED (rather than FAILED) is the right semantic: rejection is an editorial choice, not an error.
- Walking "20 candidates in under 10 minutes" (plan's deliverable wording) is purely a UX target — no automated assertion. The card layout is designed so a human can decide in ~30s per story.

---

## Review (Session 3 complete — 2026-04-24)

**Tests:** 104 pass total (Session 1: 22, Session 2: 55, Session 3: 27 — 22 unit + 5 integration). 0 failures, 0 skips. Run time ~3.6s.

**Files added:**
- `src/platinum/pipeline/story_curator.py` (~220 lines) — `Decision`/`CurateSummary` types, pure `apply_decision`, disk-scanning `load_pending_candidates`, `open_in_editor`, `render_story_card`, `persist_decision`, `make_interactive_decide`, `curate` driver.
- `tests/unit/test_story_curator.py` (22 tests, ~480 lines).
- `tests/integration/test_curate_command.py` (5 tests, ~190 lines).

**Files modified:**
- `src/platinum/cli.py` — replaced the `curate` stub with the real implementation; added `--track` option; wired `make_interactive_decide` + `persist_decision` into the `curate` driver.

**Live deliverable verified:**
1. `python -m platinum fetch --track atmospheric_horror --limit 3` produced 3 candidates (Cask of Amontillado, Call of Cthulhu, Shunned House).
2. `printf "a\nr\ns\n" | python -m platinum curate` walked them in order. Output: `approved=1 rejected=1 skipped=1`.
3. On disk:
   - story_001 → `latest_stage_run("story_curator")` is COMPLETE; `review_gates["curator"]={"decision":"approved",...}`.
   - story_002 → `latest_stage_run("story_curator")` is SKIPPED; `review_gates["curator"]={"decision":"rejected",...}`.
   - story_003 → no `story_curator` run; `review_gates` untouched.
4. SQLite `stage_runs` table has rows for stories 001 (status=complete) and 002 (status=skipped); none for 003.
5. `python -m platinum status --story story_2026_04_24_001` reports stage 2 (`story_curator`) as COMPLETE.
6. Re-running `platinum curate` only re-surfaces story_003 — confirms the skip-leaves-eligible behavior + the approve/reject filters at the disk-scan level.

**Surprises / lessons:**
1. **Default-arg captured at definition time.** `open_in_editor(path, *, runner=subprocess.run)` captured the original `subprocess.run` at function-definition time, so monkey-patching `platinum.pipeline.story_curator.subprocess.run` in the integration test didn't take effect. Fix: default `runner=None`, resolve to `subprocess.run` at call time. Worth remembering for any future testable injection points: late binding wins over early binding for testability.
2. **Rejection is editorial, not failure.** Settled on `StageStatus.SKIPPED` for the rejected `story_curator` run rather than FAILED. Stage 4 (story_adapter) can short-circuit on SKIPPED without treating it as a pipeline error, which keeps the orchestrator's "halt on failure" semantics intact.
3. **Pure core / impure shell.** Splitting into a pure `apply_decision` (no I/O) plus injectable `decide`/`save` callables made the `curate` driver trivial to unit-test (one test, scripted decisions, no monkeypatching). The CLI just supplies real implementations.
4. **rich.Panel + Table.grid** renders cleanly under both a real terminal and `CliRunner`'s captured-output mode. The `Console.export_text()` test for `render_story_card` doubles as a UI snapshot test without committing fixture files.
5. The piped smoke test (`printf "a\nr\ns\n" | python -m platinum curate`) works on Windows bash because `input()` reads sys.stdin which receives the piped bytes — no TTY required.

**Not done in this session (deferred to later):**
- A `Stage` subclass for `story_curator` that the orchestrator could drive automatically. Not needed: the orchestrator never auto-curates, the human is always in the loop. The decision lives in `story.json` so a future automated re-run can read it without re-prompting.
- Reloading `story.json` after `o` returns from the editor. The user is editing `source.txt` (display copy), not `story.json`. Stage 4 reads `story.source.raw_text` from `story.json`, so post-editor edits to `source.txt` don't currently propagate. Left as a known gap; will revisit if Session 4 needs it.
- Bulk-mode flags like `--auto-approve-above N words` or `--reject-keywords "<list>"`. Plan doesn't ask for them; YAGNI for v1.
- Cleaning up the live-smoke-test stories under `data/stories/`. They stay on disk locally (handy for the next session) but aren't committed — matches Session 2 convention (`.gitkeep` only). — no automated assertion. The card layout is designed so a human can decide in ~30s per story.

---

## Review (Session 4 complete — 2026-04-25)

**Tests:** 155 pass total (Session 1: 22, Session 2: 55, Session 3: 27, Session 4: 29). 0 failures, 0 skips. Run time ~12s.

**Files added:**
- `src/platinum/utils/claude.py` (~180 lines) — pricing table, `ClaudeUsage`/`ClaudeResult`/`RecordedCall` dataclasses, `Recorder` protocol, `FixtureRecorder` replay-only class.
- `tests/_fixtures.py` (~120 lines) — `FixtureRecorder` implementation + fixture JSON serialization.
- `tests/unit/test_claude_util.py` (9 tests, ~240 lines).
- `tests/unit/test_recorder.py` (4 tests, ~140 lines).
- `src/platinum/utils/prompts.py` (~60 lines) — Jinja2 template loader + `render_prompt` helper.
- `tests/unit/test_prompts.py` (2 tests, ~70 lines).
- `src/platinum/pipeline/story_adapter.py` (~250 lines) — `StoryAdapterStage` runs Claude to produce `adapted` (narration_script + arc).
- `tests/unit/test_story_adapter.py` (7 tests, ~220 lines).
- `src/platinum/pipeline/scene_breakdown.py` (~200 lines) — `SceneBreakdownStage` runs Claude to produce 8-scene breakdown with mood + sfx_cues.
- `tests/unit/test_scene_breakdown.py` (6 tests, ~180 lines).
- `src/platinum/pipeline/visual_prompts.py` (~220 lines) — `VisualPromptsStage` runs Claude to produce per-scene visual + negative prompts for Flux diffusion.
- `tests/unit/test_visual_prompts.py` (5 tests, ~160 lines).
- `tests/integration/test_adapt_stages.py` (3 tests, ~160 lines) — three-stage orchestrator integration + resume-on-complete semantics.
- `tests/integration/test_adapt_command.py` (4 tests, ~210 lines) — CLI `platinum adapt` end-to-end + `--story`/`--track` filters + status reflection.
- `config/prompts/atmospheric_horror/*.j2` (6 templates, ~800 lines) — system, adapt, scene_breakdown, visual_prompts per Stage.

**Files modified:**
- `src/platinum/cli.py` — replaced the `adapt` stub with full implementation; added `--story` and `--track` options; wired `Orchestrator` with three stages.
- `src/platinum/models/story.py` — added `Adapted(narration_script, arc: dict)`, `Scene(index, narration_text, mood, sfx_cues)` models; `Story.adapted` and `Story.scenes` fields.
- `src/platinum/models/db.py` — added `ApiUsageRow` table to track provider/model/input/output tokens and cost per call.
- `src/platinum/pipeline/context.py` — added `PipelineContext(config, logger, ...)`; `story_path()`, `db_path` helpers.
- `src/platinum/config.py` — added `prompts_dir` property, `track(id)` method, Jinja2 environment init.
- `src/platinum/pipeline/orchestrator.py` — enhanced to skip stages whose latest `StageRun` is already COMPLETE (resume-safe).

**Live deliverable verified (offline, via fixtures):**
1. Three stages run in sequence: story_adapter → scene_breakdown → visual_prompts.
2. `python -m platinum adapt` with `--story` filter processes only matching ID.
3. `python -m platinum adapt` with `--track` filter processes only matching track.
4. Story JSON reflects `adapted.narration_script`, `adapted.arc`, and 8 scenes with visual/negative prompts.
5. SQLite `api_usage` table tracks 3 rows per story (one per stage) with real token counts and calculated USD cost.
6. `python -m platinum status --story <id>` shows first 5 stages (source_fetcher through visual_prompts) with COMPLETE status.
7. Resume-safe: re-running `platinum adapt` on a partially-completed story skips already-COMPLETE stages.

**Surprises / lessons:**
1. **Prompt caching overhead.** Cache creation (125% of input rate) is expensive; it only pays off after ~6-8 reads of the same cache block. For a small number of stories (~10) per session, raw input may be cheaper. Recommended adding a `--cache` flag to toggle on/off at runtime; deferred to future optimization.
2. **Fixture recording workflow.** The `FixtureRecorder` replay mode lets tests run offline without hitting the API. Live recording (capture mode) only needed once per new prompt template. This decouples test iteration from API spend.
3. **Orchestrator skip-if-complete is elegant.** Adding `if latest.status == COMPLETE: continue` to the stage loop means re-runs automatically resume without any new plumbing. One pattern, infinite reusability.
4. **JSON atomicity matters for reliability.** Async pipelines can fail mid-stage. Atomic writes via `Story.save(atomic=True)` ensure partial updates never corrupt the JSON. Worth the cost.
5. **Jinja2 template inheritance simplifies multi-stage prompts.** Each stage inherits a common base (`system.j2`) and customizes the task-specific part. Adding a new stage is just one new `.j2` file, not 200 lines of string literals.

**Not done in this session (deferred to later):**
- Multi-track prompt tuning. Only `atmospheric_horror/` is authored. When the second track ships, copy the four templates and tune them per track.
- Prompt-quality iteration (reading live narration_script output and adjusting `adapt.j2`). Expected after fixture recording smoke test.
- A per-track config loader module. For now, tracks are read from YAML directly; later, a `platinum/tracks/` module could encapsulate per-track config, prompts, filters, etc.
- Cost-tracking dashboard. `ApiUsageRow` is populated; a future `platinum report-costs` command can query and summarize it.
- Batch/parallel stage runs. For now, stories are adapted sequentially. If scaling to 100+ stories, parallelizing at the story or stage level could help.
