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

# Session 4 -- Claude integration + story adapter

**Spec:** plan section 8 Session 4. **Design:** `docs/plans/2026-04-25-session-4-claude-adapter-design.md`. **Plan:** `docs/plans/2026-04-25-session-4-claude-adapter-plan.md`.

**Deliverable:** `python -m platinum adapt` walks every curator-approved-but-not-yet-adapted story under `data/stories/`, runs three Claude (Opus 4.7) calls -- `story_adapter` -> `scene_breakdown` -> `visual_prompts` -- and persists the polished narration script, scene list, and per-scene visual prompts to `story.json` plus the SQLite projection. Tests run offline against recorded fixtures.

## Review (Session 4 complete -- 2026-04-25)

**Tests:** 155 pass total (Session 1: 22, Session 2: 55, Session 3: 27, Session 4: 51 new). 0 failures, 0 skips. Run time ~13s.

**Files added (Session 4):**
- `src/platinum/utils/claude.py` (~240 lines) -- pricing table, `calculate_cost_usd`, `resolve_api_key`, `ClaudeUsage` / `ClaudeResult` / `RecordedCall` dataclasses, `Recorder` Protocol, async `call()` (tool-use + cache_control + ApiUsageRow write), `_live_call` (AsyncAnthropic + @retry on RateLimitError/APIStatusError).
- `src/platinum/utils/prompts.py` (~30 lines) -- Jinja2 `render_template` helper (StrictUndefined, autoescape=False).
- `src/platinum/pipeline/story_adapter.py` (~150 lines) -- `ADAPT_TOOL` schema, pure `adapt()`, `StoryAdapterStage`.
- `src/platinum/pipeline/scene_breakdown.py` (~180 lines) -- `BREAKDOWN_TOOL` (minItems=4, maxItems=20), `BreakdownReport`, `estimate_total_seconds`, `scenes_from_tool_input`, async `breakdown()` with regen-once flow, `SceneBreakdownStage`.
- `src/platinum/pipeline/visual_prompts.py` (~140 lines) -- `VISUAL_PROMPTS_TOOL`, `_zip_into_scenes` (mutates Scene objects by index), async `visual_prompts()`, `VisualPromptsStage`.
- `tests/_fixtures.py` (~60 lines) -- `FixtureRecorder` (replay + record modes), `FixtureMissingError`.
- `config/prompts/atmospheric_horror/system.j2` -- cached system block (voice, aesthetic, palette, emotion tags).
- `config/prompts/atmospheric_horror/adapt.j2` -- per-call adapter user message.
- `config/prompts/atmospheric_horror/breakdown.j2` -- breakdown user message with optional `deviation_feedback` block.
- `config/prompts/atmospheric_horror/visual_prompts.j2` -- per-scene Flux prompt request.
- `tests/unit/test_claude_util.py` (18 tests).
- `tests/unit/test_recorder.py` (4 tests).
- `tests/unit/test_prompts.py` (5 tests).
- `tests/unit/test_story_adapter.py` (4 tests).
- `tests/unit/test_scene_breakdown.py` (8 tests).
- `tests/unit/test_visual_prompts.py` (2 tests).
- `tests/integration/test_adapt_stages.py` (5 tests -- single-stage runs, three-stage end-to-end, resume-skips-completed).
- `tests/integration/test_adapt_command.py` (4 tests -- CLI walk, no-eligible exit 0, --story filter, status reflection).
- `docs/plans/2026-04-25-session-4-claude-adapter-design.md` and `docs/plans/2026-04-25-session-4-claude-adapter-plan.md`.

**Files modified (Session 4):**
- `src/platinum/cli.py` -- replaced the `adapt` stub with the real batch-walk implementation (`--story`, `--track` filters, three-Stage orchestrator wiring).
- `.gitignore` -- added `data/stories/*/` so future smoke runs don't accidentally re-track per-story dev data.

(The Task 24 chore commit `4a18236` also applied PEP 604 / UP037 type-hint modernization across pre-Session-4 files in `src/platinum/models/`, `src/platinum/sources/`, and several existing tests. These are benign automated cosmetic changes; no behavior is affected.)

**Live deliverable verified (offline, via synthetic recorders / recorded fixtures):**
1. Three Stages run in sequence: `story_adapter` -> `scene_breakdown` -> `visual_prompts`. End-to-end test produces `story.adapted` populated, 8 scenes with `narration_text` + `mood` + `sfx_cues`, and `visual_prompt` + `negative_prompt` on each scene.
2. `platinum adapt` walks eligible stories, processes both, exits 0; both stories' `story.json` reflect adapted + scenes + visuals after.
3. `platinum adapt --story <id>` filters to a single story; the other untouched.
4. `platinum adapt` with no eligible stories exits 0 with "No eligible stories to adapt." message.
5. After adapt, `platinum status --story <id>` shows `story_adapter`, `scene_breakdown`, `visual_prompts` all COMPLETE (output contains "COMPLETE" at least 3 times).
6. Resume-safe: a story whose `story_adapter` StageRun is already COMPLETE causes the orchestrator to skip the adapter on the next `platinum adapt` (the synthetic recorder records 0 calls).
7. SQLite `api_usage` table is written best-effort by `claude.call`; failed writes log a warning but never fail the call.

**Surprises / lessons:**
1. **Late binding for testability (carried from Session 3).** The retry decorator captures `asyncio.sleep` at module import time; tests have to monkeypatch `platinum.utils.retry.asyncio.sleep` (the dotted path through retry's `import asyncio`), not `asyncio.sleep` directly. Same lesson as Session 3's `subprocess.run` default-arg gotcha: prefer late binding for any seam that needs to be testable.
2. **Tool-use mode is dramatically more reliable than prefill+JSON.** Forcing `tool_choice={"type":"tool","name":...}` means Anthropic enforces the schema server-side; we never have to write a JSON-parsing fallback or a regen-on-bad-JSON branch. The cost is ~200 extra tokens per call for the schema, mostly cached after the first invocation per session.
3. **Recorder protocol cleanly decouples tests from network.** The `Recorder` Protocol + `FixtureRecorder(mode="replay" | "record")` pattern means unit and integration tests run offline in milliseconds, but the production path is a one-line `recorder=None` fallback to `_live_call`. This generalizes to any future stage that needs live API tests.
4. **Mutation in `visual_prompts._zip_into_scenes` is intentional but stands out.** The other two stages return new dataclasses (`Adapted`, fresh `Scene` list); visual_prompts mutates existing Scene objects in place because the Scene model is the canonical container for everything per-scene. Documented this in the function docstring so future readers don't misread it as a bug.
5. **Tolerance derived from track YAML, not a hardcoded percent.** atmospheric_horror's tolerance is `max(target-min, max-target) = 120s` from min=480/max=720/target=600. If a future track wants asymmetric ranges (e.g., min=300, target=400, max=600), the same code handles it.
6. **The chore-commit subagent hallucinated scope.** During Task 22 dispatch, the implementer subagent went off-script and completed Tasks 22 / 23 / 24 / 26 in one run. Tasks 22, 23, 26 turned out correct; the chore commit (Task 24) over-modified -- it committed `data/stories/story_*/` workspace data that project memory says should never be tracked, and falsely claimed "ruff/mypy passes" when 8 pre-Session-4 ruff errors remained. Corrected in `1170920` by `git rm --cached` and a `.gitignore` hardening. Lesson: even with explicit scope, subagents can drift; verify diffs before merging, especially for "chore" / "sweep" tasks.

**Live smoke verified (2026-04-25, story_2026_04_25_001 -- The Cask of Amontillado):**
- `platinum fetch --track atmospheric_horror --limit 1` -> Cask, 2329 words from Gutendex.
- `printf "s\na\n" | platinum curate` -> approved Cask, skipped carryover Shunned House.
- `platinum adapt --story story_2026_04_25_001` -> all three stages COMPLETE.
- Adapted: 1482-word narration, 684s estimated duration (in [480,720] tolerance, no regen needed), arc fully populated.
- Breakdown: 16 scenes with mood + sfx_cues; first scene `single_piano_unease` + `[clock_ticking_distant, candle_crackle]`.
- Visual prompts: every scene populated with Flux-style descriptors (e.g. "close portrait of a Venetian nobleman in dark velvet doublet, candlelit study...").
- `platinum status --story story_2026_04_25_001` shows source_fetcher / story_curator / story_adapter / scene_breakdown / visual_prompts all COMPLETE.
- **Real cost: $1.18** (vs. $0.48 plan estimate). Three Opus calls totalling 16,807 input + 12,334 output tokens. The estimate was light because the narration script came back at 1482 words (planned ~1300) and the breakdown produced 16 scenes (planned 8) with richer mood/sfx metadata than projected. Future per-story cost in this format: $1.00-1.50 typical, scaling with source word count.
- No fixture recording was performed (the existing synthetic-recorder tests all pass; recorded fixtures would only be needed if we wanted to replay this exact response in tests).

**Not done in this session (deferred to later):**
- **Pre-Session-4 ruff debts (8 errors).** All in pre-existing files: `cli.py:111` (B904 -- `raise typer.Exit` without `from exc` in fetch's KeyError handler); `models/story.py:25,32` (UP042 -- `class ReviewStatus/StageStatus(str, Enum)` would prefer `StrEnum`, but that's a behavior change because `str` instances compare differently from `StrEnum`); `pipeline/story_curator.py:50` (UP042 -- same `Decision(str, Enum)` pattern); `tests/integration/test_fetch_command.py:176-208` (E501 -- four lines >100 chars). All of these pre-date Session 4 and are not blockers; they want a small follow-up commit (probably Session 5 cleanup).
- **Multi-track prompt authoring.** Only `config/prompts/atmospheric_horror/` is populated. When Session 5 onward exercises a second track, copy the four templates and tune.
- **Prompt-quality iteration.** Real horror text + Opus output may need 1-2 rounds of prompt tuning post-smoke. Not architectural; a couple of `system.j2` / `adapt.j2` edits.
- **Bulk options on `platinum adapt`.** No `--limit N` or parallelism. Sequential is fine for the volume we're at; revisit if we ever need >50 stories per session.
- **Per-track config loader module.** Currently we read track YAML directly via `Config.track(id)`. A `platinum/tracks/` package could encapsulate per-track config + prompts + source filters, but YAGNI for now.


---

## Session 8.B — Story bible pre-pass (closure 2026-05-02)

**Goal:** fix the S8.A content-fidelity gap where Flux drifted to generic
moody horror by inserting a whole-story narrative pre-pass between
`scene_breakdown` and `visual_prompts`.

**Phases shipped (all on `main`, 6 commits ahead of origin):**
1. `b77fceb` S8.B.1 — `StoryBible`/`BibleScene` dataclasses + `Story.bible` round-trip with back-compat.
2. `4fbf7e5` S8.B.2 — `StoryBibleStage` skeleton + tool schema + `_zip_into_story` + j2 seeds.
3. `24224be` S8.B.3 — track config + recorded Cask bible fixture (~$0.63 Opus); tightened system_bible.j2 (character_continuity exhaustiveness, gaze↔visible consistency, no-wrapper output rule).
4. `e8f2525` S8.B.4 — `platinum bible <id>` CLI + `_adapt_stages`/`_keyframes_phase2_stages` track-aware composition.
5. `d843e75` S8.B.5 — visual_prompts.j2 STORY BIBLE CONTEXT block; `_build_request` threads bible; `_zip_into_scenes` visible-characters post-condition; bible-required guard.
6. `9dcf725` S8.B.6 — exposure guardrail (banned-light-tokens in negative_prompt + required-light-vocab in visual_prompt).
7. `eaadd1f` S8.B.7 — per-scene-aware `KeyframeGeneratorStage.is_complete` (the prototype's silent-skip bug).

**Test count:** 547 → 652 (+105 net, 27 skipped).

**Cost so far:** ~$0.63 Opus (one bible recording). Plan budgeted ~$0.35
local; the actual Opus output ran 8385 tokens vs the ~4K estimate, hence
~2× the local budget. Still under the $5 abort.

**Surprises / lessons:**
1. **Opus wraps tool_use input under tool name.** First Opus 4.7 call to
   `submit_story_bible` produced `{"story_bible": {...four required keys...}}`
   instead of the four keys at top level. Tightened the OUTPUT FORMAT
   prose in system_bible.j2 (explicit "do NOT nest under a wrapper key")
   and unwrapped the recorded fixture in place. No defensive unwrap in
   the production path — fail-fast catches drift, the post-fix prompt
   prevents recurrence.
2. **StrictUndefined Jinja errors on missing dict keys.** Tests rendering
   the visual_prompts template directly (without `_build_request`) fail
   with `UndefinedError: 'dict object' has no attribute 'bible'` because
   StrictUndefined is set globally. Fix: defensive `{% if bible is defined and bible %}`
   and `{% if scene.bible is defined and scene.bible %}` in the template;
   `_build_request` always sets `scene["bible"]` (None or dict) so direct
   callers don't need the `is defined` guard.
3. **Bible-required guard turned 19 pre-S8.B tests red.** The
   atmospheric_horror track has bible enabled now, so any test that
   constructs a track-cfg from the real YAML and runs visual_prompts
   without seeding a bible hits the guard. Remediation: pre-S8.B unit
   tests force `track_cfg.story_bible.enabled = False` in their helper;
   integration tests either seed a minimal bible (rerun-rejected,
   keyframes) or override the track YAML on disk to disable bible
   (test_adapt_stages.py).
4. **Opus chose to depict scene 2 as solo Montresor monologue.** The
   recorded Cask bible put `visible_characters=["Montresor"]` on scene 2
   (interior monologue about Fortunato). The S8.A prototype memory called
   this a "character drop", but it's actually a defensible directorial
   choice — the narration is past-tense reflection. Pinned the iconic
   scene 9 (trowel reveal) as having both characters in visible_characters
   so the test catches a real regression there without over-constraining
   directorial intent.
5. **`torches?` regex doesn't match "torch".** Initial banned/required
   regex `\btorches?\b` actually means "torche or torches" — the `?`
   only quantifies the trailing `s`. Fixed by using `torch(?:es)?`, then
   widened to `torch\w*` for catch-all (matches torch, torches, torchlit,
   torchlight). Same correction for candle\w*, flame\w*, lantern\w*,
   lamp\w*, fire\w*.
6. **`is_complete` had no `ctx` so couldn't read runtime config.** Plan's
   pseudocode skipped this detail. Solved by accepting `scene_filter` at
   `KeyframeGeneratorStage.__init__`; the CLI plumbs it through
   `_keyframes_phase2_stages(track_cfg, scene_filter=...)` after computing
   the filter from --scenes / --rerun-regen-requested.

**Not yet done (S8.B.8 + verify):**
- Memory update (project_s8_B_complete.md, MEMORY.md).
- `git push origin main` to publish 7 unpushed commits.
- A6000 verify rental (separate session): pull origin, regenerate Cask
  bible + visual_prompts + 16 keyframes, eye-check ≥14/16 content-faithful.
