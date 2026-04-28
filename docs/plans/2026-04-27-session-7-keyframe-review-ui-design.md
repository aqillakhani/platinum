# Session 7 — Keyframe review Flask UI (human gate) + full Cask entry-test render

**Date:** 2026-04-27
**Predecessor:** Session 6.4 (closed 2026-04-27 — visual_prompts darkness-density cap + 3 plumbing fixes; 317 tests).
**Successor:** Session 8 (Wan 2.2 I2V video generator + last-frame chaining).
**Track first hit:** atmospheric_horror (`story_2026_04_25_001` — The Cask of Amontillado).

## 1. Context and motivation

S6.3 + S6.4 collectively rebuilt the Flux workflow (FluxGuidance + ModelSamplingFlux), added a brightness gate + a Canny-edge subject gate before LAION scoring, recalibrated atmospheric_horror's per-track thresholds, and capped darkness-modifier density in the visual_prompts template. S6.3 Phase 2 closed with Cask scenes 8 + 16 viable on A6000 (closure ≥2/3 met); S6.4 Item 1 ran the visual_prompts re-spec ($0.71 Claude API, two iterations) targeted at scene 1 specifically. The Cask story.json with iter-2 visual_prompts sits on local disk, gitignored, ready for S7.

Two pieces of work pile up at the S7 gate:

1. **S7 entry test — full Cask 16-scene render on a fresh A6000.** This is the first end-to-end validation that the S6.3 + S6.4 quality work produces shippable keyframes for an entire story rather than a hand-picked 2-of-3 sample. The full render also produces the input data that the new review UI will display.
2. **S7 main work — the keyframe review Flask UI** specified by the master plan §8 Session 7: a grid view of all scenes, per-scene Approve / Reject (re-prompt) / Regenerate (same prompt, new seed) actions, batch-approve above an aesthetic threshold, SQLite + story.json persistence. Deliverable: `python -m platinum review keyframes <story_id>` opens a browser; reviewer can clear a 60-scene film in under 15 minutes.

The two pieces are wired together as Phase 1 (offline TDD) → Phase 2 (live smoke) — the same pattern as S6.1, S6.2, and S6.3. Phase 1 ships independently of vast.ai cost and risk; Phase 2 is the natural validation gate at session close.

Wan 2.2 video generation is **explicitly Session 8**, not S7 — earlier project memory conflated them. The master plan (`C:\Users\claws\.claude\plans\i-added-a-prd-concurrent-book.md` line 436) puts Wan 2.2 in S8.

## 2. Brainstorming forks (decisions logged)

Captured for the same reason every Platinum session captures them: future-me re-reading this in S8+ needs to know *what* was decided AND *why*, so edge cases can be judged rather than re-litigated.

### Fork 1: Session scope — both phases this session, UI only, or render only?

| Option | Cost | Risk |
|---|---|---|
| (a) **Both phases this session: offline UI build + live entry-test render** | ~$2 vast.ai + ~75 min rental + ~3-4 hours TDD | UI build proceeds independently of rental availability; render is the natural session-close validation |
| (b) UI only; defer entry-test render to a separate session | ~0 this session, ~$2 next session | Two-session split; loses end-to-end validation rhythm |
| (c) Render first, UI in next session | ~$2 this session | Blocks UI work on rental; if render reveals quality regression we'd retro debt before any UI work |

**Chose (a).** Mirrors S6.1 / S6.2 / S6.3 rhythm: Phase 1 offline TDD ships independently; Phase 2 live smoke at close validates the full quality stack and exercises the new review UI against real data.

### Fork 2: Regenerate / re-prompt mechanics — mark-only (decoupled), live, or hybrid?

| Option | Pros | Cons |
|---|---|---|
| (a) **Mark-only.** UI updates `story.json`; user runs `platinum keyframes <id> --rerun-regen-requested` later | UI works whether GPU box is up or down; minimizes box rental time; review can be entirely local | Two-step workflow; user can't see a regen take effect during review |
| (b) Live. UI calls keyframe_generator in a background worker; spinner; in-place update | Single-step; immediate feedback; "regenerate, fix, regenerate" loop is fast | Box has to stay up for the full review (~$0.42/hr extra); UI breaks if box destroyed mid-review; harder to test offline |
| (c) Hybrid. Mark-only by default; "Regenerate now" button surfaces when `PLATINUM_COMFYUI_HOST` pings green | Best of both; instant feedback when convenient | ~30% more UI code; two code paths; "is box up" probe is one more failure mode |

**Chose (a).** The 15-min-for-60-scenes target only works if review is reading-speed (decisions in <15s/scene). GPU re-renders are 60-90s each — they'd blow the target if inline. Offline-first matches every prior S6.x. (c) revisitable in a future session if "wait, I want regen-now" becomes a real pain point.

### Fork 3: Reject (re-prompt) semantics — capture feedback for Claude, hand-edit prompt, or single-click?

| Option | Pros | Cons |
|---|---|---|
| (a) **Capture user feedback as text → CLI re-runs Claude with feedback baked into `deviation_feedback` block** | Matches "Claude is the prompt engineer" architecture; user steers with intent | ~$0.34 per re-prompt run (full visual_prompts re-call) |
| (b) Inline textarea edit of `visual_prompt` directly | Dead simple, no Claude cost, immediate | User hand-authors Flux prompts; loses value of visual_prompts stage |
| (c) Single-click reject (no feedback) | Simplest UI | Blind regen; same prompt context often produces same prompt twice |

**Chose (a).** "Claude reads my feedback and rewrites" is what makes a review UI feel like a tool, not a chore. The `deviation_feedback` block extension to `visual_prompts.j2` mirrors the existing pattern from `scene_breakdown.j2` (S4); also useful for future S6.4-style prompt-debt loops.

### Fork 4: Regenerate seeds — `regen_count` field, random offset, or wipe and re-derive?

| Option | Pros | Cons |
|---|---|---|
| (a) **Per-scene `regen_count` field; seed becomes `_seeds_for_scene(scene_index, n, regen_count=count)`** | Deterministic, history-preserving, traceable; observable in `story.json` | One new field on `Scene` |
| (b) Random offset stored on Scene per regen | Reproducible from saved offset | Loses cross-regen determinism; can't replay a specific past regen |
| (c) Wipe and re-derive (fresh randomness inside `_seeds_for_scene`) | Simplest | Breaks fixture-based tests; loses reproducibility entirely |

**Chose (a).** Matches the deterministic / replayable / fixture-friendly pattern from S6.x. Migration is small (one optional field, default 0); existing keyframe_generator tests stay green because the function signature is keyword-only with default.

### Fork 5: Per-scene candidate display — show only auto-selected, show all 3, or default + toggle?

| Option | Pros | Cons |
|---|---|---|
| (a) Auto-selected only | Simplest UI; matches plan wording verbatim ("thumbnail" singular) | No override path; trapped if auto-pick is wrong |
| (b) All 3 candidates always visible per scene | Maximum agency | 3-up mini-grids clutter the gallery; harder to scan 16 scenes |
| (c) **Auto-selected by default; "View alternatives" toggle reveals other 2 with their LAION scores; click → swap selection** | Clean default; agency on demand; one disclosure widget | One extra route + one disclosure UI element |

**Chose (c).** Auto-selection is usually right (S6.3 calibrations are tuned for atmospheric_horror), so default-collapsed keeps the grid scannable. Occasional 5.97-LAION-but-better-eye-test override is supported without cluttering normal flow.

### Fork 6: Sequencing within Phase 1

Phase 1 has roughly four tracks: data-model migration (Scene fields + seed change), pure-core decisions, Flask app + image serving, integration plumbing (`--rerun-rejected`, `--rerun-regen-requested`). Two ordering options:

(a) Strict TDD layered: model → pure core → Flask → CLI integration. Each layer's tests must pass before the next starts.
(b) Vertical slice per action: implement Approve end-to-end (model + decisions + route + JS), then Regenerate, then Reject, etc.

**Chose (a).** Layered TDD matches every prior S6.x plan and lets each piece be reviewed independently. Vertical-slice would mean leaving partial state in the model + decisions layers between commits, which complicates rollback if a slice misfires.

## 3. Architecture

### 3.1 File layout

```
src/platinum/review_ui/
├── __init__.py                  # empty package marker
├── app.py                       # Flask app factory, routes, image serving, browser launch
├── decisions.py                 # PURE: apply_approve, apply_regenerate, apply_reject,
│                                #       apply_swap_candidate, apply_batch_approve_above,
│                                #       finalize_review_if_complete
├── templates/
│   ├── base.html                # page chrome + inline <script> (~120 lines vanilla JS)
│   └── keyframe_gallery.html    # Jinja2 loop over story.scenes
└── static/
    └── style.css                # ~200 lines hand-written; CSS Grid 3-up at desktop
```

### 3.2 Pure-core / impure-shell split

`decisions.py` is the pure core — every function takes a `Story` (and optionally a scene id, action params) and returns a new mutated `Story` (or, more precisely, mutates the passed Story since the dataclasses are not frozen, and returns it for chaining). No I/O, no Flask, no SQLite. Trivially unit-testable.

`app.py` is the impure shell — Flask routes that load `story.json`, call into `decisions.py`, write back via `Story.save` (atomic) + `sync_from_story` (SQLite projection), return JSON.

This mirrors `story_curator.apply_decision` (S3) and `keyframe_generator.generate_for_scene` (S6) exactly.

### 3.3 Routes

All bound to `127.0.0.1` only. Default port `5001` (avoid 5000 conflict with anyone running default Flask on the same machine).

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Redirects to `/story/<id>` if exactly one story matches the CLI-supplied id; else 400 |
| GET | `/story/<story_id>` | Renders `keyframe_gallery.html` |
| GET | `/api/story/<story_id>` | JSON snapshot — used by JS to refresh after mutations |
| GET | `/image/<story_id>/<path:relpath>` | Serves PNGs from `data/stories/<id>/keyframes/`; uses `werkzeug.security.safe_join` |
| POST | `/api/story/<story_id>/scene/<scene_id>/approve` | → `apply_approve` |
| POST | `/api/story/<story_id>/scene/<scene_id>/regenerate` | → `apply_regenerate` |
| POST | `/api/story/<story_id>/scene/<scene_id>/reject` | body `{"feedback": "..."}` → `apply_reject` |
| POST | `/api/story/<story_id>/scene/<scene_id>/select_candidate` | body `{"index": N}` → `apply_swap_candidate` |
| POST | `/api/story/<story_id>/batch_approve` | body `{"threshold": 6.0}` → `apply_batch_approve_above` |

Every POST executes: `Story.load` → mutate (pure call) → `finalize_review_if_complete` → `Story.save` → `sync_from_story` → return JSON of the touched scene + rollup `{"approved": N, "pending": M, "rejected": K, "regen_requested": J}`.

### 3.4 Frontend (no client-side framework)

- `base.html` — page chrome, title, links to `style.css`, inline `<script>` block with a tiny `fetch`-based API helper.
- `keyframe_gallery.html` — Jinja2 loop over `story.scenes`. Each scene tile contains: thumbnail (`<img src="/image/...">`), prompt blurb (truncated; expand on click), score badge, status pill, action buttons.
- "View alternatives" → CSS toggle (`<details>`) reveals 3-up mini-grid of all candidates with their LAION scores; click → POST `select_candidate`.
- "Reject" button → opens an inline `<dialog>` with a `<textarea>` for feedback; submit → POST `reject`.
- "Approve all ≥ threshold" → top-bar control with a number input (default = track YAML's `aesthetic_min_score`); POST `batch_approve`.
- All POSTs use `fetch` + JSON; on response, the touched tile re-renders from the JSON via vanilla DOM update. No client-side library or framework.
- JS budget: ~120 lines, all inline. CSS: ~200 lines, hand-written.

### 3.5 Image serving security

`safe_join(data_root / story_id / "keyframes", relpath)` blocks `../` escape; on miss returns 404 not 500. No directory listing exposed.

### 3.6 No auth, no CSRF

Local 127.0.0.1, single-user, ephemeral process. CSRF tokens on a localhost-only single-user app are theatre.

### 3.7 Dependency

Add `flask` to `pyproject.toml` core dependencies. Per project memory, `fastapi` already lives there (S6.1 score_server); having one Flask + one FastAPI is fine — different roles (Flask = local human UI; FastAPI = remote machine-to-machine score server).

## 4. Data model

### 4.1 Existing (no change needed)

`src/platinum/models/story.py:25-29` already defines:
```python
class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGENERATE = "regenerate"
```

`Scene.review_status: ReviewStatus = ReviewStatus.PENDING` (line 161) already exists, defaults to PENDING, and is serialized/deserialized in `to_dict` / `from_dict`.

`Scene.keyframe_candidates: list[Path]` (line 151) and `Scene.keyframe_scores: list[float]` (line 152) are already populated by `keyframe_generator.generate()` (lines 320-321 of `keyframe_generator.py`). These drive the "view alternatives" toggle and batch-approve-by-threshold.

### 4.2 New `Scene` fields

```python
@dataclass
class Scene:
    ...
    review_status: ReviewStatus = ReviewStatus.PENDING        # existing
    review_feedback: str | None = None                          # NEW — populated when REJECTED
    regen_count: int = 0                                        # NEW — bumped on each REGENERATE
```

Both fields ship through `to_dict` / `from_dict` with the obvious projections (`"review_feedback"`, `"regen_count"`). Backward compat: `from_dict` defaults both to None / 0 when absent.

### 4.3 Seed function change

`src/platinum/pipeline/keyframe_generator.py::_seeds_for_scene`:

```python
def _seeds_for_scene(
    scene_index: int, n: int, *, regen_count: int = 0
) -> tuple[int, ...]:
    """Deterministic seeds: scene_index*1000 + regen_count*100 + offset."""
    return tuple(scene_index * 1000 + regen_count * 100 + i for i in range(n))
```

Backward-compatible: keyword-only `regen_count=0` reproduces existing seed sequences exactly. Capacity: up to 100 regens × up to 100 candidates per scene before collision, which is more than enough for the lifetime of any single story.

`generate_for_scene` and `generate` need a small plumbing change to thread `scene.regen_count` into the seed call: `seeds = _seeds_for_scene(scene.index, n_candidates, regen_count=scene.regen_count)`.

### 4.4 StageRun semantics for `keyframe_review`

- No StageRun appended until ALL scenes have `review_status == APPROVED`.
- When all approved (detected in `finalize_review_if_complete`), append:
  ```python
  StageRun(
      stage="keyframe_review",
      status=COMPLETE,
      started_at=<earliest scene decision timestamp>,  # tracked via review_gate metadata
      completed_at=now,
      artifacts={
          "approved_count": len(story.scenes),
          "regen_total": sum(s.regen_count for s in story.scenes),
          "rejected_total": <count of REJECTED → REGENERATE transitions>,
      },
  )
  ```
- Mid-review state lives entirely on per-scene `review_status` + `regen_count` + `review_feedback` fields. The stage_run is a derived rollup written exactly once when closure conditions are met.
- Mirrors `story_curator`'s "all-or-nothing" pattern (S3).

### 4.5 `story.review_gates["keyframe_review"]`

Mirror of `stage_run.artifacts`, written at the same moment for human-readable visibility:
```json
{
  "keyframe_review": {
    "completed_at": "2026-04-27T...",
    "reviewer": "user",
    "approved_count": 16,
    "regen_total": 3,
    "rejected_total": 1
  }
}
```
Matches the convention from `review_gates["curator"]` (S3).

### 4.6 SQLite sync

The existing `sync_from_story` projects `story.json` → SQLite `scenes` and `stage_runs` tables. The plan will detail the exact migration during planning (likely just adding two columns to the `scenes` table — `review_feedback TEXT NULL`, `regen_count INTEGER DEFAULT 0` — and reflecting them in `sync_from_story`'s scene-row builder). If the existing scene projection stores most fields as JSON blobs, the migration may be zero-column.

## 5. CLI surface + integration glue

### 5.1 New `review` sub-app

A Typer sub-app `review` is added under the main app, with `keyframes` as the first subcommand. S15 will later add `final` to the same sub-app.

```
python -m platinum review keyframes <story_id> [--port 5001] [--no-browser] [--threshold N]
```

- `--port`: Flask binding port (default 5001).
- `--no-browser`: skip `webbrowser.open()`; useful for headless / scripted runs.
- `--threshold N`: pre-fills the batch-approve input box with `N` (does NOT auto-trigger).

Errors: missing story id → Typer Exit 1 with "story <id> not found"; story has no scenes / no keyframes → Typer Exit 1 with helpful next-step message ("run `platinum keyframes <id>` first").

### 5.2 `platinum keyframes <id> --rerun-regen-requested`

A new flag on the existing `keyframes` command:

1. Builds `scene_filter = {s.index for s in story.scenes if s.review_status == ReviewStatus.REGENERATE}`.
2. Sets `runtime.scene_filter` (existing knob from S6.1 — no new code path needed) and runs `keyframe_generator`.
3. The stage already passes `scene.regen_count` into `_seeds_for_scene` (per §4.3), so seeds differ from the prior run.
4. After successful generation, sets `review_status = ReviewStatus.PENDING` for each regenerated scene (so the UI re-surfaces them for re-review). Atomic save + SQLite sync.
5. Empty `scene_filter` → Typer Exit 0 with "no scenes flagged for regeneration."

### 5.3 `platinum adapt --rerun-rejected`

A new flag on the existing `adapt` command:

1. Builds `scene_filter = {s.index for s in story.scenes if s.review_status == ReviewStatus.REJECTED}`.
2. Collects each rejected scene's `review_feedback`.
3. Renders `visual_prompts.j2` with a new optional `deviation_feedback` block (mirrors `scene_breakdown.j2` from S4):
   ```jinja
   {% if deviation_feedback %}
   The reviewer flagged the following scenes for revision. For each one,
   address the feedback while keeping the scene's narration intent and the
   atmospheric_horror track aesthetic:
   {% for entry in deviation_feedback %}
   - Scene {{ entry.index }} (current prompt: "{{ entry.current_prompt }}")
     Feedback: {{ entry.feedback }}
   {% endfor %}
   {% endif %}
   ```
4. Single Claude call processes the whole story (input already contains all narration_text — needed for cross-scene visual consistency). Output is new visual_prompts for all 16 scenes.
5. **Apply selectively**: for scenes in `scene_filter`, write the new `visual_prompt` + `negative_prompt`, clear `keyframe_path`, set `review_feedback = None`, set `review_status = ReviewStatus.REGENERATE`. For scenes NOT in `scene_filter`, IGNORE the new prompts — APPROVED and PENDING scenes retain their existing visual_prompts byte-identical.
6. Cost: ~$0.34 per re-run (full visual_prompts call). Acceptable for typical 1-3 reject scenarios; if it becomes a pain point, a per-scene mini-call optimization is a separate session.

**Why apply selectively?** A user who rejected only scene 1 doesn't expect already-approved scenes 2-16 to get reshuffled. Stability beats cost-optimization.

### 5.4 The user workflow loop

```
1. platinum keyframes <id>                            # render all 16 scenes (Phase 2)
2. platinum review keyframes <id>                     # browser opens, review pass
3. (user clicks; some scenes APPROVED, some REJECTED, some REGENERATE)
4. platinum adapt --rerun-rejected                    # re-prompt rejected scenes via Claude
5. platinum keyframes <id> --rerun-regen-requested    # re-render REGENERATE scenes
6. (browser still open; refresh; re-review the regenerated scenes)
7. Loop 4-6 until all APPROVED.
8. (browser shows "all 16 approved"; KeyframeReviewStage marked COMPLETE.)
```

`finalize_review_if_complete` triggers automatically after every POST. Once all scenes are APPROVED, the StageRun is appended once (idempotent — subsequent POSTs that would re-finalize see `latest_stage_run("keyframe_review")` is COMPLETE and skip).

## 6. Phase 1 deliverables (offline TDD, this session, no GPU spend)

### 6.1 Files

**Added:**
- `src/platinum/review_ui/__init__.py`
- `src/platinum/review_ui/app.py`
- `src/platinum/review_ui/decisions.py`
- `src/platinum/review_ui/templates/base.html`
- `src/platinum/review_ui/templates/keyframe_gallery.html`
- `src/platinum/review_ui/static/style.css`
- `tests/unit/test_review_decisions.py`
- `tests/unit/test_review_app.py`
- `tests/unit/test_seed_regen_count.py`
- `tests/integration/test_review_command.py`
- `tests/integration/test_keyframe_rerun_regen_requested.py`
- `tests/integration/test_adapt_rerun_rejected.py`
- `tests/integration/test_review_full_loop.py`

**Modified:**
- `src/platinum/models/story.py` — `Scene.review_feedback`, `Scene.regen_count` fields + serialization.
- `src/platinum/pipeline/keyframe_generator.py` — `_seeds_for_scene` regen_count parameter; thread `scene.regen_count` through `generate_for_scene` and `generate`.
- `src/platinum/pipeline/visual_prompts.py` — accept `scene_filter` + `deviation_feedback` runtime knobs; selective apply on REJECTED scenes only.
- `src/platinum/cli.py` — register `review` Typer sub-app + `keyframes` subcommand; add `--rerun-rejected` flag to `adapt` and `--rerun-regen-requested` flag to `keyframes`.
- `config/prompts/atmospheric_horror/visual_prompts.j2` — append optional `deviation_feedback` block.
- `pyproject.toml` — add `flask` to core dependencies.
- `src/platinum/models/db.py` (if needed) — migration for `review_feedback` + `regen_count` projections.

### 6.2 Tests

| Suite | File | Count | Coverage |
|---|---|---|---|
| Unit | `test_review_decisions.py` | ~14 | apply_approve / apply_regenerate / apply_reject / apply_swap_candidate / apply_batch_approve_above / finalize_review_if_complete; idempotency; status transitions; regen_count bump; threshold filtering ignoring already-decided scenes |
| Unit | `test_review_app.py` | ~10 | Routes via `app.test_client()`: GET/POST shapes, `safe_join` path-traversal protection on `/image/...`, 400 on missing reject feedback, 404 on invalid candidate index |
| Unit | `test_seed_regen_count.py` | 3 | `_seeds_for_scene` unchanged at `regen_count=0`, distinct at `regen_count=1`, deterministic per regen_count |
| Integration | `test_review_command.py` | 3 | CliRunner: missing-story exit 1, `--no-browser` skips webbrowser.open, app starts and serves a request before clean shutdown (thread + immediate POST) |
| Integration | `test_keyframe_rerun_regen_requested.py` | 4 | scene_filter built from REGENERATE-status only, bumped seeds, status flip to PENDING, empty-set exits 0 |
| Integration | `test_adapt_rerun_rejected.py` | 4 | `deviation_feedback` block rendered with per-scene feedback (FixtureRecorder replay), output applied ONLY to REJECTED scenes (APPROVED scenes' visual_prompts byte-identical), feedback cleared + status flipped REJECTED → REGENERATE post-call, empty-set exits 0 |
| Integration | `test_review_full_loop.py` | 2 | End-to-end via `decisions.py` + Stage glue: full happy path (all approve), then a one-reject-one-regen loop ending in COMPLETE |

**Total:** ~40 net new tests. Some can be parametrized; expect 30-35 actual test functions.

### 6.3 Quality gates

- `pytest -q` passes (existing 317 + new ~30-35 = ~347-352 total).
- `ruff check src tests scripts` clean.
- `mypy src/` unchanged from S6.4 baseline (still the 2 pre-existing deferrals: `config.py:15` yaml stubs, `sources/registry.py:30` SourceFetcher call-arg).
- `pip install -e .` resolves with the new `flask` dep.
- Manual smoke (Phase 1 close): launch app against a synthetic 3-scene story fixture, click approve / reject (with feedback) / regenerate / view alternatives / batch approve / image serve. Verify `story.json` mutations after each.

## 7. Phase 2 — A6000 entry-test smoke runbook

| Step | Action | Notes |
|---|---|---|
| 1 | Local: confirm `data/stories/story_2026_04_25_001/story.json` has S6.4 iter-2 visual_prompts | Per S6.4 memory it does; sanity-grep one scene's prompt for "candlelight catching" lit-pixel phrasing |
| 2 | Rent A6000 via `vastai search offers ...` then `vastai create instance ...` | Filter from S6.3 memory: `gpu_name=RTX_A6000 cpu_ram>=64 disk_space>=80 verified=true`; sort by `dph_total`. Confirm cost with user before spinning up. |
| 3 | SSH in; `bash vast_setup.sh` | Now provisions `p311` conda env per S6.4 |
| 4 | SCP local `story.json` → box (`data/stories/story_2026_04_25_001/`) | Gitignored content |
| 5 | `python scripts/preflight_check.py` | Workflow signature must match `origin/main`'s `flux_dev_keyframe.json` |
| 6 | `/opt/conda/envs/p311/bin/python -m platinum keyframes story_2026_04_25_001` | Full 16-scene run, no `--scenes` filter |
| 7 | SCP `data/stories/story_2026_04_25_001/keyframes/` back to local | All 48 candidate PNGs + selected per-scene keyframes |
| 8 | `vastai destroy instance <id>` | Mandatory teardown — box not needed for review UI |
| 9 | Local: `platinum review keyframes story_2026_04_25_001` | Browser opens to `/story/<id>` |
| 10 | Eye-check 16 scenes; click decisions | Apply mental gold standard from S6.3 (chiaroscuro that reads as a portrait, not a fade-to-black) |
| 11 | Count APPROVED first-pass | **Closure: ≥14 of 16 approveable on first pass** |
| 12 | If ≥14: optionally exercise rerun-rejected loop on the remaining 2 to validate full UX | Single full-loop iteration is a real-world test of `--rerun-rejected` and `--rerun-regen-requested` |
| 13 | If <14: log failure patterns; defer iteration to S7.1 retro debt | Same retro pattern as S6.2 → S6.3 |

**Phase 2 cost estimate:** ~$1.30-1.80 on A6000 ($0.42/hr × 60-75 min: ~12 min provisioning + 35-45 min full-Cask render + ~5 min download + immediate teardown). No second rental needed unless rerun-rejected validation also requires GPU; in that case, factor in another ~$0.30-0.50 for a brief second rental.

**Closure conditions for S7:**
- Phase 1 ships: all tests green, ruff + mypy clean, manual smoke OK.
- Phase 2 closure: ≥14/16 Cask scenes approveable on first pass.
- Cumulative: S8 (Wan 2.2 I2V video) is unblocked using the approved keyframes as inputs.

## 8. Out of scope (deferred)

- **S15 final review UI.** Separate session per master plan §8 Session 15. Will reuse `review_ui/` package with a new `final_review.html` template and new routes; foundation laid by S7 carries forward.
- **Browser-level Playwright tests.** Defer until/unless a UX issue surfaces in Phase 2 manual smoke that's hard to reproduce by hand.
- **Multi-story review per CLI invocation.** S7 supports one story at a time; multi-story queue is YAGNI for v1.
- **Per-scene mini visual_prompts call.** The full-story re-run at $0.34 is acceptable; per-scene optimization deferred until a session demonstrates it as a real cost burden.
- **Live "regenerate while box is up" mode.** UX option C from Fork 2; explicitly rejected this session. Revisit if "wait, I want regen-now" becomes a real pain point.
- **Mobile/tablet responsive layout.** Desktop-only assumption; matches single-user-local hosting model.
- **Multi-user / shared review state.** Single-user, single-machine, single-process. No collaboration features.
- **Real-time progress for `--rerun-*` modes.** They're CLI commands that print to stdout when done. UI does not show "regenerating..." for in-flight CLI work.

## 9. Lessons inherited / forward

### 9.1 Patterns reinforced (carried forward unchanged)

1. **Late binding for testability** — the Flask app uses an app-factory pattern; `decisions.py` is pure and dependency-injectable. Same lesson as S3's `subprocess.run`, S4's `asyncio.sleep`, S5's mediapipe factory, S6's transport/scorer injection.
2. **Recorder/Fake protocol pattern** — `test_adapt_rerun_rejected.py` reuses `FixtureRecorder` (S4); no new Protocol introduced.
3. **Pure-core / impure-shell at the Stage / Route boundary** — `decisions.apply_*` functions are pure; Flask routes are the impure shell. Same pattern as S3's `apply_decision` / `curate`, S6's `generate_for_scene` / `KeyframeGeneratorStage`.
4. **Per-stage atomic save** preserved — `Story.save` is called inside the route handler after every mutation; its tmp + `os.replace` atomic-write pattern guarantees `story.json` coherence even mid-process.
5. **Quality gates ordering: brightness → subject → LAION** unchanged in this session; the keyframe_generator gate stack remains as S6.3 left it.
6. **Subagent scope-drift** mitigations (S6.4 lessons) — for any subagent dispatch in this plan, use single-task prompts with explicit anti-drift language ("do exactly this and stop"); use grep verification steps for wording-fidelity tasks; drive directly when wording matters.

### 9.2 New patterns surfaced this session (to log forward)

1. **Pure-core mutation on a mutable dataclass.** `Scene` is not frozen, so `decisions.apply_*` mutates the passed scene in-place and returns the (same) Story. This is a deliberate departure from the "always return a new object" instinct because Story is the document-root and reconstructing it from a fresh scene list per call would force a deep copy. Tests assert post-state by re-reading `story.scenes[index]` — same shape as `keyframe_generator.generate` mutating scenes in place.
2. **Vanilla JS for an internal review UI.** Adding a JS framework (HTMX, Alpine, React) for a ~120-line UI is over-engineering. The fetch + DOM-update pattern is fine at this scale and keeps dep-tree clean.
3. **Selective apply on whole-story regen output.** `--rerun-rejected` runs visual_prompts for all 16 scenes' worth of context but applies output only to REJECTED scenes. Pattern likely recurs in S8 (`--rerun-rejected` for video), S13 (audio), S14 (color grade) — anything where Claude/Comfy needs cross-scene context but the user wants to update only a subset.
4. **`safe_join` for any user-content static serving.** Generalizes: any future endpoint that serves files from a user-controlled subtree (S15 final review's `graded.mp4`, S16 thumbnails) needs the same path-traversal protection.

## 10. Reference

- **Master plan:** `C:\Users\claws\.claude\plans\i-added-a-prd-concurrent-book.md` §8 Session 7 (line 431).
- **PRD:** `C:\Users\claws\OneDrive\Desktop\platinum\short_film_pipeline_prd.md`.
- **Predecessor design:** `docs/plans/2026-04-27-session-6.4-retro-debt-design.md`.
- **Predecessor plan:** `docs/plans/2026-04-27-session-6.4-retro-debt-plan.md`.
- **Phase 2 runbook (template):** `docs/runbooks/vast-ai-keyframe-smoke.md` (S6.3 version; will be updated for S7 in Phase 1's closeout).
- **Cask story id:** `story_2026_04_25_001` (16 scenes, visual_prompts iter-2 from S6.4).
- **A6000 search filter:** `vastai search offers 'gpu_name=RTX_A6000 cpu_ram>=64 disk_space>=80 verified=true' -o 'dph_total'`.
- **Flask docs (for reference during plan execution):** Flask 3.x app factory pattern, `werkzeug.security.safe_join`.
