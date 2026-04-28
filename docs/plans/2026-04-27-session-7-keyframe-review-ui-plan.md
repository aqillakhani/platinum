# Session 7 Keyframe Review UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the Flask keyframe review UI (`python -m platinum review keyframes <id>`) with per-scene Approve / Reject (re-prompt) / Regenerate (same prompt, new seed) + batch-approve-above-threshold + view-alternatives toggle, plus the `--rerun-rejected` and `--rerun-regen-requested` CLI flags that close the review→re-render loop. Validate end-to-end at session close with a full Cask 16-scene render on A6000.

**Architecture:** Strict TDD, layered (model → pure-core → Flask → CLI → templates/JS → closeout → live smoke). Mark-only persistence (UI mutates `story.json`; CLI does GPU work later). Pure-core / impure-shell at the route boundary mirrors S3 `apply_decision` and S6 `generate_for_scene`. Vanilla JS, no client framework. Local 127.0.0.1:5001, no auth.

**Tech Stack:** Python 3.11+ • Flask 3.x • Jinja2 • SQLAlchemy (existing) • pytest • ruff • mypy • Anthropic SDK (Opus 4.7, for `--rerun-rejected`) • Typer (existing CLI).

**Design source:** `docs/plans/2026-04-27-session-7-keyframe-review-ui-design.md` (commit `304ac46`).

---

## Pre-flight checklist (run before Task 1)

These are sanity checks that catch a stale tree before editing. Each line should produce the noted output; if not, halt and triage.

- [ ] **Working tree clean.** `git status` → `nothing to commit, working tree clean`. (S6.4 closed clean; we should still be clean.)
- [ ] **On `main`, up to date with `origin/main`.** `git rev-parse --abbrev-ref HEAD` → `main`. `git fetch && git status -uno` → `Your branch is up to date with 'origin/main'.`
- [ ] **317 tests pass.** `python -m pytest -q` → `317 passed in <N>s`. (S6.4 baseline.)
- [ ] **ruff clean.** `python -m ruff check src tests scripts` → `All checks passed!`
- [ ] **mypy unchanged.** `python -m mypy src` → `Found 2 errors` (the two pre-existing deferrals: `config.py:15` yaml stubs, `sources/registry.py:30` SourceFetcher call-arg). Anything else means baseline drifted.
- [ ] **Cask story.json present locally.** `ls data/stories/story_2026_04_25_001/story.json` exists. (Local-only; gitignored. Required for Phase 2 entry-test smoke; not needed for Phase 1.)
- [ ] **Design doc committed.** `git log --oneline -5 | grep "S7 keyframe review UI"` → finds commit `304ac46`.

If any check fails, do NOT proceed. Diagnose first.

---

# Phase 1 — Offline TDD (self-driven, no GPU spend)

Strict layered TDD. Each task is one RED→GREEN→COMMIT cycle. Run quality gates (`pytest -q`, `ruff check src tests scripts`) at the end of each layer.

## Layer A: Data model (Scene fields + seed regen_count)

### Task 1: Add `Scene.review_feedback` and `Scene.regen_count` fields

**Files:**
- Modify: `src/platinum/models/story.py:140-210` (the `Scene` dataclass + `to_dict`/`from_dict`).
- Test: `tests/unit/test_story_model.py` — if this file exists, append; if not, create.

**Skills:** @superpowers:test-driven-development

#### Step 1: Check if `tests/unit/test_story_model.py` exists

Run: `ls tests/unit/test_story_model.py 2>/dev/null || echo MISSING`
- If exists → append tests at end.
- If MISSING → create the file with one round-trip test plus the new field tests.

#### Step 2: Write failing tests

Append to `tests/unit/test_story_model.py` (create if missing):

```python
"""Round-trip + new-field tests for the Scene dataclass.

S7 adds review_feedback (str | None) and regen_count (int) to Scene.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from platinum.models.story import (
    Adapted,
    Scene,
    Source,
    Story,
    ReviewStatus,
)


def _build_minimal_scene(**overrides) -> Scene:
    base = {
        "id": "scene_001",
        "index": 1,
        "narration_text": "x",
    }
    base.update(overrides)
    return Scene(**base)


def test_scene_review_feedback_defaults_to_none() -> None:
    scene = _build_minimal_scene()
    assert scene.review_feedback is None


def test_scene_regen_count_defaults_to_zero() -> None:
    scene = _build_minimal_scene()
    assert scene.regen_count == 0


def test_scene_round_trip_preserves_review_feedback() -> None:
    scene = _build_minimal_scene(review_feedback="needs more candle light")
    rt = Scene.from_dict(scene.to_dict())
    assert rt.review_feedback == "needs more candle light"


def test_scene_round_trip_preserves_regen_count() -> None:
    scene = _build_minimal_scene(regen_count=3)
    rt = Scene.from_dict(scene.to_dict())
    assert rt.regen_count == 3


def test_scene_from_dict_backfills_missing_review_feedback() -> None:
    """Old story.json files without these fields must still load."""
    raw = {"id": "s1", "index": 1, "narration_text": "x"}
    scene = Scene.from_dict(raw)
    assert scene.review_feedback is None
    assert scene.regen_count == 0
```

#### Step 3: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_story_model.py -v`
Expected: 5 tests fail with `AttributeError` (no review_feedback / regen_count) or `TypeError` (unknown init kwarg).

#### Step 4: Implement — add fields + serialization

Edit `src/platinum/models/story.py` Scene dataclass, after `validation: dict[str, Any] = field(default_factory=dict)` and before `review_status: ReviewStatus = ReviewStatus.PENDING`:

```python
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_feedback: str | None = None
    regen_count: int = 0
```

Add to `to_dict` (after `"review_status": self.review_status.value,`):

```python
            "review_feedback": self.review_feedback,
            "regen_count": self.regen_count,
```

Add to `from_dict` (in the `cls(...)` call, after `review_status=...`):

```python
            review_feedback=d.get("review_feedback"),
            regen_count=int(d.get("regen_count", 0)),
```

#### Step 5: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_story_model.py -v`
Expected: 5 passed.

#### Step 6: Run full pytest suite — sanity

Run: `python -m pytest -q`
Expected: 322 passed (317 baseline + 5 new). 0 failures.

#### Step 7: Commit

```bash
git add src/platinum/models/story.py tests/unit/test_story_model.py
git commit -m "feat(models): Scene -- add review_feedback + regen_count fields (S7 §4.2)"
```

---

### Task 2: Update `_seeds_for_scene` to accept `regen_count`; thread through `generate_for_scene` and `generate`

**Files:**
- Modify: `src/platinum/pipeline/keyframe_generator.py:56-58` (`_seeds_for_scene`), `:100-103` (`generate_for_scene` seed call), `:296-320` (`generate` per-scene loop).
- Test: `tests/unit/test_seed_regen_count.py` (new) — 3 tests for the seed function alone.

**Skills:** @superpowers:test-driven-development

#### Step 1: Write failing seed tests

Create `tests/unit/test_seed_regen_count.py`:

```python
"""Verify _seeds_for_scene is backwards-compatible at regen_count=0
and produces distinct seeds at regen_count>=1.

S7 §4.3: seed = scene_index * 1000 + regen_count * 100 + offset.
"""
from __future__ import annotations

from platinum.pipeline.keyframe_generator import _seeds_for_scene


def test_seeds_unchanged_with_default_regen_count_zero() -> None:
    """Default behavior identical to pre-S7 (no regen_count parameter)."""
    # scene_index=1, n=3, regen_count default
    assert _seeds_for_scene(1, 3) == (1000, 1001, 1002)
    assert _seeds_for_scene(7, 3) == (7000, 7001, 7002)


def test_seeds_differ_with_regen_count_one() -> None:
    """regen_count=1 shifts seeds by 100 per regen."""
    base = _seeds_for_scene(1, 3, regen_count=0)
    once = _seeds_for_scene(1, 3, regen_count=1)
    twice = _seeds_for_scene(1, 3, regen_count=2)
    assert base == (1000, 1001, 1002)
    assert once == (1100, 1101, 1102)
    assert twice == (1200, 1201, 1202)
    # All three sets disjoint
    assert set(base) & set(once) == set()
    assert set(once) & set(twice) == set()


def test_seeds_deterministic_per_regen_count() -> None:
    """Same (scene_index, n, regen_count) always produces same seeds."""
    a = _seeds_for_scene(5, 4, regen_count=2)
    b = _seeds_for_scene(5, 4, regen_count=2)
    assert a == b
    assert a == (5200, 5201, 5202, 5203)
```

#### Step 2: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_seed_regen_count.py -v`
Expected: 2 fail (TypeError on the `regen_count=` kwarg); the first test (default behavior) may pass.

#### Step 3: Implement — update `_seeds_for_scene` signature

In `src/platinum/pipeline/keyframe_generator.py`, replace:

```python
def _seeds_for_scene(scene_index: int, n: int) -> tuple[int, ...]:
    """Deterministic seeds: scene_index*1000 + offset."""
    return tuple(scene_index * 1000 + i for i in range(n))
```

with:

```python
def _seeds_for_scene(
    scene_index: int, n: int, *, regen_count: int = 0
) -> tuple[int, ...]:
    """Deterministic seeds: scene_index*1000 + regen_count*100 + offset.

    regen_count=0 reproduces pre-S7 seed sequences exactly. Capacity:
    100 regens * 100 candidates per scene before collision.
    """
    return tuple(scene_index * 1000 + regen_count * 100 + i for i in range(n))
```

#### Step 4: Thread `regen_count` through `generate_for_scene` and `generate`

In `generate_for_scene` (around line 100), replace:

```python
    use_seeds = tuple(seeds) if seeds is not None else _seeds_for_scene(scene.index, n_candidates)
```

with:

```python
    regen_count = int(getattr(scene, "regen_count", 0))
    use_seeds = (
        tuple(seeds)
        if seeds is not None
        else _seeds_for_scene(scene.index, n_candidates, regen_count=regen_count)
    )
```

(`generate()` doesn't need changes — it calls `generate_for_scene` per scene, and `scene.regen_count` is read inside there.)

#### Step 5: Run seed tests, confirm GREEN

Run: `python -m pytest tests/unit/test_seed_regen_count.py -v`
Expected: 3 passed.

#### Step 6: Run keyframe_generator tests — confirm no regression

Run: `python -m pytest tests/unit/test_keyframe_generator.py tests/integration/test_keyframe_generator_stage.py -v`
Expected: ALL existing tests still pass (the default `regen_count=0` preserves behavior). If any fail, the issue is signature drift, not correctness — investigate before continuing.

#### Step 7: Run full pytest suite

Run: `python -m pytest -q`
Expected: 325 passed (322 + 3 new).

#### Step 8: Commit

```bash
git add src/platinum/pipeline/keyframe_generator.py tests/unit/test_seed_regen_count.py
git commit -m "feat(keyframe-generator): _seeds_for_scene -- accept regen_count kwarg (S7 §4.3)"
```

---

## Layer B: Pure-core decisions

### Task 3: Create `review_ui` package with `decisions.py` skeleton

**Files:**
- Create: `src/platinum/review_ui/__init__.py` (empty marker).
- Create: `src/platinum/review_ui/decisions.py` (the pure core).
- Test: `tests/unit/test_review_decisions.py` (new).

**Skills:** @superpowers:test-driven-development

#### Step 1: Create empty package marker

```bash
mkdir -p src/platinum/review_ui
touch src/platinum/review_ui/__init__.py
```

#### Step 2: Create `decisions.py` with module docstring + imports only

Create `src/platinum/review_ui/decisions.py`:

```python
"""Pure-core review decisions.

Every function takes a Story (and possibly a scene id + action params) and
mutates it in place, returning the mutated Story for chaining. NO I/O,
NO Flask, NO SQLite. The impure shell (app.py routes) wraps these with
load + save + sync_from_story.

Mirrors the pure-core / impure-shell pattern from S3 (story_curator.apply_decision)
and S6 (keyframe_generator.generate_for_scene).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from platinum.models.story import (
    ReviewStatus,
    Scene,
    StageRun,
    StageStatus,
    Story,
)
```

#### Step 3: Write failing test for `apply_approve`

Create `tests/unit/test_review_decisions.py`:

```python
"""Unit tests for review_ui.decisions pure-core functions.

S7 §3.2 / §6.2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from platinum.models.story import (
    Adapted,
    ReviewStatus,
    Scene,
    Source,
    Story,
)
from platinum.review_ui import decisions


def _make_story(*, n_scenes: int = 3, all_have_keyframes: bool = True) -> Story:
    src = Source(
        type="gutenberg",
        url="https://example.com",
        title="Test",
        author="Test",
        raw_text="hello",
        fetched_at=datetime.now(timezone.utc),
        license="PD-US",
    )
    adapted = Adapted(
        title="Test",
        synopsis="x",
        narration_script="y",
        estimated_duration_seconds=600.0,
        tone_notes="z",
    )
    scenes = []
    for i in range(n_scenes):
        scene = Scene(
            id=f"scene_{i+1:03d}",
            index=i + 1,
            narration_text=f"scene {i}",
            visual_prompt=f"prompt {i}",
            negative_prompt="bright daylight",
        )
        if all_have_keyframes:
            scene.keyframe_candidates = [
                Path(f"scene_{i+1:03d}/candidate_{c}.png") for c in range(3)
            ]
            scene.keyframe_scores = [5.5, 6.2, 5.9]
            scene.keyframe_path = scene.keyframe_candidates[1]  # auto-selected highest
        scenes.append(scene)
    return Story(
        id="story_test", track="atmospheric_horror",
        source=src, adapted=adapted, scenes=scenes,
    )


def test_apply_approve_marks_scene_approved() -> None:
    story = _make_story()
    decisions.apply_approve(story, "scene_001")
    assert story.scenes[0].review_status == ReviewStatus.APPROVED


def test_apply_approve_idempotent() -> None:
    story = _make_story()
    decisions.apply_approve(story, "scene_001")
    decisions.apply_approve(story, "scene_001")
    assert story.scenes[0].review_status == ReviewStatus.APPROVED
    # No exceptions, status unchanged on second call


def test_apply_approve_unknown_scene_id_raises() -> None:
    story = _make_story()
    with pytest.raises(KeyError, match="scene_xyz"):
        decisions.apply_approve(story, "scene_xyz")
```

#### Step 4: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_review_decisions.py -v`
Expected: 3 fail (`AttributeError: module 'platinum.review_ui.decisions' has no attribute 'apply_approve'`).

#### Step 5: Implement `apply_approve` and a small helper

Append to `src/platinum/review_ui/decisions.py`:

```python
def _find_scene(story: Story, scene_id: str) -> Scene:
    """Return the scene with matching id, or raise KeyError."""
    for s in story.scenes:
        if s.id == scene_id:
            return s
    raise KeyError(f"scene id not found: {scene_id}")


def apply_approve(story: Story, scene_id: str) -> Story:
    """Mark scene APPROVED. Idempotent.

    Does not auto-finalize the stage run; caller is responsible for
    invoking finalize_review_if_complete afterwards.
    """
    scene = _find_scene(story, scene_id)
    scene.review_status = ReviewStatus.APPROVED
    return story
```

#### Step 6: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_decisions.py -v`
Expected: 3 passed.

#### Step 7: Commit

```bash
git add src/platinum/review_ui/__init__.py src/platinum/review_ui/decisions.py tests/unit/test_review_decisions.py
git commit -m "feat(review_ui): decisions.apply_approve -- pure-core skeleton (S7 §3.2)"
```

---

### Task 4: `apply_regenerate` — bump regen_count, clear keyframe_path, set REGENERATE

**Files:**
- Modify: `src/platinum/review_ui/decisions.py`.
- Modify: `tests/unit/test_review_decisions.py`.

#### Step 1: Append failing tests

```python
def test_apply_regenerate_clears_keyframe_path() -> None:
    story = _make_story()
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].keyframe_path is None


def test_apply_regenerate_bumps_regen_count() -> None:
    story = _make_story()
    assert story.scenes[0].regen_count == 0
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].regen_count == 1
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].regen_count == 2


def test_apply_regenerate_sets_status_REGENERATE() -> None:
    story = _make_story()
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].review_status == ReviewStatus.REGENERATE


def test_apply_regenerate_preserves_visual_prompt() -> None:
    """Same prompt, new seed -- visual_prompt MUST stay intact."""
    story = _make_story()
    original_prompt = story.scenes[0].visual_prompt
    decisions.apply_regenerate(story, "scene_001")
    assert story.scenes[0].visual_prompt == original_prompt
```

#### Step 2: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k regenerate`
Expected: 4 fail.

#### Step 3: Implement

Append to `decisions.py`:

```python
def apply_regenerate(story: Story, scene_id: str) -> Story:
    """Mark scene for re-render with same prompt + new seed.

    Bumps regen_count, clears keyframe_path (so re-render runs), preserves
    visual_prompt and review_feedback. CLI then re-runs keyframe_generator
    via `platinum keyframes <id> --rerun-regen-requested`.
    """
    scene = _find_scene(story, scene_id)
    scene.regen_count += 1
    scene.keyframe_path = None
    scene.review_status = ReviewStatus.REGENERATE
    return story
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k regenerate`
Expected: 4 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/decisions.py tests/unit/test_review_decisions.py
git commit -m "feat(review_ui): decisions.apply_regenerate -- bump regen_count + clear keyframe_path (S7 §3.2)"
```

---

### Task 5: `apply_reject` — capture feedback, clear visual_prompt + keyframe_path, set REJECTED

**Files:** Same as Task 4.

#### Step 1: Append failing tests

```python
def test_apply_reject_writes_feedback() -> None:
    story = _make_story()
    decisions.apply_reject(story, "scene_001", feedback="too dark; need amber lighting")
    assert story.scenes[0].review_feedback == "too dark; need amber lighting"


def test_apply_reject_clears_keyframe_path_and_visual_prompt() -> None:
    story = _make_story()
    decisions.apply_reject(story, "scene_001", feedback="bad")
    assert story.scenes[0].keyframe_path is None
    assert story.scenes[0].visual_prompt is None


def test_apply_reject_sets_status_REJECTED() -> None:
    story = _make_story()
    decisions.apply_reject(story, "scene_001", feedback="bad")
    assert story.scenes[0].review_status == ReviewStatus.REJECTED


def test_apply_reject_empty_feedback_raises() -> None:
    """A reject with no feedback is meaningless -- guard against accidental empty submissions."""
    story = _make_story()
    with pytest.raises(ValueError, match="feedback"):
        decisions.apply_reject(story, "scene_001", feedback="")
    with pytest.raises(ValueError, match="feedback"):
        decisions.apply_reject(story, "scene_001", feedback="   ")
```

#### Step 2: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k reject`
Expected: 4 fail.

#### Step 3: Implement

```python
def apply_reject(story: Story, scene_id: str, *, feedback: str) -> Story:
    """Reject scene with textual feedback for `--rerun-rejected` Claude regen.

    Clears visual_prompt and keyframe_path (visual_prompts will rewrite the
    prompt; keyframe_generator re-renders from the new prompt).
    """
    if not feedback or not feedback.strip():
        raise ValueError("feedback is required and must not be blank")
    scene = _find_scene(story, scene_id)
    scene.review_feedback = feedback.strip()
    scene.visual_prompt = None
    scene.keyframe_path = None
    scene.review_status = ReviewStatus.REJECTED
    return story
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k reject`
Expected: 4 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/decisions.py tests/unit/test_review_decisions.py
git commit -m "feat(review_ui): decisions.apply_reject -- capture feedback + clear prompt (S7 §3.2)"
```

---

### Task 6: `apply_swap_candidate` — change keyframe_path to a different candidate

**Files:** Same.

#### Step 1: Append failing tests

```python
def test_apply_swap_candidate_updates_keyframe_path() -> None:
    story = _make_story()
    original = story.scenes[0].keyframe_path
    decisions.apply_swap_candidate(story, "scene_001", candidate_index=0)
    assert story.scenes[0].keyframe_path == story.scenes[0].keyframe_candidates[0]
    assert story.scenes[0].keyframe_path != original


def test_apply_swap_candidate_invalid_index_raises() -> None:
    story = _make_story()  # 3 candidates per scene
    with pytest.raises(IndexError, match="candidate_index"):
        decisions.apply_swap_candidate(story, "scene_001", candidate_index=99)


def test_apply_swap_candidate_preserves_review_status() -> None:
    """Swapping should not silently approve / unapprove."""
    story = _make_story()
    story.scenes[0].review_status = ReviewStatus.PENDING
    decisions.apply_swap_candidate(story, "scene_001", candidate_index=0)
    assert story.scenes[0].review_status == ReviewStatus.PENDING
```

#### Step 2: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k swap`
Expected: 3 fail.

#### Step 3: Implement

```python
def apply_swap_candidate(story: Story, scene_id: str, *, candidate_index: int) -> Story:
    """Override the auto-selected best with a different candidate.

    Used by the 'view alternatives' UX to let a reviewer pick a candidate
    the gates ranked lower but that reads better on eye-check. Does not
    change review_status -- the user must explicitly approve afterward.
    """
    scene = _find_scene(story, scene_id)
    if candidate_index < 0 or candidate_index >= len(scene.keyframe_candidates):
        raise IndexError(
            f"candidate_index out of range: {candidate_index} "
            f"(scene has {len(scene.keyframe_candidates)} candidates)"
        )
    scene.keyframe_path = scene.keyframe_candidates[candidate_index]
    return story
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k swap`
Expected: 3 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/decisions.py tests/unit/test_review_decisions.py
git commit -m "feat(review_ui): decisions.apply_swap_candidate -- override auto-pick (S7 §3.2)"
```

---

### Task 7: `apply_batch_approve_above` — bulk approve PENDING scenes ≥ threshold

**Files:** Same.

#### Step 1: Append failing tests

```python
def test_apply_batch_approve_above_threshold_marks_pending_only() -> None:
    """Already-decided scenes (REJECTED, REGENERATE, APPROVED) must NOT be touched."""
    story = _make_story(n_scenes=4)
    # Scene 0: PENDING, score 6.2 (above)
    # Scene 1: PENDING, score 5.0 (below)
    # Scene 2: REJECTED already, score 6.5 (above but already decided)
    # Scene 3: APPROVED already, score 6.5
    story.scenes[1].keyframe_scores = [5.0, 5.0, 5.0]
    story.scenes[1].keyframe_path = story.scenes[1].keyframe_candidates[0]
    story.scenes[2].review_status = ReviewStatus.REJECTED
    story.scenes[3].review_status = ReviewStatus.APPROVED

    decisions.apply_batch_approve_above(story, threshold=6.0)

    assert story.scenes[0].review_status == ReviewStatus.APPROVED  # promoted
    assert story.scenes[1].review_status == ReviewStatus.PENDING   # below threshold
    assert story.scenes[2].review_status == ReviewStatus.REJECTED  # left alone
    assert story.scenes[3].review_status == ReviewStatus.APPROVED  # already approved


def test_apply_batch_approve_above_uses_selected_candidate_score() -> None:
    """Threshold compares against the score of the SELECTED candidate, not max(scores)."""
    story = _make_story(n_scenes=1)
    story.scenes[0].keyframe_scores = [5.5, 6.5, 5.9]
    # Manually point keyframe_path at candidate 0 (score 5.5)
    story.scenes[0].keyframe_path = story.scenes[0].keyframe_candidates[0]
    decisions.apply_batch_approve_above(story, threshold=6.0)
    # Selected has score 5.5, below threshold -> not approved
    assert story.scenes[0].review_status == ReviewStatus.PENDING


def test_apply_batch_approve_above_skips_no_keyframe_scenes() -> None:
    """A scene with keyframe_path=None has no selected score; must not approve."""
    story = _make_story(n_scenes=1, all_have_keyframes=False)
    decisions.apply_batch_approve_above(story, threshold=0.0)
    assert story.scenes[0].review_status == ReviewStatus.PENDING
```

#### Step 2: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k batch_approve`
Expected: 3 fail.

#### Step 3: Implement

```python
def apply_batch_approve_above(story: Story, *, threshold: float) -> Story:
    """Approve all PENDING scenes whose selected candidate's score >= threshold.

    Already-decided scenes (REJECTED / REGENERATE / APPROVED) are left
    untouched -- batch action is additive, never overrides prior intent.
    Scenes without a selected keyframe (keyframe_path is None) are skipped.
    """
    for scene in story.scenes:
        if scene.review_status != ReviewStatus.PENDING:
            continue
        if scene.keyframe_path is None:
            continue
        # Find selected candidate's score
        try:
            selected_idx = scene.keyframe_candidates.index(scene.keyframe_path)
        except ValueError:
            continue  # keyframe_path not in candidates (shouldn't happen, defensive)
        if selected_idx >= len(scene.keyframe_scores):
            continue
        if scene.keyframe_scores[selected_idx] >= threshold:
            scene.review_status = ReviewStatus.APPROVED
    return story
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k batch_approve`
Expected: 3 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/decisions.py tests/unit/test_review_decisions.py
git commit -m "feat(review_ui): decisions.apply_batch_approve_above -- bulk approve PENDING (S7 §3.2)"
```

---

### Task 8: `finalize_review_if_complete` — append StageRun + write review_gate

**Files:** Same.

#### Step 1: Append failing tests

```python
from platinum.models.story import StageStatus


def test_finalize_no_op_when_pending_remains() -> None:
    story = _make_story(n_scenes=3)
    decisions.apply_approve(story, "scene_001")
    decisions.apply_approve(story, "scene_002")
    # scene_003 still PENDING
    decisions.finalize_review_if_complete(story)
    assert story.latest_stage_run("keyframe_review") is None
    assert "keyframe_review" not in story.review_gates


def test_finalize_appends_stagerun_when_all_approved() -> None:
    story = _make_story(n_scenes=3)
    for s in story.scenes:
        decisions.apply_approve(story, s.id)
    decisions.finalize_review_if_complete(story)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE
    assert run.completed_at is not None


def test_finalize_writes_review_gate() -> None:
    story = _make_story(n_scenes=3)
    for s in story.scenes:
        decisions.apply_approve(story, s.id)
    decisions.finalize_review_if_complete(story)
    gate = story.review_gates.get("keyframe_review")
    assert gate is not None
    assert gate["approved_count"] == 3


def test_finalize_idempotent() -> None:
    """Running finalize twice on an already-final story does NOT append a second StageRun."""
    story = _make_story(n_scenes=2)
    for s in story.scenes:
        decisions.apply_approve(story, s.id)
    decisions.finalize_review_if_complete(story)
    decisions.finalize_review_if_complete(story)
    runs = [r for r in story.stages if r.stage == "keyframe_review"]
    assert len(runs) == 1


def test_finalize_records_regen_total_in_artifacts() -> None:
    story = _make_story(n_scenes=2)
    decisions.apply_regenerate(story, "scene_001")  # bump 0->1
    decisions.apply_regenerate(story, "scene_001")  # bump 1->2
    decisions.apply_regenerate(story, "scene_002")  # bump 0->1
    # Now approve them -- pretend GPU re-render happened in between
    for s in story.scenes:
        s.review_status = ReviewStatus.APPROVED
    decisions.finalize_review_if_complete(story)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.artifacts["regen_total"] == 3
```

#### Step 2: Run tests, confirm RED

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k finalize`
Expected: 5 fail.

#### Step 3: Implement

```python
def finalize_review_if_complete(story: Story) -> Story:
    """If every scene is APPROVED, append a COMPLETE StageRun for
    'keyframe_review' (idempotent) and write the review_gates summary.
    """
    if not story.scenes:
        return story
    if any(s.review_status != ReviewStatus.APPROVED for s in story.scenes):
        return story
    # All approved. Check we haven't already finalized.
    existing = story.latest_stage_run("keyframe_review")
    if existing is not None and existing.status == StageStatus.COMPLETE:
        return story  # idempotent

    now = datetime.now()
    regen_total = sum(s.regen_count for s in story.scenes)
    artifacts: dict[str, Any] = {
        "approved_count": len(story.scenes),
        "regen_total": regen_total,
    }
    story.stages.append(
        StageRun(
            stage="keyframe_review",
            status=StageStatus.COMPLETE,
            started_at=now,
            completed_at=now,
            artifacts=dict(artifacts),
        )
    )
    story.review_gates["keyframe_review"] = {
        "completed_at": now.isoformat(),
        "reviewer": "user",
        **artifacts,
    }
    return story
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_decisions.py -v -k finalize`
Expected: 5 passed.

#### Step 5: Run full pytest, ruff, mypy

Run: `python -m pytest -q`
Expected: ~340+ passed.

Run: `python -m ruff check src tests scripts`
Expected: `All checks passed!`

Run: `python -m mypy src`
Expected: `Found 2 errors` (the pre-existing deferrals).

#### Step 6: Commit

```bash
git add src/platinum/review_ui/decisions.py tests/unit/test_review_decisions.py
git commit -m "feat(review_ui): decisions.finalize_review_if_complete -- append stagerun on all-approved (S7 §4.4)"
```

---

## Layer C: Flask app + routes + image serving

### Task 9: Add `flask` to pyproject.toml; verify pip install resolves

**Files:**
- Modify: `pyproject.toml`.

#### Step 1: Open pyproject.toml and find `dependencies = [...]` block

Run: `python -c "import pathlib; print(pathlib.Path('pyproject.toml').read_text())"`
Note the exact location of the core dependencies array (NOT dev-extras).

#### Step 2: Add `"flask>=3.0"` to dependencies

Edit `pyproject.toml`. In the `[project] dependencies = [...]` array (where you see anthropic, fastapi, sqlalchemy, etc), add a new line: `    "flask>=3.0",`.

#### Step 3: Run pip install -e

Run: `pip install -e . 2>&1 | tail -20`
Expected: `Successfully installed flask-3.x.x ...` (or "Requirement already satisfied" if flask was a transitive dep). Confirm flask is importable: `python -c "import flask; print(flask.__version__)"` → version string.

#### Step 4: Commit

```bash
git add pyproject.toml
git commit -m "chore(deps): add flask>=3.0 for review_ui (S7 §3.7)"
```

---

### Task 10: Flask app factory skeleton with health-check route

**Files:**
- Modify: `src/platinum/review_ui/app.py` (new).
- Test: `tests/unit/test_review_app.py` (new).

#### Step 1: Write failing test

Create `tests/unit/test_review_app.py`:

```python
"""Flask app + route tests via app.test_client().

S7 §3.3 / §6.2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from platinum.models.story import (
    Adapted,
    Scene,
    Source,
    Story,
)


@pytest.fixture
def story_factory(tmp_path: Path):
    """Build a story on disk under tmp_path/data/stories/<id>/, return (story, story_dir)."""
    def _make(*, n_scenes: int = 3) -> tuple[Story, Path]:
        src = Source(
            type="gutenberg", url="https://example.com",
            title="Test", author="A", raw_text="hello",
            fetched_at=datetime.now(timezone.utc), license="PD-US",
        )
        adapted = Adapted(
            title="Test", synopsis="x", narration_script="y",
            estimated_duration_seconds=600.0, tone_notes="z",
        )
        scenes = []
        for i in range(n_scenes):
            scene = Scene(
                id=f"scene_{i+1:03d}", index=i + 1,
                narration_text=f"scene {i}",
                visual_prompt=f"prompt {i}",
                negative_prompt="bright daylight",
            )
            scenes.append(scene)
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
        )
        story_dir = tmp_path / "data" / "stories" / story.id
        story_dir.mkdir(parents=True, exist_ok=True)
        story.save(story_dir / "story.json")
        return story, story_dir
    return _make


def test_app_factory_creates_app(story_factory, tmp_path: Path) -> None:
    """create_app(story_id, data_root) returns a Flask app instance."""
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory(n_scenes=3)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    assert app is not None
    assert app.config["STORY_ID"] == story.id


def test_health_check_route_returns_200(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json == {"status": "ok"}
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/unit/test_review_app.py -v`
Expected: 2 fail (`ImportError: cannot import 'create_app' from 'platinum.review_ui.app'`).

#### Step 3: Implement minimal app factory

Create `src/platinum/review_ui/app.py`:

```python
"""Flask app for the keyframe review UI.

S7 §3 — local 127.0.0.1 only, no auth, single-user. The app factory
takes a story_id (required at boot — one story per process) plus
data_root and returns a configured Flask instance.

Routes are added in subsequent tasks; this skeleton ships only the
factory + healthcheck.
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify


def create_app(*, story_id: str, data_root: Path) -> Flask:
    """Build a Flask app bound to one story + the data_root containing
    its keyframes/ subtree.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["STORY_ID"] = story_id
    app.config["DATA_ROOT"] = Path(data_root)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    return app
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_app.py -v`
Expected: 2 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/app.py tests/unit/test_review_app.py
git commit -m "feat(review_ui): app factory + /healthz (S7 §3.3)"
```

---

### Task 11: `GET /api/story/<story_id>` JSON snapshot

**Files:** Same.

#### Step 1: Append failing test

Append to `tests/unit/test_review_app.py`:

```python
def test_get_api_story_returns_snapshot(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory(n_scenes=3)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/api/story/{story.id}")
    assert resp.status_code == 200
    data = resp.json
    assert data["id"] == story.id
    assert len(data["scenes"]) == 3
    assert data["rollup"]["pending"] == 3
    assert data["rollup"]["approved"] == 0


def test_get_api_story_404_on_missing(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id="story_doesnt_exist",
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get("/api/story/story_doesnt_exist")
    assert resp.status_code == 404
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/unit/test_review_app.py::test_get_api_story_returns_snapshot tests/unit/test_review_app.py::test_get_api_story_404_on_missing -v`
Expected: 2 fail (404 from unmapped route).

#### Step 3: Implement — add helper + route

Edit `src/platinum/review_ui/app.py`. Add imports and helpers below the existing imports:

```python
from flask import Flask, abort, jsonify

from platinum.models.story import ReviewStatus, Story


def _story_path(data_root: Path, story_id: str) -> Path:
    return data_root / story_id / "story.json"


def _load_story_or_404(data_root: Path, story_id: str) -> Story:
    path = _story_path(data_root, story_id)
    if not path.exists():
        abort(404, description=f"story not found: {story_id}")
    return Story.load(path)


def _rollup(story: Story) -> dict[str, int]:
    counts = {"pending": 0, "approved": 0, "rejected": 0, "regen_requested": 0}
    for scene in story.scenes:
        if scene.review_status == ReviewStatus.PENDING:
            counts["pending"] += 1
        elif scene.review_status == ReviewStatus.APPROVED:
            counts["approved"] += 1
        elif scene.review_status == ReviewStatus.REJECTED:
            counts["rejected"] += 1
        elif scene.review_status == ReviewStatus.REGENERATE:
            counts["regen_requested"] += 1
    return counts
```

Add the route inside `create_app`:

```python
    @app.get("/api/story/<story_id>")
    def api_story(story_id: str):
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        body = story.to_dict()
        body["rollup"] = _rollup(story)
        return jsonify(body)
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_app.py -v`
Expected: 4 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/app.py tests/unit/test_review_app.py
git commit -m "feat(review_ui): GET /api/story/<id> -- JSON snapshot + rollup (S7 §3.3)"
```

---

### Task 12: `GET /image/<story_id>/<path:relpath>` with `safe_join` traversal protection

**Files:** Same.

#### Step 1: Append failing tests

```python
def test_get_image_serves_png(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, story_dir = story_factory()
    keyframes_dir = story_dir / "keyframes" / "scene_001"
    keyframes_dir.mkdir(parents=True)
    png_path = keyframes_dir / "candidate_0.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/image/{story.id}/scene_001/candidate_0.png")
    assert resp.status_code == 200
    assert resp.data == b"\x89PNG\r\n\x1a\nfake"


def test_get_image_404_on_missing(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/image/{story.id}/nonexistent/x.png")
    assert resp.status_code == 404


def test_get_image_blocks_path_traversal(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    secret = tmp_path / "secret.txt"
    secret.write_text("hunter2")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/image/{story.id}/../../../secret.txt")
    # safe_join returns None for traversal -> 404
    assert resp.status_code == 404
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/unit/test_review_app.py -v -k get_image`
Expected: 3 fail.

#### Step 3: Implement

Add to `app.py` imports:

```python
from werkzeug.security import safe_join
from flask import send_file
```

Add the route inside `create_app`:

```python
    @app.get("/image/<story_id>/<path:relpath>")
    def image(story_id: str, relpath: str):
        keyframes_root = app.config["DATA_ROOT"] / story_id / "keyframes"
        full = safe_join(str(keyframes_root), relpath)
        if full is None:
            abort(404)
        full_path = Path(full)
        if not full_path.exists() or not full_path.is_file():
            abort(404)
        return send_file(full_path)
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/unit/test_review_app.py -v -k get_image`
Expected: 3 passed.

#### Step 5: Commit

```bash
git add src/platinum/review_ui/app.py tests/unit/test_review_app.py
git commit -m "feat(review_ui): GET /image -- safe_join path-traversal protection (S7 §3.5)"
```

---

### Task 13: POST routes for approve / regenerate / reject / select_candidate / batch_approve

**Files:** Same.

This bundles 5 small POST routes into one task because each is the same load → mutate → finalize → save → return JSON pattern.

#### Step 1: Append failing tests

```python
def test_post_approve_persists_to_disk(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app
    from platinum.models.story import ReviewStatus, Story

    story, story_dir = story_factory(n_scenes=3)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(f"/api/story/{story.id}/scene/scene_001/approve")
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].review_status == ReviewStatus.APPROVED
    assert resp.json["rollup"]["approved"] == 1


def test_post_regenerate_bumps_count_and_clears_keyframe(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app
    from platinum.models.story import Story

    story, story_dir = story_factory()
    # Give scene_001 a keyframe_path to clear
    story.scenes[0].keyframe_path = Path("scene_001/candidate_0.png")
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(f"/api/story/{story.id}/scene/scene_001/regenerate")
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].keyframe_path is None
    assert rt.scenes[0].regen_count == 1


def test_post_reject_requires_feedback(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/scene/scene_001/reject",
            json={},  # no feedback field
        )
    assert resp.status_code == 400


def test_post_reject_persists_feedback(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app
    from platinum.models.story import ReviewStatus, Story

    story, story_dir = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/scene/scene_001/reject",
            json={"feedback": "scene 1 face needs amber"},
        )
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].review_feedback == "scene 1 face needs amber"
    assert rt.scenes[0].review_status == ReviewStatus.REJECTED


def test_post_select_candidate_swaps_keyframe_path(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app
    from platinum.models.story import Story

    story, story_dir = story_factory()
    # Give scene 0 candidate paths
    story.scenes[0].keyframe_candidates = [
        Path("scene_001/candidate_0.png"),
        Path("scene_001/candidate_1.png"),
        Path("scene_001/candidate_2.png"),
    ]
    story.scenes[0].keyframe_scores = [5.5, 6.2, 5.9]
    story.scenes[0].keyframe_path = story.scenes[0].keyframe_candidates[1]
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/scene/scene_001/select_candidate",
            json={"index": 0},
        )
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].keyframe_path == Path("scene_001/candidate_0.png")


def test_post_batch_approve_marks_pending_above_threshold(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app
    from platinum.models.story import ReviewStatus, Story

    story, story_dir = story_factory(n_scenes=2)
    # scene 0 selected score = 6.2 (above 6.0)
    story.scenes[0].keyframe_candidates = [Path("a.png"), Path("b.png")]
    story.scenes[0].keyframe_scores = [5.0, 6.2]
    story.scenes[0].keyframe_path = story.scenes[0].keyframe_candidates[1]
    # scene 1 selected score = 5.5 (below 6.0)
    story.scenes[1].keyframe_candidates = [Path("c.png")]
    story.scenes[1].keyframe_scores = [5.5]
    story.scenes[1].keyframe_path = story.scenes[1].keyframe_candidates[0]
    story.save(story_dir / "story.json")

    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.post(
            f"/api/story/{story.id}/batch_approve",
            json={"threshold": 6.0},
        )
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    assert rt.scenes[0].review_status == ReviewStatus.APPROVED
    assert rt.scenes[1].review_status == ReviewStatus.PENDING


def test_post_finalizes_when_all_approved(story_factory, tmp_path: Path) -> None:
    """The last approve should append a keyframe_review StageRun."""
    from platinum.review_ui.app import create_app
    from platinum.models.story import StageStatus, Story

    story, story_dir = story_factory(n_scenes=2)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        client.post(f"/api/story/{story.id}/scene/scene_001/approve")
        resp = client.post(f"/api/story/{story.id}/scene/scene_002/approve")
    assert resp.status_code == 200
    rt = Story.load(story_dir / "story.json")
    run = rt.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/unit/test_review_app.py -v -k "post_"`
Expected: 7 fail.

#### Step 3: Implement — add the 5 routes + a small mutation helper

Edit `app.py`. Add import:

```python
from platinum.review_ui import decisions
```

Inside `create_app`, add a helper closure + the routes:

```python
    def _save_and_respond(story: Story, *, scene_id: str | None = None):
        """Common tail: finalize → save → return JSON of the touched scene + rollup."""
        decisions.finalize_review_if_complete(story)
        story.save(_story_path(app.config["DATA_ROOT"], story.id))
        body: dict = {"rollup": _rollup(story)}
        if scene_id is not None:
            for sc in story.scenes:
                if sc.id == scene_id:
                    body["scene"] = sc.to_dict()
                    break
        return jsonify(body)

    @app.post("/api/story/<story_id>/scene/<scene_id>/approve")
    def post_approve(story_id: str, scene_id: str):
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_approve(story, scene_id)
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/scene/<scene_id>/regenerate")
    def post_regenerate(story_id: str, scene_id: str):
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_regenerate(story, scene_id)
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/scene/<scene_id>/reject")
    def post_reject(story_id: str, scene_id: str):
        from flask import request
        body = request.get_json(silent=True) or {}
        feedback = body.get("feedback", "")
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_reject(story, scene_id, feedback=feedback)
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        except ValueError as exc:
            abort(400, description=str(exc))
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/scene/<scene_id>/select_candidate")
    def post_select_candidate(story_id: str, scene_id: str):
        from flask import request
        body = request.get_json(silent=True) or {}
        if "index" not in body:
            abort(400, description="'index' field required")
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        try:
            decisions.apply_swap_candidate(story, scene_id, candidate_index=int(body["index"]))
        except KeyError:
            abort(404, description=f"scene not found: {scene_id}")
        except IndexError as exc:
            abort(400, description=str(exc))
        return _save_and_respond(story, scene_id=scene_id)

    @app.post("/api/story/<story_id>/batch_approve")
    def post_batch_approve(story_id: str):
        from flask import request
        body = request.get_json(silent=True) or {}
        if "threshold" not in body:
            abort(400, description="'threshold' field required")
        story = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        decisions.apply_batch_approve_above(story, threshold=float(body["threshold"]))
        return _save_and_respond(story)
```

#### Step 4: Run all review_app tests

Run: `python -m pytest tests/unit/test_review_app.py -v`
Expected: 11 passed (4 from earlier + 7 new).

#### Step 5: Run full pytest, ruff

Run: `python -m pytest -q`
Expected: ~352 passed.

Run: `python -m ruff check src tests scripts`
Expected: clean.

#### Step 6: Commit

```bash
git add src/platinum/review_ui/app.py tests/unit/test_review_app.py
git commit -m "feat(review_ui): POST routes -- approve/regenerate/reject/swap/batch (S7 §3.3)"
```

---

## Layer D: Templates + frontend

### Task 14: Render `keyframe_gallery.html` from `GET /story/<story_id>`

**Files:**
- Modify: `src/platinum/review_ui/app.py` (add the GET /story route).
- Create: `src/platinum/review_ui/templates/base.html`.
- Create: `src/platinum/review_ui/templates/keyframe_gallery.html`.
- Create: `src/platinum/review_ui/static/style.css`.
- Modify: `tests/unit/test_review_app.py`.

This is the largest file-dump task — UI assets are mostly static content with one route.

#### Step 1: Append failing test

```python
def test_get_story_renders_template(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory(n_scenes=2)
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get(f"/story/{story.id}")
    assert resp.status_code == 200
    assert b"Keyframe Review" in resp.data
    assert story.id.encode() in resp.data
    # Each scene appears
    assert b"scene_001" in resp.data
    assert b"scene_002" in resp.data


def test_get_root_redirects_to_story(story_factory, tmp_path: Path) -> None:
    from platinum.review_ui.app import create_app

    story, _ = story_factory()
    app = create_app(
        story_id=story.id,
        data_root=tmp_path / "data" / "stories",
    )
    with app.test_client() as client:
        resp = client.get("/")
    assert resp.status_code == 302
    assert f"/story/{story.id}" in resp.location
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/unit/test_review_app.py -v -k "get_story_renders or get_root_redirects"`
Expected: 2 fail.

#### Step 3: Create `base.html`

Create `src/platinum/review_ui/templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{% block title %}Keyframe Review{% endblock %}</title>
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
<header>
  <h1>Keyframe Review</h1>
  <div id="rollup-bar">
    <span class="pill pending"   id="rollup-pending">Pending: {{ rollup.pending }}</span>
    <span class="pill approved"  id="rollup-approved">Approved: {{ rollup.approved }}</span>
    <span class="pill rejected"  id="rollup-rejected">Rejected: {{ rollup.rejected }}</span>
    <span class="pill regen"     id="rollup-regen">Regen: {{ rollup.regen_requested }}</span>
  </div>
  <div id="batch-bar">
    <label>Approve all PENDING with score &geq;
      <input type="number" id="batch-threshold" value="{{ default_threshold }}" step="0.1" min="0" max="10">
    </label>
    <button id="batch-approve-btn">Batch approve</button>
  </div>
</header>
<main id="story-grid" data-story-id="{{ story.id }}">
  {% block content %}{% endblock %}
</main>
<dialog id="reject-dialog">
  <form method="dialog">
    <h2>Reject scene <span id="reject-scene-id"></span></h2>
    <p>What went wrong with this keyframe? Feedback is passed to Claude on
       <code>platinum adapt --rerun-rejected</code>.</p>
    <textarea id="reject-feedback" rows="4" cols="60"
              placeholder="e.g. face is in shadow; needs amber side lighting"></textarea>
    <menu>
      <button value="cancel">Cancel</button>
      <button id="reject-submit-btn" value="confirm">Reject</button>
    </menu>
  </form>
</dialog>
<script>
{% include 'review_ui.js' %}
</script>
</body>
</html>
```

#### Step 4: Create `keyframe_gallery.html`

Create `src/platinum/review_ui/templates/keyframe_gallery.html`:

```html
{% extends "base.html" %}
{% block content %}
{% for scene in story.scenes %}
<article class="scene-tile" data-scene-id="{{ scene.id }}"
         data-status="{{ scene.review_status.value }}">
  <header class="scene-header">
    <span class="scene-index">#{{ scene.index }}</span>
    <span class="status-pill {{ scene.review_status.value }}">
      {{ scene.review_status.value }}
    </span>
    {% if scene.regen_count %}
    <span class="regen-badge" title="regen attempts">↻{{ scene.regen_count }}</span>
    {% endif %}
  </header>
  {% if scene.keyframe_path %}
  <img class="thumb" alt="keyframe scene {{ scene.index }}"
       src="{{ url_for('image', story_id=story.id, relpath=scene_relpath(scene)) }}">
  {% else %}
  <div class="thumb missing">no keyframe — regenerate or run keyframes</div>
  {% endif %}
  <div class="prompt-blurb" title="{{ scene.visual_prompt or '' }}">
    {{ (scene.visual_prompt or '(prompt cleared)') | truncate(140) }}
  </div>
  <div class="score-row">
    {% set selected_score = selected_score_for(scene) %}
    <span class="score">LAION
      {% if selected_score is not none %}
        {{ "%.2f"|format(selected_score) }}
      {% else %}—{% endif %}
    </span>
    {% if scene.keyframe_candidates|length > 1 %}
    <details class="alternatives">
      <summary>View alternatives</summary>
      <div class="alt-grid">
        {% for c in scene.keyframe_candidates %}
        {% set ci = loop.index0 %}
        <button class="alt-thumb"
                data-scene="{{ scene.id }}"
                data-candidate="{{ ci }}"
                {% if scene.keyframe_path == c %}disabled{% endif %}>
          <img alt="candidate {{ ci }}"
               src="{{ url_for('image', story_id=story.id, relpath=candidate_relpath(scene, ci)) }}">
          <span>
            {% if ci < scene.keyframe_scores|length %}
              {{ "%.2f"|format(scene.keyframe_scores[ci]) }}
            {% else %}—{% endif %}
          </span>
        </button>
        {% endfor %}
      </div>
    </details>
    {% endif %}
  </div>
  <div class="actions">
    <button class="btn approve"    data-scene="{{ scene.id }}">Approve</button>
    <button class="btn regenerate" data-scene="{{ scene.id }}">Regenerate</button>
    <button class="btn reject"     data-scene="{{ scene.id }}">Reject</button>
  </div>
</article>
{% endfor %}
{% endblock %}
```

#### Step 5: Create `style.css`

Create `src/platinum/review_ui/static/style.css`:

```css
:root {
  --bg: #1a1a1a;
  --fg: #eaeaea;
  --muted: #888;
  --accent: #d4a05c;
  --pending: #888;
  --approved: #4a8;
  --rejected: #c66;
  --regenerate: #ca6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--fg);
}
header {
  position: sticky; top: 0; z-index: 10;
  background: #222; padding: 1rem 2rem;
  border-bottom: 1px solid #333;
  display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
}
h1 { font-size: 1.4rem; margin: 0; }
.pill {
  display: inline-block; padding: 0.2rem 0.6rem; border-radius: 1rem;
  font-size: 0.85rem; margin-right: 0.5rem;
}
.pill.pending { background: var(--pending); color: black; }
.pill.approved { background: var(--approved); color: black; }
.pill.rejected { background: var(--rejected); color: black; }
.pill.regen { background: var(--regenerate); color: black; }

#batch-bar input[type=number] { width: 4rem; }
#batch-bar button { padding: 0.4rem 0.8rem; }

main#story-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 1rem; padding: 1rem 2rem;
}

.scene-tile {
  background: #222; border: 2px solid transparent;
  border-radius: 8px; padding: 1rem; display: flex; flex-direction: column;
  gap: 0.75rem; transition: border-color 0.2s;
}
.scene-tile[data-status="approved"] { border-color: var(--approved); }
.scene-tile[data-status="rejected"] { border-color: var(--rejected); }
.scene-tile[data-status="regenerate"] { border-color: var(--regenerate); }

.scene-header {
  display: flex; align-items: center; justify-content: space-between;
}
.scene-index { font-weight: bold; color: var(--accent); }
.status-pill {
  font-size: 0.8rem; padding: 0.15rem 0.5rem; border-radius: 0.8rem;
  background: var(--pending); color: black;
}
.status-pill.approved { background: var(--approved); }
.status-pill.rejected { background: var(--rejected); }
.status-pill.regenerate { background: var(--regenerate); }
.regen-badge { color: var(--regenerate); font-family: monospace; }

.thumb {
  width: 100%; aspect-ratio: 1 / 1; object-fit: cover;
  border-radius: 4px; background: #111;
}
.thumb.missing {
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-style: italic; padding: 2rem;
}

.prompt-blurb {
  font-size: 0.85rem; color: var(--muted); line-height: 1.3;
  max-height: 3em; overflow: hidden; text-overflow: ellipsis;
}

.score-row {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 0.9rem;
}
.score { color: var(--accent); font-family: monospace; }

details.alternatives summary {
  cursor: pointer; color: var(--muted); font-size: 0.85rem;
}
.alt-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.4rem;
  margin-top: 0.5rem;
}
.alt-thumb {
  background: none; border: 1px solid #444; border-radius: 4px;
  padding: 0.3rem; cursor: pointer; display: flex; flex-direction: column;
  align-items: center; gap: 0.2rem; color: var(--fg);
}
.alt-thumb:disabled { opacity: 0.4; cursor: default; border-color: var(--accent); }
.alt-thumb img { width: 100%; aspect-ratio: 1/1; object-fit: cover; }
.alt-thumb span { font-size: 0.75rem; color: var(--accent); font-family: monospace; }

.actions {
  display: flex; gap: 0.4rem; margin-top: auto;
}
.btn {
  flex: 1; padding: 0.5rem; border: 0; border-radius: 4px;
  font-weight: 600; cursor: pointer;
}
.btn.approve    { background: var(--approved); color: black; }
.btn.regenerate { background: var(--regenerate); color: black; }
.btn.reject     { background: var(--rejected); color: black; }

dialog { background: #2a2a2a; color: var(--fg); border: 1px solid #444;
         border-radius: 6px; padding: 1.5rem; }
dialog textarea { width: 100%; background: #1a1a1a; color: var(--fg);
                  border: 1px solid #444; padding: 0.5rem; }
dialog menu { display: flex; gap: 0.5rem; padding: 0; margin-top: 1rem;
              justify-content: flex-end; }
```

#### Step 6: Add the route + helpers to `app.py`

In `app.py`, add imports:

```python
from flask import render_template, redirect, url_for
```

Inside `create_app`, after the `_save_and_respond` helper, define:

```python
    def _scene_relpath(scene) -> str:
        """The keyframe_path is stored absolute or relative; return a relpath
        usable in url_for('image', relpath=...). Falls back to filename."""
        if scene.keyframe_path is None:
            return ""
        return f"scene_{scene.index:03d}/{Path(scene.keyframe_path).name}"

    def _candidate_relpath(scene, idx: int) -> str:
        return f"scene_{scene.index:03d}/candidate_{idx}.png"

    def _selected_score(scene) -> float | None:
        if scene.keyframe_path is None:
            return None
        try:
            i = scene.keyframe_candidates.index(scene.keyframe_path)
        except ValueError:
            return None
        if i >= len(scene.keyframe_scores):
            return None
        return scene.keyframe_scores[i]
```

Add the routes inside `create_app`:

```python
    @app.get("/")
    def index():
        return redirect(url_for("story", story_id=app.config["STORY_ID"]))

    @app.get("/story/<story_id>")
    def story(story_id: str):
        story_obj = _load_story_or_404(app.config["DATA_ROOT"], story_id)
        return render_template(
            "keyframe_gallery.html",
            story=story_obj,
            rollup=_rollup(story_obj),
            default_threshold=app.config.get("DEFAULT_THRESHOLD", 6.0),
            scene_relpath=_scene_relpath,
            candidate_relpath=_candidate_relpath,
            selected_score_for=_selected_score,
        )
```

#### Step 7: Run the new tests

Run: `python -m pytest tests/unit/test_review_app.py -v -k "renders or redirects"`
Expected: 2 passed (after adding a stub `review_ui.js` template — see next step).

#### Step 8: Add stub `review_ui.js` template

Create `src/platinum/review_ui/templates/review_ui.js`:

```js
// Stub -- task 15 fills in the full fetch + DOM update logic.
```

(Empty content acceptable for now — Task 15 fills it in.)

Re-run tests:
Run: `python -m pytest tests/unit/test_review_app.py -v`
Expected: 13 passed (11 from before + 2 new).

#### Step 9: Commit

```bash
git add src/platinum/review_ui/templates/ src/platinum/review_ui/static/style.css src/platinum/review_ui/app.py tests/unit/test_review_app.py
git commit -m "feat(review_ui): GET /story render -- gallery template + style.css (S7 §3.4)"
```

---

### Task 15: Inline JS for fetch + DOM updates

**Files:**
- Modify: `src/platinum/review_ui/templates/review_ui.js`.

This is a static frontend file — no Python tests for it; manual smoke covers the JS in Layer F.

#### Step 1: Replace the stub `review_ui.js` with the full client logic

Replace `src/platinum/review_ui/templates/review_ui.js` with:

```js
const STORY_ID = document.querySelector('#story-grid').dataset.storyId;

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : null,
  });
  if (!resp.ok) {
    const detail = await resp.text();
    alert(`Error ${resp.status}: ${detail}`);
    return null;
  }
  return await resp.json();
}

function updateRollup(rollup) {
  document.querySelector('#rollup-pending').textContent  = `Pending: ${rollup.pending}`;
  document.querySelector('#rollup-approved').textContent = `Approved: ${rollup.approved}`;
  document.querySelector('#rollup-rejected').textContent = `Rejected: ${rollup.rejected}`;
  document.querySelector('#rollup-regen').textContent    = `Regen: ${rollup.regen_requested}`;
}

function updateSceneTile(sceneId, sceneJson) {
  const tile = document.querySelector(`.scene-tile[data-scene-id="${sceneId}"]`);
  if (!tile) return;
  tile.dataset.status = sceneJson.review_status;
  const pill = tile.querySelector('.status-pill');
  pill.className = `status-pill ${sceneJson.review_status}`;
  pill.textContent = sceneJson.review_status;
  // Refresh regen badge
  const header = tile.querySelector('.scene-header');
  let badge = header.querySelector('.regen-badge');
  if (sceneJson.regen_count > 0) {
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'regen-badge';
      badge.title = 'regen attempts';
      header.appendChild(badge);
    }
    badge.textContent = `↻${sceneJson.regen_count}`;
  } else if (badge) {
    badge.remove();
  }
}

document.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button');
  if (!btn) return;
  const sceneId = btn.dataset.scene;

  if (btn.classList.contains('approve')) {
    const data = await postJSON(`/api/story/${STORY_ID}/scene/${sceneId}/approve`);
    if (data) { updateRollup(data.rollup); updateSceneTile(sceneId, data.scene); }
  }
  else if (btn.classList.contains('regenerate')) {
    const data = await postJSON(`/api/story/${STORY_ID}/scene/${sceneId}/regenerate`);
    if (data) { updateRollup(data.rollup); updateSceneTile(sceneId, data.scene); }
  }
  else if (btn.classList.contains('reject')) {
    const dlg = document.querySelector('#reject-dialog');
    document.querySelector('#reject-scene-id').textContent = sceneId;
    document.querySelector('#reject-feedback').value = '';
    dlg.dataset.sceneId = sceneId;
    dlg.showModal();
  }
  else if (btn.classList.contains('alt-thumb')) {
    const candidateIdx = parseInt(btn.dataset.candidate, 10);
    const data = await postJSON(
      `/api/story/${STORY_ID}/scene/${btn.dataset.scene}/select_candidate`,
      {index: candidateIdx},
    );
    if (data) {
      // Reload to refresh the thumbnail src + the disabled state on alt-thumbs.
      window.location.reload();
    }
  }
  else if (btn.id === 'batch-approve-btn') {
    const threshold = parseFloat(document.querySelector('#batch-threshold').value);
    if (!isFinite(threshold)) { alert('threshold must be a number'); return; }
    const data = await postJSON(`/api/story/${STORY_ID}/batch_approve`, {threshold});
    if (data) {
      // Many scenes may have changed; reload to re-render all tiles.
      window.location.reload();
    }
  }
  else if (btn.id === 'reject-submit-btn') {
    ev.preventDefault();
    const dlg = document.querySelector('#reject-dialog');
    const sceneId = dlg.dataset.sceneId;
    const feedback = document.querySelector('#reject-feedback').value.trim();
    if (!feedback) { alert('feedback required'); return; }
    const data = await postJSON(
      `/api/story/${STORY_ID}/scene/${sceneId}/reject`,
      {feedback},
    );
    dlg.close();
    if (data) { updateRollup(data.rollup); updateSceneTile(sceneId, data.scene); }
  }
});
```

#### Step 2: Run all review_app tests — confirm no regression

Run: `python -m pytest tests/unit/test_review_app.py -v`
Expected: 13 passed.

#### Step 3: Commit

```bash
git add src/platinum/review_ui/templates/review_ui.js
git commit -m "feat(review_ui): inline JS -- fetch + DOM updates for review actions (S7 §3.4)"
```

---

## Layer E: CLI integration

### Task 16: Replace stub `review` command with Typer sub-app + `keyframes` subcommand

**Files:**
- Modify: `src/platinum/cli.py:358-367` (the existing `review` stub).
- Test: `tests/integration/test_review_command.py` (new).

#### Step 1: Write failing test

Create `tests/integration/test_review_command.py`:

```python
"""Integration tests for `platinum review keyframes`.

S7 §5.1 / §6.2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from platinum.cli import app
from platinum.models.story import Adapted, Scene, Source, Story


@pytest.fixture
def cask_story_factory(tmp_path: Path, monkeypatch):
    """Build a story on disk under tmp_path/data/stories/, return its id."""
    monkeypatch.chdir(tmp_path)
    def _make() -> str:
        src = Source(
            type="gutenberg", url="https://example.com",
            title="Cask", author="Poe", raw_text="hello",
            fetched_at=datetime.now(timezone.utc), license="PD-US",
        )
        adapted = Adapted(
            title="Cask", synopsis="x", narration_script="y",
            estimated_duration_seconds=600.0, tone_notes="z",
        )
        scenes = [
            Scene(id=f"scene_{i+1:03d}", index=i+1, narration_text=f"s{i}",
                  visual_prompt=f"p{i}", negative_prompt="bright daylight")
            for i in range(2)
        ]
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
        )
        d = tmp_path / "data" / "stories" / story.id
        d.mkdir(parents=True, exist_ok=True)
        story.save(d / "story.json")
        return story.id
    return _make


def test_review_keyframes_missing_story_exit_1(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "stories").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(app, ["review", "keyframes", "story_does_not_exist"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_review_keyframes_no_browser_skips_open(cask_story_factory) -> None:
    story_id = cask_story_factory()
    runner = CliRunner()
    with patch("webbrowser.open") as mock_open, \
         patch("flask.Flask.run") as mock_run:  # do not actually run the server
        result = runner.invoke(
            app, ["review", "keyframes", story_id, "--no-browser"],
        )
    assert result.exit_code == 0, result.output
    mock_open.assert_not_called()
    mock_run.assert_called_once()
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/integration/test_review_command.py -v`
Expected: 2 fail (current `review` stub returns "not implemented yet" exit 1 even with valid args; `webbrowser.open` and `Flask.run` are not invoked).

#### Step 3: Replace the `review` stub with a Typer sub-app

In `src/platinum/cli.py`, find the existing `review` command (around line 358-367) and replace it with:

```python
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
    actions. See docs/plans/2026-04-27-session-7-... for the full design.
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
```

(Make sure to delete the existing stub function body for `review` to avoid duplicate command registration.)

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/integration/test_review_command.py -v`
Expected: 2 passed.

#### Step 5: Run full pytest

Run: `python -m pytest -q`
Expected: ~354 passed.

#### Step 6: Commit

```bash
git add src/platinum/cli.py tests/integration/test_review_command.py
git commit -m "feat(cli): platinum review keyframes -- Typer sub-app + Flask launch (S7 §5.1)"
```

---

### Task 17: `platinum keyframes <id> --rerun-regen-requested` flag

**Files:**
- Modify: `src/platinum/cli.py:262-347` (the existing `keyframes` command).
- Test: `tests/integration/test_keyframe_rerun_regen_requested.py` (new).

#### Step 1: Write failing tests

Create `tests/integration/test_keyframe_rerun_regen_requested.py`:

```python
"""Integration tests for `platinum keyframes --rerun-regen-requested`.

S7 §5.2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from platinum.cli import app
from platinum.models.story import (
    Adapted, ReviewStatus, Scene, Source, StageRun, StageStatus, Story,
)


@pytest.fixture
def cask_story_factory(tmp_path: Path, monkeypatch):
    """Build a story with visual_prompts COMPLETE, varied review statuses."""
    monkeypatch.chdir(tmp_path)
    def _make() -> str:
        src = Source(
            type="gutenberg", url="https://example.com",
            title="Cask", author="Poe", raw_text="hello",
            fetched_at=datetime.now(timezone.utc), license="PD-US",
        )
        adapted = Adapted(
            title="Cask", synopsis="x", narration_script="y",
            estimated_duration_seconds=600.0, tone_notes="z",
        )
        scenes = [
            Scene(id=f"scene_{i+1:03d}", index=i+1, narration_text=f"s{i}",
                  visual_prompt=f"p{i}", negative_prompt="bright daylight",
                  keyframe_path=Path(f"scene_{i+1:03d}/candidate_0.png")
                  if i != 1 else None,  # scene 2 will be REGENERATE (cleared)
                  review_status=(
                      ReviewStatus.APPROVED if i == 0
                      else ReviewStatus.REGENERATE if i == 1
                      else ReviewStatus.PENDING
                  ),
                  regen_count=1 if i == 1 else 0)
            for i in range(3)
        ]
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
            stages=[
                StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(timezone.utc)),
            ],
        )
        d = tmp_path / "data" / "stories" / story.id
        d.mkdir(parents=True, exist_ok=True)
        story.save(d / "story.json")
        return story.id
    return _make


def test_rerun_regen_requested_filters_to_REGENERATE_status(
    cask_story_factory, monkeypatch,
) -> None:
    """--rerun-regen-requested builds scene_filter from REGENERATE-status only."""
    story_id = cask_story_factory()
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["keyframes", story_id, "--rerun-regen-requested", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    # Dry-run prints the planned scene set; should be exactly [2] (the REGENERATE scene)
    assert "Would generate keyframes for scenes [2]" in result.output


def test_rerun_regen_requested_empty_set_exits_zero(
    cask_story_factory, monkeypatch, tmp_path: Path,
) -> None:
    """If no scenes are flagged REGENERATE, exit 0 with helpful message."""
    monkeypatch.chdir(tmp_path)
    src = Source(type="gutenberg", url="https://example.com", title="t",
                 author="a", raw_text="x",
                 fetched_at=datetime.now(timezone.utc), license="PD-US")
    adapted = Adapted(title="t", synopsis="x", narration_script="y",
                      estimated_duration_seconds=600.0, tone_notes="z")
    story = Story(
        id="story_x", track="atmospheric_horror",
        source=src, adapted=adapted,
        scenes=[
            Scene(id="scene_001", index=1, narration_text="x",
                  visual_prompt="p", negative_prompt="bright daylight",
                  review_status=ReviewStatus.APPROVED),
        ],
        stages=[
            StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                     completed_at=datetime.now(timezone.utc)),
        ],
    )
    d = tmp_path / "data" / "stories" / story.id
    d.mkdir(parents=True, exist_ok=True)
    story.save(d / "story.json")

    runner = CliRunner()
    result = runner.invoke(
        app, ["keyframes", story.id, "--rerun-regen-requested"],
    )
    assert result.exit_code == 0
    assert "no scenes flagged" in result.output.lower()
```

#### Step 2: Run, confirm RED

Run: `python -m pytest tests/integration/test_keyframe_rerun_regen_requested.py -v`
Expected: 2 fail (no `--rerun-regen-requested` flag).

#### Step 3: Implement

In `cli.py`, modify the `keyframes` command. Add a new option after `dry_run`:

```python
    rerun_regen_requested: bool = typer.Option(
        False, "--rerun-regen-requested",
        help="Auto-build --scenes filter from REGENERATE-status scenes (S7 review loop).",
    ),
```

After `cfg = Config()` and the story load, add the filter-building logic:

```python
    if rerun_regen_requested:
        if scenes is not None:
            raise typer.BadParameter(
                "--rerun-regen-requested is mutually exclusive with --scenes",
                param_hint="--rerun-regen-requested",
            )
        regen_indices = sorted({
            s.index for s in s.scenes
            if s.review_status == ReviewStatus.REGENERATE
        })
        if not regen_indices:
            console.print(
                "[yellow]No scenes flagged for regeneration. Run "
                "'platinum review keyframes <id>' first.[/yellow]"
            )
            raise typer.Exit(code=0)
        scene_filter = set(regen_indices)
        # Skip the --scenes parsing block below by short-circuiting via flag check
```

You'll need to import `ReviewStatus`:

```python
    from platinum.models.story import ReviewStatus, StageStatus, Story
```

(replace existing import on the line just before the function body).

Restructure the `--scenes` parsing so it's skipped when `rerun_regen_requested` is True. Easiest: add `if scene_filter is None and scenes is not None:` guard around the existing `--scenes` parsing block.

After successful generation, set `review_status = PENDING` for the regenerated scenes:

After `asyncio.run(orchestrator.run(s, ctx))` and console.print success:

```python
    if rerun_regen_requested:
        for scene in s.scenes:
            if scene.index in scene_filter and scene.keyframe_path is not None:
                scene.review_status = ReviewStatus.PENDING
        s.save(story_path)
```

#### Step 4: Run tests, confirm GREEN

Run: `python -m pytest tests/integration/test_keyframe_rerun_regen_requested.py -v`
Expected: 2 passed.

#### Step 5: Add 2 more tests for actually rerun behavior — use `--dry-run` to avoid GPU

The two tests above use `--dry-run` (test 1) or empty-set (test 2) to avoid actually invoking the orchestrator. The "actually re-renders + sets status" path needs orchestrator mocking. Defer the deeper test to `test_review_full_loop.py` in Task 19 (which uses the FakeComfyClient).

#### Step 6: Commit

```bash
git add src/platinum/cli.py tests/integration/test_keyframe_rerun_regen_requested.py
git commit -m "feat(cli): platinum keyframes -- --rerun-regen-requested flag (S7 §5.2)"
```

---

### Task 18: `platinum adapt --rerun-rejected` flag + visual_prompts deviation_feedback

**Files:**
- Modify: `config/prompts/atmospheric_horror/visual_prompts.j2` (append optional block).
- Modify: `src/platinum/pipeline/visual_prompts.py` (accept `scene_filter` + `deviation_feedback` runtime knobs; selective apply).
- Modify: `src/platinum/cli.py` (`adapt` command — add `--rerun-rejected` flag).
- Test: `tests/integration/test_adapt_rerun_rejected.py` (new).

This is the most architecturally involved CLI task because it threads through the visual_prompts pipeline. Read `src/platinum/pipeline/visual_prompts.py` end-to-end before starting.

#### Step 1: Read `visual_prompts.py` to map the change surface

Run Read on `src/platinum/pipeline/visual_prompts.py`. Note:
- The `visual_prompts(...)` async function: takes `story` and a `recorder` (or live SDK), renders the user message via `render_template("visual_prompts.j2", ...)`, calls Claude with VISUAL_PROMPTS_TOOL, and `_zip_into_scenes` mutates each scene by index.
- `VisualPromptsStage.run` pulls deps from ctx and calls `visual_prompts(...)`.

#### Step 2: Write failing tests (offline replay via FixtureRecorder)

Create `tests/integration/test_adapt_rerun_rejected.py`:

```python
"""Integration tests for `platinum adapt --rerun-rejected`.

S7 §5.3.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from platinum.cli import app
from platinum.models.story import (
    Adapted, ReviewStatus, Scene, Source, StageRun, StageStatus, Story,
)


@pytest.fixture
def rejected_story_factory(tmp_path: Path, monkeypatch):
    """Build a story with story_curator + visual_prompts COMPLETE; one scene REJECTED."""
    monkeypatch.chdir(tmp_path)
    def _make() -> tuple[str, Path]:
        src = Source(
            type="gutenberg", url="https://example.com",
            title="Cask", author="Poe", raw_text="hello",
            fetched_at=datetime.now(timezone.utc), license="PD-US",
        )
        adapted = Adapted(
            title="Cask", synopsis="x", narration_script="y",
            estimated_duration_seconds=600.0, tone_notes="z",
        )
        scenes = [
            Scene(
                id="scene_001", index=1,
                narration_text="In the catacombs",
                visual_prompt="dark catacombs, candlelight",
                negative_prompt="bright daylight",
                review_status=ReviewStatus.REJECTED,
                review_feedback="too dark; need more amber lighting",
            ),
            Scene(
                id="scene_002", index=2,
                narration_text="Walking deeper",
                visual_prompt="deeper catacombs, torchlight",
                negative_prompt="bright daylight",
                review_status=ReviewStatus.APPROVED,
            ),
            Scene(
                id="scene_003", index=3,
                narration_text="Final cellar",
                visual_prompt="cellar, lanterns",
                negative_prompt="bright daylight",
                review_status=ReviewStatus.PENDING,
            ),
        ]
        story = Story(
            id="story_test", track="atmospheric_horror",
            source=src, adapted=adapted, scenes=scenes,
            stages=[
                StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(timezone.utc),
                         artifacts={"decision": "approved"}),
                StageRun(stage="story_adapter", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(timezone.utc)),
                StageRun(stage="scene_breakdown", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(timezone.utc)),
                StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                         completed_at=datetime.now(timezone.utc)),
            ],
        )
        d = tmp_path / "data" / "stories" / story.id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "story.json"
        story.save(path)
        return story.id, path
    return _make


def test_rerun_rejected_empty_set_exits_zero(tmp_path: Path, monkeypatch) -> None:
    """If no scenes are REJECTED, exit 0."""
    monkeypatch.chdir(tmp_path)
    src = Source(type="gutenberg", url="https://example.com", title="t",
                 author="a", raw_text="x",
                 fetched_at=datetime.now(timezone.utc), license="PD-US")
    adapted = Adapted(title="t", synopsis="x", narration_script="y",
                      estimated_duration_seconds=600.0, tone_notes="z")
    story = Story(
        id="story_x", track="atmospheric_horror",
        source=src, adapted=adapted,
        scenes=[
            Scene(id="scene_001", index=1, narration_text="x",
                  visual_prompt="p", negative_prompt="bright daylight",
                  review_status=ReviewStatus.APPROVED),
        ],
        stages=[
            StageRun(stage="story_curator", status=StageStatus.COMPLETE,
                     completed_at=datetime.now(timezone.utc),
                     artifacts={"decision": "approved"}),
            StageRun(stage="visual_prompts", status=StageStatus.COMPLETE,
                     completed_at=datetime.now(timezone.utc)),
        ],
    )
    d = tmp_path / "data" / "stories" / story.id
    d.mkdir(parents=True, exist_ok=True)
    story.save(d / "story.json")
    runner = CliRunner()
    result = runner.invoke(app, ["adapt", "--story", story.id, "--rerun-rejected"])
    assert result.exit_code == 0
    assert "no rejected" in result.output.lower()


def test_rerun_rejected_only_applies_new_prompts_to_REJECTED(
    rejected_story_factory, monkeypatch,
) -> None:
    """Re-run rewrites only the REJECTED scene; APPROVED+PENDING untouched."""
    story_id, story_path = rejected_story_factory()

    # Inject a synthetic recorder that returns 3 new prompts (one per scene)
    from tests._fixtures import FixtureRecorder

    fixture_dir = Path("tests/fixtures/anthropic/visual_prompts")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "rerun_rejected__1.json"
    fixture_path.write_text(json.dumps({
        "input": {"any": True},
        "output": {
            "tool_use": {
                "name": "visual_prompts",
                "input": {
                    "scenes": [
                        {"index": 1, "visual_prompt": "REWRITTEN scene 1 with amber",
                         "negative_prompt": "bright daylight"},
                        {"index": 2, "visual_prompt": "WOULD-be-rewritten scene 2",
                         "negative_prompt": "bright daylight"},
                        {"index": 3, "visual_prompt": "WOULD-be-rewritten scene 3",
                         "negative_prompt": "bright daylight"},
                    ]
                },
            },
            "usage": {"input_tokens": 100, "output_tokens": 100},
        },
    }))

    # Wire the recorder via env var that platinum.utils.claude.call honors via test seam.
    # The Config layer should pick up settings["test"]["claude_recorder"]; for this test we
    # monkeypatch the live call directly.
    from platinum.utils import claude as claude_mod
    fixture_record = FixtureRecorder(
        path=fixture_dir, mode="replay",
    )
    monkeypatch.setattr(claude_mod, "_live_call", lambda *a, **kw: (
        _ for _ in ()).throw(RuntimeError("must use recorder")))

    # Use the recorder via context settings injection:
    # adapt CLI reads cfg.settings["test"]["claude_recorder"] -- set via env or monkeypatch.
    # Simplest: monkeypatch Config to add the recorder.
    from platinum.config import Config
    original_init = Config.__init__
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.settings.setdefault("test", {})["claude_recorder"] = fixture_record
    monkeypatch.setattr(Config, "__init__", _patched_init)

    runner = CliRunner()
    result = runner.invoke(
        app, ["adapt", "--story", story_id, "--rerun-rejected"],
    )
    assert result.exit_code == 0, result.output

    rt = Story.load(story_path)
    # Scene 1 (was REJECTED): visual_prompt rewritten, status now REGENERATE
    assert rt.scenes[0].visual_prompt == "REWRITTEN scene 1 with amber"
    assert rt.scenes[0].review_status == ReviewStatus.REGENERATE
    assert rt.scenes[0].review_feedback is None
    # Scene 2 (was APPROVED): visual_prompt UNCHANGED
    assert rt.scenes[1].visual_prompt == "deeper catacombs, torchlight"
    assert rt.scenes[1].review_status == ReviewStatus.APPROVED
    # Scene 3 (was PENDING): visual_prompt UNCHANGED
    assert rt.scenes[2].visual_prompt == "cellar, lanterns"
    assert rt.scenes[2].review_status == ReviewStatus.PENDING
```

(Note: the test wiring above is rough — adjust to match the actual `FixtureRecorder` API by re-reading `tests/_fixtures.py`. The principle is the same: replay a canned visual_prompts response and assert the selective-apply behavior.)

#### Step 3: Update `visual_prompts.j2`

Edit `config/prompts/atmospheric_horror/visual_prompts.j2`. Append (before any closing block):

```jinja
{% if deviation_feedback %}

DEVIATION FEEDBACK (reviewer flagged the following scenes for rewrite):

{% for entry in deviation_feedback %}
- Scene {{ entry.index }}
  Current prompt: "{{ entry.current_prompt }}"
  Feedback: {{ entry.feedback }}
{% endfor %}

For each flagged scene, address the feedback while keeping the scene's
narration intent and the atmospheric_horror track aesthetic. Other scenes
should be re-emitted unchanged from their current visual_prompts.
{% endif %}
```

#### Step 4: Modify `visual_prompts.py` to accept `scene_filter` + apply selectively

In `src/platinum/pipeline/visual_prompts.py::visual_prompts(...)`, accept a new optional `scene_filter: set[int] | None = None` parameter. Inside, when rendering the template, also pass:

```python
deviation_feedback = []
if scene_filter is not None:
    for scene in story.scenes:
        if scene.index in scene_filter and scene.review_feedback:
            deviation_feedback.append({
                "index": scene.index,
                "current_prompt": scene.visual_prompt or "(cleared)",
                "feedback": scene.review_feedback,
            })
```

Pass `deviation_feedback=deviation_feedback` into `render_template`.

In `_zip_into_scenes` (the per-scene mutator), gate the apply:

```python
def _zip_into_scenes(story, scenes_input, *, scene_filter: set[int] | None = None):
    ...
    for entry in scenes_input:
        idx = entry["index"]
        if scene_filter is not None and idx not in scene_filter:
            continue
        scene = next(s for s in story.scenes if s.index == idx)
        scene.visual_prompt = entry["visual_prompt"]
        scene.negative_prompt = entry["negative_prompt"]
        # NEW: if this was a re-prompt (scene was REJECTED), reset status + clear keyframe
        if scene.review_status == ReviewStatus.REJECTED:
            scene.review_status = ReviewStatus.REGENERATE
            scene.review_feedback = None
            scene.keyframe_path = None
```

Add `scene_filter` and `ReviewStatus` imports as needed.

In `VisualPromptsStage.run`, pull `scene_filter` from `ctx.config.settings.get("runtime", {}).get("scene_filter")` and forward.

#### Step 5: Update `cli.py::adapt` to add `--rerun-rejected`

Add option:

```python
    rerun_rejected: bool = typer.Option(
        False, "--rerun-rejected",
        help="Re-run visual_prompts only for REJECTED scenes; uses review_feedback (S7 review loop).",
    ),
```

Inside the `adapt` body, before the eligibility loop, branch:

```python
    if rerun_rejected:
        # Different eligibility: visual_prompts COMPLETE + at least one REJECTED scene.
        from platinum.models.story import ReviewStatus
        eligible = []
        for story_dir in sorted(p for p in cfg.stories_dir.iterdir() if p.is_dir()):
            ...
            if story is not None and s.id != story:
                continue
            vp = s.latest_stage_run("visual_prompts")
            if vp is None or vp.status != StageStatus.COMPLETE:
                continue
            rejected_indices = {sc.index for sc in s.scenes
                                if sc.review_status == ReviewStatus.REJECTED}
            if not rejected_indices:
                continue
            eligible.append((s, rejected_indices))
        if not eligible:
            console.print("[yellow]No rejected scenes found.[/yellow]")
            raise typer.Exit(code=0)

        ctx = PipelineContext(config=cfg, logger=logging.getLogger("platinum.adapt"))
        # Run only the visual_prompts stage with scene_filter set per story
        for s, rejected_indices in eligible:
            cfg.settings.setdefault("runtime", {})["scene_filter"] = rejected_indices
            console.print(f"[cyan]Re-prompting {s.id} (scenes={sorted(rejected_indices)})...[/cyan]")
            stage = VisualPromptsStage()
            try:
                asyncio.run(stage.run(s, ctx))
            except Exception as exc:
                console.print(f"[red]{s.id} failed: {exc}[/red]")
                raise
        console.print(f"[green]Re-prompted {len(eligible)} story candidate(s).[/green]")
        return
```

#### Step 6: Run tests, confirm GREEN

Run: `python -m pytest tests/integration/test_adapt_rerun_rejected.py -v`
Expected: 2 passed (with `FixtureRecorder` plumbing right).

#### Step 7: Run full pytest, ruff

Run: `python -m pytest -q`
Expected: ~358 passed.

Run: `python -m ruff check src tests scripts`
Expected: clean.

#### Step 8: Commit

```bash
git add config/prompts/atmospheric_horror/visual_prompts.j2 src/platinum/pipeline/visual_prompts.py src/platinum/cli.py tests/integration/test_adapt_rerun_rejected.py
git commit -m "feat(visual_prompts): deviation_feedback + selective apply -- platinum adapt --rerun-rejected (S7 §5.3)"
```

---

### Task 19: End-to-end review-loop integration test

**Files:**
- Test: `tests/integration/test_review_full_loop.py` (new).

#### Step 1: Write the test

Create `tests/integration/test_review_full_loop.py`:

```python
"""End-to-end happy-path test for the full review loop.

Simulates the user workflow: render → review → reject some + regenerate
some → re-render → re-review → finalize.

Uses FakeComfyClient + MappedFakeScorer (no GPU), but exercises the real
keyframe_generator + decisions + finalize_review_if_complete.
S7 §6.2.
"""
from __future__ import annotations

# (full body left as a TDD exercise — pattern: build a 3-scene story,
#  run keyframe_generator with FakeComfyClient + MappedFakeScorer,
#  call apply_approve / apply_regenerate / apply_reject through decisions,
#  re-run keyframe_generator with --rerun-regen-requested filter,
#  approve all, assert finalize_review_if_complete writes the StageRun.)


def test_full_loop_approve_all_via_batch_then_finalize():
    """All scenes pass; batch-approve clears them; stage finalizes."""
    from platinum.review_ui import decisions
    from platinum.models.story import (
        Adapted, ReviewStatus, Scene, Source, Story, StageStatus,
    )
    from datetime import datetime, timezone
    from pathlib import Path

    src = Source(
        type="gutenberg", url="https://example.com", title="t", author="a",
        raw_text="x", fetched_at=datetime.now(timezone.utc), license="PD-US",
    )
    adapted = Adapted(
        title="t", synopsis="x", narration_script="y",
        estimated_duration_seconds=600.0, tone_notes="z",
    )
    scenes = []
    for i in range(3):
        s = Scene(
            id=f"scene_{i+1:03d}", index=i+1, narration_text=f"s{i}",
            visual_prompt=f"p{i}", negative_prompt="bright daylight",
        )
        s.keyframe_candidates = [Path(f"scene_{i+1:03d}/candidate_{c}.png") for c in range(3)]
        s.keyframe_scores = [5.5, 6.3, 5.9]
        s.keyframe_path = s.keyframe_candidates[1]
        scenes.append(s)
    story = Story(
        id="story_test", track="atmospheric_horror",
        source=src, adapted=adapted, scenes=scenes,
    )

    # All scenes selected score = 6.3 (above 6.0)
    decisions.apply_batch_approve_above(story, threshold=6.0)
    decisions.finalize_review_if_complete(story)
    assert all(s.review_status == ReviewStatus.APPROVED for s in story.scenes)
    run = story.latest_stage_run("keyframe_review")
    assert run is not None and run.status == StageStatus.COMPLETE


def test_full_loop_reject_then_regen_then_approve_then_finalize():
    """Rejected → regen → approve loop. Stage finalizes only at the end."""
    from platinum.review_ui import decisions
    from platinum.models.story import (
        Adapted, ReviewStatus, Scene, Source, Story, StageStatus,
    )
    from datetime import datetime, timezone
    from pathlib import Path

    src = Source(
        type="gutenberg", url="https://example.com", title="t", author="a",
        raw_text="x", fetched_at=datetime.now(timezone.utc), license="PD-US",
    )
    adapted = Adapted(
        title="t", synopsis="x", narration_script="y",
        estimated_duration_seconds=600.0, tone_notes="z",
    )
    scenes = []
    for i in range(2):
        s = Scene(
            id=f"scene_{i+1:03d}", index=i+1, narration_text=f"s{i}",
            visual_prompt=f"p{i}", negative_prompt="bright daylight",
        )
        s.keyframe_candidates = [Path(f"scene_{i+1:03d}/candidate_0.png")]
        s.keyframe_scores = [6.5]
        s.keyframe_path = s.keyframe_candidates[0]
        scenes.append(s)
    story = Story(id="t", track="atmospheric_horror", source=src,
                   adapted=adapted, scenes=scenes)

    # Approve scene 1, reject scene 2
    decisions.apply_approve(story, "scene_001")
    decisions.apply_reject(story, "scene_002", feedback="too dark")
    decisions.finalize_review_if_complete(story)
    # Not all approved → no StageRun
    assert story.latest_stage_run("keyframe_review") is None

    # Simulate `platinum adapt --rerun-rejected`: rewrite prompt, set REGENERATE
    story.scenes[1].visual_prompt = "rewritten with amber"
    story.scenes[1].review_status = ReviewStatus.REGENERATE
    story.scenes[1].review_feedback = None
    story.scenes[1].keyframe_path = None

    # Simulate `platinum keyframes --rerun-regen-requested`: regenerate keyframe
    story.scenes[1].keyframe_candidates = [Path("scene_002/candidate_0.png")]
    story.scenes[1].keyframe_scores = [6.7]
    story.scenes[1].keyframe_path = story.scenes[1].keyframe_candidates[0]
    story.scenes[1].review_status = ReviewStatus.PENDING

    # User reviews + approves
    decisions.apply_approve(story, "scene_002")
    decisions.finalize_review_if_complete(story)

    run = story.latest_stage_run("keyframe_review")
    assert run is not None
    assert run.status == StageStatus.COMPLETE
    assert run.artifacts["approved_count"] == 2
```

#### Step 2: Run, confirm GREEN immediately (no implementation gaps if Tasks 1-18 succeeded)

Run: `python -m pytest tests/integration/test_review_full_loop.py -v`
Expected: 2 passed. If anything fails, the issue is upstream — go fix it before continuing.

#### Step 3: Commit

```bash
git add tests/integration/test_review_full_loop.py
git commit -m "test(review_ui): end-to-end review loop integration test (S7 §6.2)"
```

---

## Layer F: Phase 1 closeout

### Task 20: Manual smoke against a synthetic 3-scene fixture

**Files:** none modified — manual exercise.

#### Step 1: Build a synthetic story on disk

In a Python REPL or scratch script:

```python
from datetime import datetime, timezone
from pathlib import Path
from platinum.models.story import (
    Adapted, ReviewStatus, Scene, Source, Story,
)
from tests._fixtures import make_synthetic_png

# Build a 3-scene story
src = Source(
    type="gutenberg", url="https://example.com", title="Cask",
    author="Poe", raw_text="hello",
    fetched_at=datetime.now(timezone.utc), license="PD-US",
)
adapted = Adapted(
    title="Cask Smoke", synopsis="x", narration_script="y",
    estimated_duration_seconds=600.0, tone_notes="z",
)
scenes = []
for i in range(3):
    s = Scene(
        id=f"scene_{i+1:03d}", index=i+1,
        narration_text=f"narration {i}",
        visual_prompt=f"a candle scene {i}, candlelit", negative_prompt="bright daylight",
    )
    scenes.append(s)
story = Story(
    id="story_smoke", track="atmospheric_horror",
    source=src, adapted=adapted, scenes=scenes,
)
story_dir = Path("data/stories") / story.id
keyframes_dir = story_dir / "keyframes"
story_dir.mkdir(parents=True, exist_ok=True)
for i, scene in enumerate(scenes):
    scene_dir = keyframes_dir / f"scene_{i+1:03d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    for c in range(3):
        png_path = scene_dir / f"candidate_{c}.png"
        make_synthetic_png(png_path, kind="gradient", seed=i*10+c)
        scene.keyframe_candidates.append(png_path)
    scene.keyframe_scores = [5.5 + 0.2*c for c in range(3)]
    scene.keyframe_path = scene.keyframe_candidates[2]  # auto-pick highest
story.save(story_dir / "story.json")
print(f"Synthetic story written to {story_dir}")
```

#### Step 2: Launch the review UI

Run: `python -m platinum review keyframes story_smoke --no-browser`
Expected: console prints `Review UI listening on http://127.0.0.1:5001/`. Open browser manually.

#### Step 3: Click through actions

In the browser:
- [ ] Page renders 3 tiles. Each shows thumbnail + prompt blurb + score.
- [ ] Click **Approve** on scene 1. Status pill turns green; rollup updates.
- [ ] Click **Regenerate** on scene 2. Status pill turns yellow; ↻1 badge appears.
- [ ] Click **Reject** on scene 3. Dialog opens. Type "needs more candle light". Submit. Status pill turns red.
- [ ] Click **View alternatives** on scene 1. Three thumbnails appear with scores. Click one. Page reloads.
- [ ] Click **Batch approve** with threshold 5.0. All PENDING (none in this state) get approved. (No effect since scene 1 is APPROVED, scene 2 REGENERATE, scene 3 REJECTED.)
- [ ] Verify `data/stories/story_smoke/story.json` reflects each click — `cat data/stories/story_smoke/story.json | python -m json.tool | grep -E 'review_(status|feedback)|regen_count'`.

#### Step 4: If anything is broken, fix and re-test before continuing

If a click doesn't update, or the JSON doesn't reflect — diagnose. Common issues:
- Inline JS syntax errors → check browser console.
- Image 404 → check `/image/...` route logs in Flask stdout.
- POST 400 → check request payload in browser DevTools Network tab.

#### Step 5: Tear down the smoke story

```bash
rm -rf data/stories/story_smoke
```

#### Step 6: Commit (no diff to commit if smoke clean)

If smoke surfaces a real bug, fix it, add a regression test, commit. If clean, skip this commit.

---

### Task 21: Final ruff + mypy + pytest sweep + closeout commit

**Files:** none modified, just verification.

#### Step 1: Run all tests

Run: `python -m pytest -q`
Expected: ~358-360 passed. 0 failures.

#### Step 2: Run ruff

Run: `python -m ruff check src tests scripts`
Expected: `All checks passed!`

#### Step 3: Run mypy

Run: `python -m mypy src`
Expected: `Found 2 errors in 2 files (checked N source files)` — the same two pre-existing deferrals from S6.4. Anything more is regression.

#### Step 4: Push to origin

Run: `git push origin main`
Expected: branch pushed. (User confirms before push.)

#### Step 5: Closeout commit (only if any miscellaneous cleanups remain)

If anything trivial — a stray comment, a missed import — sweep it.

```bash
git add -A
git commit -m "chore(s7): Phase 1 close -- review UI ready for live A6000 entry-test smoke"
git push origin main
```

If nothing to commit, skip.

---

# Phase 2 — Live A6000 entry-test smoke (USER-DRIVEN, requires GPU rental)

## Task 22: User-driven A6000 rental + full Cask 16-scene render

**Files:** none modified locally; runbook execution on remote box.

This phase requires a live vast.ai rental. **DO NOT proceed without user confirmation of the cost (~$1.30-1.80).**

### Step 1: Confirm with user

Output to user: "Phase 1 ships ($X spent so far on Claude in S6.4). Phase 2 needs an A6000 rental for ~$1.30-1.80 (60-75 min). Proceed?"

Wait for explicit go-ahead.

### Step 2: Confirm Cask story.json on local disk

Run: `ls data/stories/story_2026_04_25_001/story.json`
Expected: file exists. (Per S6.4 memory it does, with iter-2 visual_prompts.)

Run: `python -c "from platinum.models.story import Story; s = Story.load('data/stories/story_2026_04_25_001/story.json'); print('scenes:', len(s.scenes)); print('scene 1 visual_prompt:', s.scenes[0].visual_prompt[:200])"`
Expected: 16 scenes; scene 1 visual_prompt mentions lit features (e.g. "candlelight catching", per S6.4 iter-2 lit-pixel rule).

### Step 3: Search + rent A6000

Run: `vastai search offers 'gpu_name=RTX_A6000 cpu_ram>=64 disk_space>=80 verified=true' -o 'dph_total'`
Pick a cheap, verified offer.

Run: `vastai create instance <offer_id> --image pytorch/pytorch:latest --disk 80`
Note the instance id.

Run: `vastai show instances --raw` until status is `running`.

### Step 4: Provision the box

Run: `vastai ssh-url <instance_id>` → e.g. `ssh://root@ssh5.vast.ai:18526`.
Run: `ssh -i ~/.ssh/id_ed25519 -o UserKnownHostsFile=~/.ssh/known_hosts_vastai root@ssh5.vast.ai -p 18526`

On the box:
```bash
git clone https://github.com/aqillakhani/platinum /workspace/platinum
cd /workspace/platinum
bash scripts/vast_setup.sh
```

Expected: setup completes in ~12-15 min. Conda `p311` env created (per S6.4 Item 4). ComfyUI + score-server running. HF token resolved.

### Step 5: SCP local Cask story.json → box

From local:
```bash
scp -i ~/.ssh/id_ed25519 -P 18526 -o UserKnownHostsFile=~/.ssh/known_hosts_vastai \
  data/stories/story_2026_04_25_001/story.json \
  root@ssh5.vast.ai:/workspace/platinum/data/stories/story_2026_04_25_001/story.json
```

(Create the dir on the box first if needed: `ssh ... 'mkdir -p /workspace/platinum/data/stories/story_2026_04_25_001'`)

### Step 6: Preflight check

On the box:
```bash
/opt/conda/envs/p311/bin/python /workspace/platinum/scripts/preflight_check.py
```

Expected: all 5 checks pass (HF token resolves, workflow JSON valid, ComfyUI alive, score-server alive, workflow signature matches origin/main).

### Step 7: Run keyframe_generator for full Cask

On the box:
```bash
cd /workspace/platinum
/opt/conda/envs/p311/bin/python -m platinum keyframes story_2026_04_25_001
```

Expected: ~30-45 min runtime. 16 scenes × 3 candidates = 48 PNGs generated. Per-scene console output: `scene N selected candidate X (score=Y, fallback=Z)`. No `KeyframeGenerationError` halts.

If a scene halts, that's important data — note which scene and why (likely brightness / subject gate failure). Don't push past it without diagnosis.

### Step 8: SCP keyframes back to local

From local:
```bash
scp -ri ~/.ssh/id_ed25519 -P 18526 -o UserKnownHostsFile=~/.ssh/known_hosts_vastai \
  root@ssh5.vast.ai:/workspace/platinum/data/stories/story_2026_04_25_001/keyframes/ \
  data/stories/story_2026_04_25_001/

# Also re-pull the updated story.json (which now has keyframe_path + scores per scene)
scp -i ~/.ssh/id_ed25519 -P 18526 -o UserKnownHostsFile=~/.ssh/known_hosts_vastai \
  root@ssh5.vast.ai:/workspace/platinum/data/stories/story_2026_04_25_001/story.json \
  data/stories/story_2026_04_25_001/story.json
```

### Step 9: Tear down the box

Run: `vastai destroy instance <instance_id>`
Expected: instance destroyed. Don't skip.

### Step 10: Local — launch review UI against real keyframes

Run: `python -m platinum review keyframes story_2026_04_25_001`
Expected: browser opens to `http://127.0.0.1:5001/story/story_2026_04_25_001`. 16 tiles render.

---

## Task 23: Phase 2 closure — eye-check + closure decision

### Step 1: Eye-check 16 scenes

For each scene tile:
- Apply mental gold standard from S6.3: chiaroscuro that reads as a portrait/scene, not a fade-to-black; subjects are recognizable; atmospheric_horror tone honored.
- Click Approve / Reject / Regenerate per scene.

### Step 2: Count first-pass APPROVED

After clicking through all 16:
- ≥14 APPROVED → **closure met**. S6.3 + S6.4 quality work landed; S7 ships.
- <14 APPROVED → log failure patterns (which scenes; what's wrong with each). Likely an S7.1 retro debt session is needed.

### Step 3: If ≥14: optionally exercise rerun-rejected loop

If 1-2 scenes were rejected:
```bash
# Re-rent A6000 (small cost, ~$0.30-0.50)
vastai create instance ... && wait ... && bash setup ...
# scp story.json with the rejection feedback up
# /opt/conda/envs/p311/bin/python -m platinum adapt --rerun-rejected --story story_2026_04_25_001
# /opt/conda/envs/p311/bin/python -m platinum keyframes story_2026_04_25_001 --rerun-regen-requested
# scp keyframes back
# tear down
# Reload review UI; re-review the regenerated scenes
```

This validates the full review loop end-to-end. Optional — defer to a separate session if energy is low.

### Step 4: Phase 2 retro

Update `.claude/projects/.../memory/project_flux_workflow_s63.md` (or write a new memory file `project_s7_review_ui.md`) with:
- Phase 2 actual cost
- Phase 2 actual closure result (X of 16 first-pass)
- Any patterns that surfaced
- Whether the rerun-rejected loop was exercised

### Step 5: Phase 2 closeout commit

If `story.json` ended up with non-trivial review state we want to preserve (probably not — `data/stories/*` is gitignored), nothing to commit. Otherwise, sweep any documentation deltas:

```bash
git add docs/runbooks/vast-ai-keyframe-smoke.md  # if updated
git commit -m "docs(runbooks): vast-ai-keyframe-smoke -- S7 review-loop notes"
git push origin main
```

---

## Closure conditions for S7

- [ ] Phase 1: all tests green (~358 total), ruff clean, mypy unchanged, manual smoke clean.
- [ ] Phase 2: ≥14/16 Cask scenes approveable on first pass.
- [ ] Memory updated with Phase 2 actuals.
- [ ] `git log` shows ~21+ commits on main (one per task), all pushed.
- [ ] Cumulative S6.x + S7 spend under user's running budget (~$10 total across all sessions).

S8 (Wan 2.2 I2V video generator) is unblocked once S7 closes — approved keyframes feed into the video stage.

---

## Appendix A: Files touched (Phase 1)

**Created (12):**
- `src/platinum/review_ui/__init__.py`
- `src/platinum/review_ui/app.py`
- `src/platinum/review_ui/decisions.py`
- `src/platinum/review_ui/templates/base.html`
- `src/platinum/review_ui/templates/keyframe_gallery.html`
- `src/platinum/review_ui/templates/review_ui.js`
- `src/platinum/review_ui/static/style.css`
- `tests/unit/test_review_decisions.py`
- `tests/unit/test_review_app.py`
- `tests/unit/test_seed_regen_count.py`
- `tests/integration/test_review_command.py`
- `tests/integration/test_keyframe_rerun_regen_requested.py`
- `tests/integration/test_adapt_rerun_rejected.py`
- `tests/integration/test_review_full_loop.py`
- `tests/unit/test_story_model.py` (if not already present)

**Modified (5):**
- `src/platinum/models/story.py` (Scene fields + serialization)
- `src/platinum/pipeline/keyframe_generator.py` (`_seeds_for_scene` regen_count)
- `src/platinum/pipeline/visual_prompts.py` (scene_filter + selective apply)
- `src/platinum/cli.py` (review sub-app + `--rerun-*` flags)
- `config/prompts/atmospheric_horror/visual_prompts.j2` (deviation_feedback block)
- `pyproject.toml` (flask dep)

**Total commit count target:** ~21 commits on `main` (one per Phase 1 task + occasional sweep commits).
