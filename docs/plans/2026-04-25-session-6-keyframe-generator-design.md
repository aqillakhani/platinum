# Session 6 -- Keyframe generator + ComfyUI client + workflow injection (design)

**Date:** 2026-04-25
**Status:** Design approved; ready for implementation plan.
**Spec:** plan section 8 Session 6. **Predecessor:** Session 5 (`utils/aesthetics.py` Protocol/Fake, `utils/validate.py` five primitives), commit `b4e2429` + `54c4994` + `15138d1` + `697c58f`.

## Goal

Ship the offline-testable scaffolding for the keyframe generation stage: a `ComfyClient` Protocol with a Fake and an HTTP impl, a workflow-JSON injector, and a `KeyframeGeneratorStage` that drives 3 candidate generations per scene, scores them via the Session 5 `AestheticScorer` Protocol, anatomy-checks them via `check_hand_anomalies`, and persists a selection back onto each `Scene`.

By end of session:

- `src/platinum/utils/comfyui.py` exports `ComfyClient` Protocol, `FakeComfyClient`, `HttpComfyClient`.
- `src/platinum/utils/workflow.py` exports `load_workflow(name)` and `inject(workflow, ...)`.
- `config/workflows/flux_dev_keyframe.json` is a static, hand-written ComfyUI workflow template tagging variable nodes via a top-level `_meta` block.
- `src/platinum/pipeline/keyframe_generator.py` exports a pure async `generate_for_scene` + `generate`, a `KeyframeReport` dataclass, and a `KeyframeGeneratorStage` subclass.
- `MappedFakeScorer` is added to `utils/aesthetics.py` alongside the existing `FakeAestheticScorer` so tests can drive selection scenarios per-image.
- `RemoteAestheticScorer` is stubbed (raises `NotImplementedError` with a Session-6.1 pointer); tests do not exercise it.
- All ~34 new tests offline. `pytest -q` shows ~224 tests, 0 fail, 0 skip.

This session does NOT provision vast.ai, NOT generate real keyframes against live Flux, and NOT exercise the real LAION model. Those land in Session 6.1 as a focused smoke run once the offline contract is locked.

## Decisions (three brainstorming forks)

| # | Question | Decision |
|---|---|---|
| 1 | Session 6 scope -- offline-first, hybrid, or full live | **Offline-first.** All code + tests with FakeComfyClient + FakeAestheticScorer. Mirror Session 4/5 cadence. Live smoke becomes its own focused Session 6.1. Zero $ in this session, low risk. |
| 2 | Continuity feature scope -- baseline vs IP-Adapter vs full | **Baseline Flux only.** Workflow includes Flux.1 Dev + LAION scoring; IP-Adapter FaceID and ControlNet Depth deferred. Smallest surface, quickest path to a working pipeline. `atmospheric_horror.yaml`'s `use_ip_adapter_faceid: true` flag stays parsed but ignored by S6. |
| 3 | Real `AestheticScorer` impl -- ship a skeleton or defer entirely | **Stub only.** `RemoteAestheticScorer.__init__` raises `NotImplementedError("Session 6.1: implement SSH+script LAION scorer")`. Production code path is wired but visibly unfinished; tests inject `FakeAestheticScorer` / `MappedFakeScorer`. |

## Architecture

```
src/platinum/
  utils/
    comfyui.py              NEW (~180 lines)
      ComfyClient (Protocol)         async generate_image / upload_image / health_check
      FakeComfyClient (dataclass)    fixture-path map keyed by workflow signature
      HttpComfyClient (class)        vendored from gold + extended (transport, timeout)

    aesthetics.py           MODIFIED (+~10 lines)
      AestheticScorer (Protocol)     unchanged
      FakeAestheticScorer            unchanged
      MappedFakeScorer (NEW)         per-path score map for selection-scenario tests
      RemoteAestheticScorer (NEW)    stub; raises NotImplementedError in __init__

    workflow.py             NEW (~60 lines)
      load_workflow(name)            reads config/workflows/<name>.json
      inject(workflow, ...)          deepcopy + tagged-node mutation; pure

  pipeline/
    keyframe_generator.py   NEW (~220 lines)
      KeyframeReport (frozen dataclass)
      KeyframeGenerationError (Exception)
      generate_for_scene(...)        async pure function (3 candidates, score, anatomy, select)
      generate(...)                  iterate scenes -> list[KeyframeReport]
      KeyframeGeneratorStage         Stage subclass; per-scene checkpoint via story.save

config/workflows/
  flux_dev_keyframe.json    NEW   static workflow JSON; nodes tagged via _meta.role

tests/
  fixtures/
    keyframes/
      candidate_0.png       NEW (~1 KB synthetic 64x64 PNG, mid-grey)
      candidate_1.png       NEW (~1 KB synthetic 64x64 PNG, checkerboard)
      candidate_2.png       NEW (~1 KB synthetic 64x64 PNG, dark-grey)
  _fixtures.py              MODIFIED -- hoist `_make_fake_hands_factory` here from test_validate.py
  unit/
    test_workflow.py                  NEW (~6 tests)
    test_comfyui.py                   NEW (~10 tests)
    test_keyframe_generator.py        NEW (~14 tests)
    test_aesthetics.py                MODIFIED (+~3 tests for MappedFakeScorer + Remote stub)
  integration/
    test_keyframe_generator_stage.py  NEW (~4 tests)
```

**Zero new top-level dependencies.** `httpx` (already in core deps for Anthropic SDK and Session 2 fetchers) handles the ComfyUI HTTP transport; `Pillow` (already in core deps) writes synthetic PNG fixtures.

## Components

### `utils/comfyui.py`

```python
from typing import Protocol, runtime_checkable
from pathlib import Path

@runtime_checkable
class ComfyClient(Protocol):
    """Talk to a ComfyUI server that runs Flux/Wan/etc on a remote GPU.

    Workflow-injection is the caller's job; clients receive an already-shaped
    workflow dict. This separates the HTTP transport from the JSON schema.
    """

    async def generate_image(
        self,
        *,
        workflow: dict,           # already-injected workflow JSON
        output_path: Path,        # local path the resulting PNG lands at
    ) -> Path: ...

    async def upload_image(self, image_path: Path) -> str: ...

    async def health_check(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class FakeComfyClient:
    """Deterministic client for tests. Returns prebaked fixture PNGs.

    `responses` maps a *workflow signature* (sha256 of canonicalised JSON)
    to a list of fixture-PNG paths. Each generate_image call rotates through
    that list; once exhausted it reuses the last entry.
    """
    responses: dict[str, list[Path]]


class HttpComfyClient:
    """Real client; vendored shape from gold/utils/comfyui_client.py.

    Constructor takes (host, *, timeout, transport=None). transport plumbs
    through to httpx.AsyncClient -- tests pass MockTransport for unit-level
    HTTP-shape verification without using a Fake.
    """
    def __init__(self, host: str, *, timeout: float = 600.0, transport=None): ...
```

Notes:
- `generate_image` takes an *already-injected* workflow; it does not know about prompts/seeds. Separation: `workflow.py` knows JSON schema, `comfyui.py` knows transport.
- `transport=None` defaults to `httpx.AsyncHTTPTransport()`; tests pass `httpx.MockTransport(handler)` for unit-level HTTP-shape verification.
- `upload_image` exists for future IP-Adapter wiring; ships as a no-op-friendly endpoint so the Protocol does not need a refactor in S6.1.
- `timeout=600.0` because Flux can run 30-90s; keep slack.

### `utils/workflow.py`

```python
def load_workflow(name: str) -> dict:
    """Load config/workflows/<name>.json relative to the package root.

    Raises FileNotFoundError if the named workflow is not under config/workflows/.
    """

def inject(
    workflow: dict,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    output_prefix: str = "flux_dev",
) -> dict:
    """Return a new workflow dict with variable fields swapped in.

    Looks up nodes by `_meta.role` tag we add to the workflow JSON
    ("positive_prompt", "negative_prompt", "sampler", "save_image",
    "empty_latent"), so node-id renumbering doesn't break us.

    Raises KeyError if a required role is missing from `_meta`.
    """
```

Pure function: deepcopy the workflow, walk `_meta.role` -> node-id map, mutate inputs by node-id, return. No I/O after `load_workflow` returns. Required roles for `flux_dev_keyframe.json`: `positive_prompt`, `negative_prompt`, `sampler`, `empty_latent`, `save_image`.

### `config/workflows/flux_dev_keyframe.json`

Hand-authored ComfyUI workflow with the minimum nodes for Flux.1 Dev image generation. Top-level structure:

```json
{
  "_meta": {
    "title": "Flux.1 Dev keyframe baseline",
    "role": {
      "positive_prompt": "3",
      "negative_prompt": "4",
      "empty_latent": "5",
      "sampler": "6",
      "save_image": "8"
    }
  },
  "1": { "class_type": "UNETLoader", ... },
  "2": { "class_type": "DualCLIPLoader", ... },
  "9": { "class_type": "VAELoader", ... },
  "3": { "class_type": "CLIPTextEncode", "inputs": {"text": "<positive>", "clip": ["2", 0]} },
  "4": { "class_type": "CLIPTextEncode", "inputs": {"text": "<negative>", "clip": ["2", 0]} },
  "5": { "class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1} },
  "6": { "class_type": "KSampler", "inputs": {"seed": 0, "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]} },
  "7": { "class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["9", 0]} },
  "8": { "class_type": "SaveImage", "inputs": {"filename_prefix": "flux_dev", "images": ["7", 0]} }
}
```

Loader paths (`flux1-dev.safetensors`, `clip_l.safetensors`, `t5xxl_fp16.safetensors`, `ae.safetensors`) match what `vast_setup.sh` downloads. The workflow ships with default values that `inject` overwrites.

### `utils/aesthetics.py` additions

```python
@dataclass(frozen=True, slots=True)
class MappedFakeScorer:
    """Test scorer: returns score per image_path; falls back to default for unmapped."""
    scores_by_path: dict[Path, float]
    default: float = 0.0

    async def score(self, image_path: Path) -> float:
        return self.scores_by_path.get(image_path, self.default)


class RemoteAestheticScorer:
    """Real LAION-Aesthetics v2 scorer. Calls a Python script over SSH on the
    vast.ai box. Implementation lands in Session 6.1.
    """

    def __init__(self, *, host: str, ssh_user: str = "root", ssh_key_path: Path | None = None):
        raise NotImplementedError(
            "Session 6.1: implement SSH+script LAION scorer. "
            "Until then, inject FakeAestheticScorer / MappedFakeScorer in tests."
        )

    async def score(self, image_path: Path) -> float:
        raise NotImplementedError
```

The stub matters because the production `KeyframeGeneratorStage.run` references `RemoteAestheticScorer` as the default fallback. We want that path to exist *and* fail loudly with a clear pointer, not silently return zeros or import errors.

### `pipeline/keyframe_generator.py`

```python
@dataclass(frozen=True, slots=True)
class KeyframeReport:
    scene_index: int
    candidates: list[Path]
    scores: list[float]
    anatomy_passed: list[bool]
    selected_index: int
    selected_via_fallback: bool


class KeyframeGenerationError(Exception):
    """Raised when all candidates for a scene fail. Carries the per-candidate exceptions."""
    def __init__(self, scene_index: int, exceptions: list[BaseException]):
        ...


async def generate_for_scene(
    scene: Scene,
    *,
    track_visual: dict,
    quality_gates: dict,
    comfy: ComfyClient,
    scorer: AestheticScorer,
    output_dir: Path,
    n_candidates: int = 3,
    seeds: Sequence[int] | None = None,
    mp_hands_factory: Callable[[], Any] | None = None,
) -> KeyframeReport: ...


async def generate(
    story: Story,
    *,
    config: Config,
    comfy: ComfyClient,
    scorer: AestheticScorer,
    output_root: Path,
    mp_hands_factory: Callable[[], Any] | None = None,
) -> list[KeyframeReport]: ...


class KeyframeGeneratorStage(Stage):
    name = "keyframe_generator"

    async def run(self, ctx: PipelineContext, story: Story) -> Story: ...

    def is_complete(self, story: Story) -> bool:
        return all(s.keyframe_path is not None for s in story.scenes)
```

Two-tier: pure async functions take all dependencies as args; the `Stage` subclass is the impure shell that pulls dependencies from `ctx`. Same pure-core/impure-shell pattern as Session 3's curator and Session 4's adapter.

The DI seam reads `ctx.config.settings["test"]["comfy_client"]` and `ctx.config.settings["test"]["aesthetic_scorer"]`, mirroring Session 4 exactly. Production path defaults to `HttpComfyClient(...)` + `RemoteAestheticScorer(...)`.

## Data flow (one Story, N scenes, sequential)

```
KeyframeGeneratorStage.run(ctx, story)
  load workflow JSON
  for scene in story.scenes:
    if scene.keyframe_path is not None: continue          # checkpoint resume
    output_dir = data/stories/<id>/keyframes/scene_NNN/
    seeds = (scene.index*1000+0, +1, +2)
    for seed in seeds:
      prompt   = scene.visual_prompt + " " + track_visual.aesthetic
      negative = scene.negative_prompt or track_visual.negative_prompt
      wf = workflow.inject(template, prompt=..., negative_prompt=...,
                           seed=seed, output_prefix=f"scene_{idx:03d}")
      path = output_dir / f"candidate_{i}.png"
      try: await comfy.generate_image(workflow=wf, output_path=path)
      except: record exception; candidate gets score=0.0, anatomy=False
    for path in candidate_paths:
      score   = await scorer.score(path)
      anatomy = check_hand_anomalies(path, mp_hands_factory=...)
      eligible[i] = (score >= gates["aesthetic_min_score"]) and anatomy.passed
    if any(eligible):
      selected_index = argmax(score for i where eligible[i])  # ties -> lowest index
      fallback = False
    else:
      selected_index = 0
      fallback = True
    scene.keyframe_candidates = candidate_paths
    scene.keyframe_scores     = scores
    scene.keyframe_path       = candidate_paths[selected_index]
    scene.validation["keyframe_anatomy"] = [a.passed for a in anatomy_results]
    scene.validation["keyframe_selected_via_fallback"] = fallback
    story.save(...)                                       # atomic per-scene checkpoint
  append StageRun(stage="keyframe_generator", status=COMPLETE)
  story.save(...)
  return story
```

### Key properties

- **Per-scene checkpoint** via `story.save()` after each scene; a crash on scene 14 leaves scenes 1-13 intact, resume skips scenes whose `keyframe_path is not None`.
- **Determinism.** Seeds derived from `scene.index`; same Story re-run produces same candidate set. Lets us regression-test selection logic.
- **Failure isolation.** Per-candidate try/except converts a `comfy.generate_image` exception into `score=0.0, anatomy=False` rather than killing the whole run. If all 3 candidates of a scene throw, `KeyframeGenerationError` is raised, the StageRun is marked FAILED with the joined error text, and the Stage stops.
- **No coupling between selection and Stage.** `generate_for_scene` returns a `KeyframeReport`; the Stage maps reports back onto Scene fields. Pure -> mutate boundary at the Stage.

## Error handling

| Failure mode | Behaviour |
|---|---|
| `comfy.generate_image` raises for one candidate | Caught at per-candidate boundary. That candidate gets `score=0.0`, `anatomy_passed=False`, no path recorded. Other candidates continue. Logged at WARN. |
| `comfy.generate_image` raises for ALL 3 candidates of a scene | `generate_for_scene` raises `KeyframeGenerationError`. Stage catches, marks scene's StageRun FAILED with joined error text, stops processing remaining scenes. Resume re-tries from this scene. |
| `scorer.score` raises | Same as `generate_image`: convert to `score=0.0` for that candidate, log WARN, continue. |
| `check_hand_anomalies` raises (e.g., mediapipe model load fails) | Propagate. Deterministic env-level failure -- want loud failure, not silent rejection. |
| Workflow JSON missing or unreadable | `load_workflow` raises `FileNotFoundError` at Stage start, before any GPU calls. Loud, fail-fast. |
| Workflow `_meta.role` tag missing | `inject` raises `KeyError` at Stage start. Loud. |
| LAION returns NaN / inf | Treat as "did not pass aesthetic gate"; candidate stays eligible for fallback. No crash. |
| `story.save` fails mid-stage | Atomic-write means on-disk story.json is fully old or fully new -- never partial. Stage propagates I/O exception; resume re-runs the scene. |
| Scene with no `visual_prompt` populated | Stage raises `ValueError` at scene entry: "scene 7 has no visual_prompt; run `platinum adapt` first." Surface unmet upstream contract clearly. |

### Logging

`logging` (not `print`) at module level. Per-scene: one INFO on entry, one INFO per candidate, one INFO on selection with chosen score and fallback flag. Per Stage: one INFO at start, one at completion with `12/16 scenes selected via score, 3/16 via fallback, 1/16 failed`. Existing `utils/logger.py` setup carries forward unchanged.

## Testing strategy

### Test count and shape

| File | Type | Tests | What it covers |
|---|---|---|---|
| `tests/unit/test_workflow.py` | unit | ~6 | `load_workflow` reads from package data, `inject` swaps tagged nodes, raises `KeyError` on missing role, leaves untagged nodes alone, deepcopy semantics (no input-dict mutation), file-not-found |
| `tests/unit/test_comfyui.py` | unit | ~10 | `ComfyClient` Protocol shape, `FakeComfyClient` round-trip + call recording, `HttpComfyClient` happy path via `httpx.MockTransport`, polling timeout, `/view` byte download, `upload_image` form-data shape, health_check 200/non-200 |
| `tests/unit/test_keyframe_generator.py` | unit | ~14 | `generate_for_scene` happy path, all-fail fallback, anatomy rejects high-score candidate, scorer-injection, deterministic seeds per `scene.index`, ties -> lowest index, candidate exception isolation, missing `visual_prompt` raises, `KeyframeReport` shape, all-3-throw -> `KeyframeGenerationError`, per-candidate file naming |
| `tests/unit/test_aesthetics.py` | unit | +3 | `MappedFakeScorer` returns mapped score, falls back to default for unmapped, `RemoteAestheticScorer.__init__` raises with Session-6.1 pointer message |
| `tests/integration/test_keyframe_generator_stage.py` | integration | ~4 | Full `Stage.run` against `tmp_project` with seeded Story, checkpoint resume (idempotent), `story.json` round-trip with `keyframe_path` populated, failure path leaves a FAILED `StageRun` |

Total: ~37 new tests (3 of which are appended to `test_aesthetics.py`). Run time target: under 3s.

### Fixture strategy

Three permanent fixture PNGs committed to `tests/fixtures/keyframes/candidate_{0,1,2}.png`:

- 64x64 RGB
- `candidate_0.png` -- solid grey 128 (mid-luminance, no hands)
- `candidate_1.png` -- black + white checkerboard (high contrast, no hands)
- `candidate_2.png` -- solid grey 64 (low-luminance, no hands)

Generated via a one-shot Python helper that ships in `tests/_fixtures.py` (a `make_synthetic_png(path, kind)` helper), then committed so CI does not regenerate them. They are real PNGs (~1 KB each), not byte-blobs.

`FakeComfyClient` is constructed with a `responses` map keyed on a workflow signature (sha256 of canonicalised JSON, so seed + prompt distinguish them). Each call rotates through the configured fixture list. This lets one test simulate "candidate 0 = image A, candidate 1 = image B, candidate 2 = image C" without ever touching real ComfyUI.

`MappedFakeScorer` from `utils/aesthetics.py` lets tests drive selection scenarios: `MappedFakeScorer({path_a: 7.5, path_b: 5.0, path_c: 3.0})` makes candidate A win when threshold is 6.0; lowering threshold to 4.5 makes B eligible too and selection compares scores; threshold 8.0 fails all and triggers fallback.

### Mediapipe injection

`generate_for_scene` accepts `mp_hands_factory=None` and threads it into every `check_hand_anomalies` call. `_make_fake_hands_factory` from Session 5's `tests/unit/test_validate.py` is hoisted into `tests/_fixtures.py` so both modules share it. Tests inject the hoisted helper.

### TDD rhythm

Each task in the implementation plan follows the Session 5 cadence: (1) failing test, (2) run-see-fail, (3) minimum-implementation, (4) run-see-pass, (5) commit. Each task ends green.

### Quality bar (after Task N)

- `pytest -q` shows 190 + ~37 = ~227 tests, 0 fail, 0 skip.
- `ruff check src tests` clean.
- `mypy src` clean (no new errors; pre-existing `config.py:15` and `sources/registry.py:30` stay deferred per Session 5 review).
- One offline smoke run: instantiate `KeyframeGeneratorStage` against an in-memory Story with 3 scenes, drive it with `FakeComfyClient` + `MappedFakeScorer`, assert resulting `story.json` has all 3 `keyframe_path` fields populated and that two used the high-score candidate while one fell back. Print the StageRun summary.

### What is deliberately NOT tested in S6

- Real ComfyUI HTTP traffic. The `HttpComfyClient` `MockTransport` unit tests cover wire shape; the Session 6.1 smoke validates end-to-end behaviour against a real box.
- Real LAION scoring. `RemoteAestheticScorer` stub raises `NotImplementedError`; we test that it raises with the expected pointer message, but not that it produces correct scores.
- Workflow JSON correctness against a live ComfyUI. The JSON is hand-authored; first-time live failures surface in Session 6.1.

## Out of scope (deferred)

- **vast.ai provisioning** -- Session 6.1.
- **Real LAION-Aesthetics implementation** (SSH+script over to vast.ai) -- Session 6.1.
- **IP-Adapter FaceID** wiring -- Session 6.2 or piggybacked onto Session 8 video generator.
- **ControlNet Depth** wiring -- same as IP-Adapter, deferred.
- **Per-track `image_model` flag honoured** (`flux_dev` vs `flux_pro_api`) -- only `flux_dev` supported in S6; the flag is parsed but raises `NotImplementedError` if set to `flux_pro_api`.
- **Concurrent candidate generation per scene** -- sequential is fine; one GPU serialises Flux calls anyway.
- **Reset-stage CLI** (`platinum reset-stage --story X --stage keyframe_generator`) -- Session 7 concern when review UI surfaces "regenerate this scene."
- **Multi-track prompt authoring** -- still only `atmospheric_horror` populated under `config/prompts/`; copy templates per track when subsequent tracks are exercised.

## Lessons carried in

1. **Late binding for testability.** `mp_hands_factory=None` resolved at call time; same pattern as Session 3's `subprocess.run` injection, Session 4's `asyncio.sleep` retry, Session 5's mediapipe factory.
2. **Recorder/Fake protocol pattern.** `ComfyClient` Protocol + `FakeComfyClient` mirrors Session 4's `Recorder` + `FixtureRecorder` and Session 5's `AestheticScorer` + `FakeAestheticScorer`. Production injects the real impl; tests inject the fake. No global state, no monkeypatching network.
3. **Pure-core / impure-shell.** `generate_for_scene` and `generate` are pure async functions taking all dependencies as args; `KeyframeGeneratorStage` is the impure shell that pulls deps from `ctx`. Same as Session 3's `apply_decision` / `curate` split.
4. **Subagent scope drift on chore tasks.** Session 4's Task 24 chore-sweep accidentally committed `data/stories/` workspace data; Session 5 tightened guardrails. Session 6 dispatches stay narrow; verify diffs before claiming complete.
5. **Per-stage atomic save.** Story.save uses tmp + os.replace; per-scene save inside a Stage gives free crash-resilience. Already proven through Sessions 3-5.
