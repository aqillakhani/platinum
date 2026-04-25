# Session 5 -- Aesthetic scoring & validation utilities (design)

**Date:** 2026-04-25
**Status:** Design approved; ready for implementation plan.
**Spec:** plan section 8 Session 5. **Predecessor:** Session 4 (`story_adapter` / `scene_breakdown` / `visual_prompts`), commit `b0e2fd6`.

## Goal

Ship a small library of "quality gate" primitives that downstream sessions (6 keyframe generator / 8 video generator / 13 assembly) call on every produced asset before surfacing it to a human reviewer. By end of session:

- `src/platinum/utils/validate.py` exports a uniform `CheckResult` dataclass and five sync functions: `check_black_frames`, `check_motion`, `check_hand_anomalies`, `check_audio_levels`, `check_duration_match`.
- `src/platinum/utils/aesthetics.py` exports an `AestheticScorer` Protocol and a `FakeAestheticScorer`. Real GPU-backed implementation deferred to Session 6.
- Each `config/tracks/*.yaml` carries a `quality_gates:` block that supplies the thresholds.
- All five primitives are unit-tested offline against synthetic fixtures generated at test time. No binary fixtures committed.

This session does NOT wire the primitives into any pipeline `Stage`; it provides the leaf utilities that Session 6+ compose.

## Decisions (four brainstorming forks)

| # | Question | Decision |
|---|---|---|
| 1 | LAION-Aesthetics implementation strategy | **Defer real implementation to Session 6.** Ship `AestheticScorer` Protocol + `FakeAestheticScorer` only. The model wants a GPU and the cloud-first transition lands in Session 6 anyway. |
| 2 | Threshold config location | **Per-track YAML** (`track.quality_gates`). Horror tolerates darker frames than children's fables; per-track is the right granularity. |
| 3 | Test fixture strategy | **Synthetic at test time + mock mediapipe.** OpenCV / ffmpeg / duration tests use cv2.VideoWriter and the stdlib `wave` module; mediapipe gets mocked at its `mp.solutions.hands.Hands` boundary via a late-bound factory injection. |
| 4 | Return type contract | **Uniform `CheckResult` dataclass** -- `passed: bool`, `metric: float`, `threshold: float`, `reason: str`. All five functions named `check_*`; `passed=True` always means good. |

## Architecture

```
src/platinum/utils/
  +-- aesthetics.py        NEW (~30 lines)
  |     - AestheticScorer (Protocol)            # async score(image_path) -> 0.0-10.0
  |     - FakeAestheticScorer (dataclass)       # for tests
  |
  +-- validate.py          NEW (~200 lines)
        - CheckResult (frozen dataclass)        # passed, metric, threshold, reason
        - check_black_frames(video_path, *, max_black_ratio, luminance_threshold=8.0)
        - check_motion(video_path, *, min_flow_magnitude, sample_every_n_frames=6)
        - check_hand_anomalies(image_path, *, mp_hands_factory=None)
        - check_audio_levels(audio_path, *, target_lufs, tolerance_db)
        - check_duration_match(audio_path, *, target_seconds, tolerance_seconds)

config/tracks/
  +-- atmospheric_horror.yaml   MODIFIED -- add `track.quality_gates` block
  +-- folktales_world_myths.yaml, childrens_fables.yaml, scifi_concept.yaml,
      slice_of_life.yaml        MODIFIED -- same block, track-tuned values

tests/
  +-- _fixtures.py              MODIFIED -- add synthetic media helpers
  +-- unit/test_aesthetics.py   NEW (~3 tests)
  +-- unit/test_validate.py     NEW (~18 tests)
  +-- integration/test_quality_gates_config.py  NEW (~3 tests)
```

**Zero new dependencies.** All required packages (`opencv-python`, `mediapipe`, `ffmpeg-python`, `numpy`) are already in core deps and verified importable on the user's Python 3.14.3 environment.

## Components

### `utils/aesthetics.py`

```python
from pathlib import Path
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@runtime_checkable
class AestheticScorer(Protocol):
    """Score an image for cinematic quality on a 0.0-10.0 scale (LAION-Aesthetics v2)."""
    async def score(self, image_path: Path) -> float: ...

@dataclass(frozen=True, slots=True)
class FakeAestheticScorer:
    """Deterministic scorer for tests."""
    fixed_score: float

    async def score(self, image_path: Path) -> float:
        return self.fixed_score
```

`async` on the Protocol because the Session 6 real implementation will be SSH-to-vast.ai, naturally async. Setting that contract now means zero refactor when Session 6 lands. `@runtime_checkable` lets tests assert `isinstance(scorer, AestheticScorer)` without importing concrete types.

### `utils/validate.py`

```python
@dataclass(frozen=True, slots=True)
class CheckResult:
    passed: bool         # True = asset clears the gate
    metric: float        # measured value
    threshold: float     # what it was compared against
    reason: str          # human-readable, e.g. "0.8% black frames (max 1%)"
```

Five sync `check_*` functions. CPU-bound numpy / OpenCV / ffmpeg work; async wrapping would add noise without parallelism benefit. If a future Stage wants concurrency it does `asyncio.to_thread(check_motion, path, ...)` at the call site.

`check_hand_anomalies` accepts `mp_hands_factory=None` and resolves to `mediapipe.solutions.hands.Hands` at call time -- the same late-binding pattern that fixed Sessions 3 (`subprocess.run`) and 4 (`asyncio.sleep` retry). Tests inject a fake factory; production gets the real mediapipe singleton.

#### Function semantics

| Function | Returns `passed=True` when... | Implementation sketch |
|---|---|---|
| `check_black_frames` | share of frames whose mean luminance < `luminance_threshold` is at most `max_black_ratio` | cv2.VideoCapture loop, frame.mean() per frame, count below threshold |
| `check_motion` | mean Farneback dense optical flow magnitude across sampled frame pairs is at least `min_flow_magnitude` | cv2.calcOpticalFlowFarneback every Nth frame, magnitude.mean() |
| `check_hand_anomalies` | mediapipe detects no hands OR every detected hand has exactly 21 landmarks | `mp_hands_factory()` -> `.process(rgb)` -> count landmarks per detected hand |
| `check_audio_levels` | measured integrated LUFS is within `tolerance_db` of `target_lufs` | `ffmpeg -i ... -af loudnorm=print_format=json -f null -` parse JSON from stderr |
| `check_duration_match` | abs(measured_duration - target_seconds) <= tolerance_seconds | `ffprobe -show_entries format=duration` |

## Threshold YAML

New `quality_gates` block under `track:` in each track YAML. Initial values for `atmospheric_horror.yaml`:

```yaml
track:
  ...existing keys...

  # Quality gates -- automated thresholds enforced before human review
  quality_gates:
    aesthetic_min_score: 6.0          # LAION-Aesthetics v2 (0-10); horror runs ~0.5 lower than bright tracks
    black_frame_max_ratio: 0.02       # max share of near-black frames (horror has intentional fade-to-blacks)
    luminance_threshold: 8.0          # 0-255 mean below = "black frame"
    motion_min_flow_magnitude: 0.5    # Farneback dense flow, pixels/frame; below = still image
    audio_target_lufs: -16.0          # YouTube narration standard
    audio_lufs_tolerance_db: 1.5      # +/- around target
    duration_tolerance_seconds: 0.5   # narration audio vs. scene target
```

The same block is seeded into the four other track YAMLs with track-tuned `aesthetic_min_score` and `motion_min_flow_magnitude`. `audio_target_lufs` stays at -16 across all tracks.

## Data flow (Session 6 preview, NOT shipped this session)

```python
async def run(self, ctx, story):
    gates = ctx.config.track(story.track)["quality_gates"]
    scorer: AestheticScorer = ctx.scorer            # FakeAestheticScorer in tests, real impl in S6

    for scene in story.scenes:
        candidates = await generate_flux_candidates(scene, n=3)
        passing = []
        for path in candidates:
            score = await scorer.score(path)
            if score < gates["aesthetic_min_score"]:
                continue
            anatomy = check_hand_anomalies(path)
            if not anatomy.passed:
                logger.info("rejected scene %d candidate: %s", scene.idx, anatomy.reason)
                continue
            passing.append((score, path))
        scene.keyframe_path = max(passing)[1] if passing else candidates[0]
```

Thresholds come from YAML; scorer is dependency-injected; `CheckResult.reason` strings are ready for review-UI display. The pipeline orchestrator from Session 1 needs no changes -- quality gates are leaf-level utilities that Stages compose.

## Error handling

- Missing file -> raise `FileNotFoundError`. Loud, never swallowed.
- Corrupt video / cv2 returns False on first read -> `CheckResult(passed=False, reason="cannot read video")`.
- Audio file with no audio stream -> `CheckResult(passed=False, reason="no audio stream")`.
- ffmpeg / ffprobe not on PATH -> propagate `FileNotFoundError` with a clear message. We want this to fail loudly in CI, not silently pass.
- mediapipe model load failure -> propagate the underlying exception.
- LUFS measurement returns -inf (silent audio) -> `CheckResult(passed=False, metric=-inf, reason="silent audio (-inf LUFS)")`.

## Testing strategy

### Synthetic fixture helpers (`tests/_fixtures.py` extension)

```python
def make_test_video(path, *, n_frames=30, fps=24, color=(0,0,0), size=(64,64))
def make_test_video_with_motion(path, *, n_frames=30, fps=24, size=(64,64))
def make_test_audio(path, *, seconds=2.0, lufs=-16.0, freq_hz=440.0)
def make_silent_audio(path, *, seconds=2.0)
```

- cv2.VideoWriter produces a valid MP4 in ~50ms per fixture; tests use the `mp4v` codec for portability.
- WAV writing uses the stdlib `wave` module (no new dep).
- `make_test_audio(lufs=-16.0)` calibrates amplitude so that round-tripping through `check_audio_levels` measures within tolerance -- this exercises the real ffmpeg `loudnorm` path, not a mock.

### TDD checklist (~24 tests, run time under 2s)

`tests/unit/test_aesthetics.py` (3):
- `test_fake_scorer_returns_fixed_score`
- `test_fake_scorer_satisfies_protocol` -- `isinstance(scorer, AestheticScorer)` via `@runtime_checkable`
- `test_fake_scorer_score_is_awaitable`

`tests/unit/test_validate.py` (~18):
- Black frames: clean / partial-black / all-black / corrupt video / missing file
- Motion: static / moving / threshold-boundary / corrupt video
- Hand anomalies: no-hands / valid-hands / extra-landmarks / missing file (all via `mp_hands_factory` mock)
- Audio levels: in-range / too-loud / too-quiet / no-audio-stream / missing file
- Duration match: exact / within-tolerance / outside-tolerance / missing file

`tests/integration/test_quality_gates_config.py` (3):
- `test_atmospheric_horror_yaml_loads_gates` -- round-trip through `Config.track(...)["quality_gates"]`, all keys typed correctly
- `test_all_tracks_have_quality_gates` -- sweep `config/tracks/*.yaml`, assert presence
- `test_check_audio_levels_with_tone_round_trips` -- generate -16 LUFS tone via `make_test_audio`, verify `check_audio_levels(target_lufs=-16, tolerance_db=1.5)` passes (real ffmpeg, real path)

### Quality gates

- `pytest -q` -- Sessions 1+2+3+4+5 ~= 180 tests, 0 fail, 0 skip.
- `ruff check src tests` -- clean.
- `mypy src` -- clean.
- One smoke run: feed a generated -16 LUFS tone WAV plus a known-duration test video to the real `check_audio_levels` / `check_duration_match` functions, log the `CheckResult.reason` strings to confirm the messages read well.

## Out of scope (deferred)

- **Real `AestheticScorer` implementation** -- Session 6, on vast.ai GPU.
- **Stage subclasses wrapping the primitives** (e.g., `KeyframeValidatorStage`) -- Session 6 if needed; otherwise leaf-level composition by individual Stages.
- **Validation orchestrator** that runs all gates and aggregates results -- defer until a caller needs it; Session 6 will likely just call the primitives in sequence.
- **LUT calibration data, ASS subtitle validation, audio-clipping detection** -- not requested in plan section 8 Session 5.

## Lessons carried in

1. **Late binding for testability.** `mp_hands_factory=None` resolved at call time, just like Session 3's `subprocess.run` and Session 4's `asyncio.sleep`. Default-arg early binding is the antipattern.
2. **Recorder/Fake protocol pattern.** `AestheticScorer` Protocol + `FakeAestheticScorer` mirrors Session 4's `Recorder` Protocol + `FixtureRecorder`. Production injects the real impl; tests inject the fake. No global state, no monkeypatching network calls.
3. **Subagent scope drift on chore tasks.** When dispatching this session's tasks, keep each subagent's scope narrow and verify diffs before claiming complete. Session 4's Task 24 chore-sweep accidentally committed `data/stories/` workspace data; do not repeat.
