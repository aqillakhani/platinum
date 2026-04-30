# Session 8 — Wan 2.2 I2V video generator (Phase A) — Design

**Date:** 2026-04-30
**Scope:** Phase A only — per-scene Wan 2.2 image-to-video with quality gates and retry-once. Cross-scene last-frame chaining deferred to S8.B.
**Predecessor:** S7.1 (keyframe verify shipped 16/16 Cask on bare Flux + new visual_prompts language).
**Plan reference:** `C:\Users\claws\.claude\plans\i-added-a-prd-concurrent-book.md` Section 8 — Session 8.

---

## 1. Context

S7.1 verify produced 16 viable Cask of Amontillado keyframes (3 candidates per scene under `data/stories/story_2026_04_25_001/keyframes/`). With `Scene.keyframe_path` populated, S8's job is producing one 720p × 5s MP4 per scene via Wan 2.2 I2V-A14B running on vast.ai A6000.

The plan's full S8 spec bundles three architecturally distinct pieces: per-scene I2V, validation gates with retry, and cross-scene last-frame chaining for continuity. Phase A here ships the first two and defers chaining to S8.B if scene-to-scene continuity review (a manual eye-check after Rental 2) demands it.

This design follows the established platinum patterns: ComfyClient Protocol + Fake/Http variants (S6), pure-core/impure-shell at Stage boundary (S3+), `_meta.role`-based workflow injection (S6.3), per-scene atomic save with resume (S6.1), per-track YAML quality-gate thresholds (S5), preflight checks with workflow signature (S6.3), two-rental probe→full live verify (S6.3), and halt-on-fail with `VideoGenerationError` matching the keyframe halt pattern (S6.1+).

## 2. Brainstorming forks

Architectural decisions locked in during brainstorming, with the chosen option **bolded**.

### 2.1 Phase split — what ships in S8?

- **A. Phase A only — per-scene I2V + 3 gates + retry-once. Defer chaining to S8.B.**
- B. Phase A + last-frame chaining (full S8 spec).
- C. Bare-bones Phase A (no gates, no retry).

**Why A:** Mirrors S7.1.A / S6.1 ship-the-independent-piece pattern. Last-frame chaining requires sequential ordering, dual-image conditioning, and a new last-frame extraction primitive — all distinct enough to be its own session if needed. Validate.py already has black/motion/duration primitives; gates and retry are minimal additional code on top of bare I2V.

### 2.2 Live verify scope

- **A. Two-rental probe→full. Rental 1: 3 scenes (Cask 1, 8, 16). Rental 2: full 16-scene Cask.**
- B. Single rental, full 16-scene Cask (skip probe).
- C. Single rental, tiny smoke only (defer full Cask).

**Why A:** Three unknowns to derisk before committing full budget — Wan 2.2 weights URL may have moved, ComfyUI Wan workflow doesn't exist yet, and which extension provides Wan nodes is TBD. Matches S6.3 pattern exactly.

**Budget:** Rental 1 ≈ $2-5 (~30-45 min on A6000 at $0.42/hr including provision time). Rental 2 ≈ $5-15 (~80-100 min). Total ~$7-20. Within the project's per-film cost band.

### 2.3 Clip duration policy

- **A. Fixed 5s clips per scene (Wan 2.2 default 80 frames @ 16fps).**
- B. Variable per-scene capped at 10s.
- C. Per-scene multi-pass concatenation matching narration_duration_seconds fully.

**Why A:** Predictable VRAM and per-scene cost. Per-scene independent I2V is the Phase A scope — duration matching belongs in S13 assembly which can speed-stretch / pad / loop. Avoids variable-frame-count complexity until a downstream stage needs it.

### 2.4 Architectural approach

- **A. ComfyUI extension — mirror keyframe_generator.**
- B. Direct Wan CLI via SSH (new SshClient protocol).
- C. Standalone wan_server FastAPI daemon (mirror score_server).

**Why A:** Smallest delta to existing code. ComfyClient Protocol + FakeComfyClient + HttpComfyClient already cover the required surface (`upload_image / submit_workflow / wait / download`). New code is one pipeline module + one workflow JSON + one CLI command + minor `inject()` extension for Wan-specific roles.

## 3. Architecture overview

**Stage shape (mirrors `keyframe_generator.py`):**

```
VideoGeneratorStage (impure shell)
   │  pulls deps from ctx.config.settings["test"]["comfy_client"],
   │                  ctx.config.settings["runtime"]["scene_filter"]
   ▼
generate_video(story, *, comfy_client, gates_cfg, scene_filter)   # pure async
   │  iterates story.scenes; resume-on-video_path; respects scene_filter
   ▼
generate_video_for_scene(scene, *, comfy_client, gates_cfg)        # pure async
   │  Wan 2.2 I2V → MP4 → black_frames + motion + duration_match gates
   │  retry-once with new seed on content failure
   │  raise VideoGenerationError on infra failure or 2nd content fail
   ▼
returns VideoReport(scene_id, success, mp4_path, duration_s,
                    gates_passed, retry_used)
```

**Workflow shape (`config/workflows/wan22_i2v.json`):**

ComfyUI graph using `kijai/ComfyUI-WanVideoWrapper` extension nodes. `_meta.role` keys for `image_in` (LoadImage filename for keyframe), `prompt` (visual_prompt text → UMT5 encoder), `seed`, `width`, `height`, `frame_count`, `fps`. Output role `video_out` writes MP4 via `VHS_VideoCombine` or wrapper-provided equivalent. The `inject()` helper in `utils/workflow.py` extends to set Wan-specific roles, all optional like `model_sampling_flux` was.

**ComfyClient call sequence (per scene):**

1. `comfy_client.upload_image(scene.keyframe_path) → comfy_filename`
2. Build workflow with injected roles (image, prompt, seed, dimensions, frames=80 @ 16fps).
3. `comfy_client.submit_workflow(wf) → prompt_id`
4. `comfy_client.wait(prompt_id, timeout=600)` → output filenames map.
5. `comfy_client.download(remote_mp4, local_tmp_mp4)`

**Memory contention with Flux:** workflow includes `Comfy.UnloadAllModels` (or wrapper's `mm.soft_empty_cache`) at front of graph. Sequential phase model — keyframe phase finishes → video phase starts → Wan loads cleanly. No simultaneous Flux + Wan residency required.

**Hardware default:** A6000 48GB (proven in S6.3, $0.42/hr). Escalation to H100 80GB only if Rental 1 OOMs at fp16 × 720p × 5s.

## 4. Components

### 4.1 Files added

| Path | Purpose | Approx LoC |
|---|---|---|
| `src/platinum/pipeline/video_generator.py` | `VideoReport` dataclass, `VideoGenerationError`, `_seed_for_scene`, pure `generate_video_for_scene` + `generate_video`, impure `VideoGeneratorStage` | ~250 |
| `config/workflows/wan22_i2v.json` | Wan 2.2 I2V ComfyUI workflow with `_meta.role` keys | ~50 nodes |
| `tests/unit/test_video_generator.py` | Seed determinism, gate ordering, retry, halt rules, resume | ~12 tests |
| `tests/integration/test_video_generator_stage.py` | Stage integration with FakeComfyClient + synthetic MP4 fixtures | ~4 tests |
| `tests/integration/test_video_command.py` | CLI tests (dry-run, --scenes, missing story, missing keyframe) | ~5 tests |
| `tests/fixtures/videos/scene_001_5s.mp4` | Synthetic 5s MP4 fixture for Fake comfy responses | — |
| `docs/plans/2026-04-30-session-8-wan22-i2v-design.md` | This document | — |
| `docs/plans/2026-04-30-session-8-wan22-i2v-plan.md` | TDD task list (writing-plans output) | — |
| `docs/runbooks/vast-ai-video-smoke.md` | Two-rental probe→full runbook | — |

### 4.2 Files modified

| Path | Change |
|---|---|
| `src/platinum/utils/workflow.py` | `inject()` learns Wan roles: `image_in`, `frame_count`, `fps`, `width`, `height` (all optional, absence non-fatal — same pattern as `model_sampling_flux`). |
| `src/platinum/utils/comfyui.py` | `ComfyClient` Protocol gains `upload_image(path) -> str`. Both `FakeComfyClient` and `HttpComfyClient` implement. |
| `src/platinum/cli.py` | New `platinum video <story> [--scenes csv] [--dry-run] [--rerun-all]` command. |
| `src/platinum/pipeline/orchestrator.py` | Add `VideoGeneratorStage` to default pipeline assembly (gated by `keyframe_path is not None` precondition). |
| `config/tracks/{atmospheric_horror,folktales_world_myths,childrens_fables,scifi_concept,slice_of_life}.yaml` | New `video_gates:` block under `quality_gates`: `duration_target_seconds`, `duration_tolerance_seconds`, `black_frame_max_ratio`, `motion_min_flow`. |
| `scripts/vast_setup.sh` | Replace stale single-file Wan URL with MoE expert files + VAE + UMT5 text encoder; clone `ComfyUI-WanVideoWrapper`; install its requirements. |
| `scripts/preflight_check.py` | +3 checks: Wan workflow JSON valid, Wan weights present (4 files: 2 experts + VAE + UMT5), WanVideoWrapper extension importable. |
| `tests/_fixtures.py` | New `make_synthetic_mp4(path, duration_s=5.0, motion=True)` (cv2.VideoWriter, mp4v codec). |
| `tests/unit/test_workflow.py` | +5 tests for new Wan-specific roles. |
| `tests/unit/test_comfyui.py` | +6 tests: `upload_image` on Fake + Http (httpx.MockTransport for wire shape). |
| `tests/integration/test_quality_gates_config.py` | +1 parametrized (5 cases) for `video_gates` block in all tracks. |
| `tests/unit/test_preflight_check.py` | +3 tests for the new checks. |
| `tests/unit/test_fixture_helpers.py` | +2 tests for `make_synthetic_mp4`. |

**Test count:** ~38 net new. 521 → ~559 total.

**No new top-level Python dependencies.** WanVideoWrapper is box-only (ComfyUI extension); not added to `pyproject.toml`.

## 5. Data flow

### 5.1 Per-scene happy path

```
Scene (input)
├── id: "scene_007"
├── keyframe_path: data/stories/<id>/keyframes/scene_007/candidate_2.png
├── visual_prompt: "Underground crypt, single torch flickering, robed figure..."
├── narration_duration_seconds: 23.4 (informational only — not used in S8 Phase A)
└── video_path: None  ← target

         │
         ▼  generate_video_for_scene
         │
   ┌─────┴─────────────────────────────────────────┐
   │ 1. comfy.upload_image(keyframe_path)          │
   │    → "scene_007_input.png" (server-side)      │
   ├───────────────────────────────────────────────┤
   │ 2. inject(workflow, roles={                   │
   │       image_in: "scene_007_input.png",        │
   │       prompt: scene.visual_prompt,            │
   │       seed: 7000 + retry,                     │
   │       width: 1280, height: 720,               │
   │       frame_count: 80, fps: 16                │
   │    })                                         │
   ├───────────────────────────────────────────────┤
   │ 3. comfy.submit_workflow(wf) → prompt_id      │
   │    comfy.wait(prompt_id, timeout=600)         │
   │    comfy.download(remote_mp4, local_tmp_mp4)  │
   ├───────────────────────────────────────────────┤
   │ 4. Quality gates (in order):                  │
   │    a. check_duration_match (5.0 ± 0.2s)       │
   │    b. check_black_frames   (max 5% black)     │
   │    c. check_motion         (flow > track min) │
   │                                               │
   │    All pass → keep MP4                        │
   │    Any fail + retry==0 → bump retry, re-seed, │
   │                          GOTO step 2          │
   │    Any fail + retry==1 → raise VideoGenError  │
   ├───────────────────────────────────────────────┤
   │ 5. atomic move local_tmp_mp4 →                │
   │    data/stories/<id>/clips/scene_007_raw.mp4  │
   ├───────────────────────────────────────────────┤
   │ 6. Scene.video_path = clips/scene_007_raw.mp4 │
   │    Scene.video_duration_seconds = 5.0         │
   │    story.json saved atomically (tmp+replace)  │
   └───────────────────────────────────────────────┘
```

### 5.2 Filesystem layout under `data/stories/<id>/`

```
keyframes/                      ← already populated by S7.1 verify
   scene_001/candidate_{0,1,2}.png
   ...

clips/                          ← S8 output (gitignored)
   scene_001_raw.mp4            ← 720p × 5s, Wan 2.2 output
   scene_002_raw.mp4
   ...

story.json                      ← updated atomically per scene
   scenes[i].video_path = "clips/scene_001_raw.mp4"
   scenes[i].video_duration_seconds = 5.0
```

### 5.3 Whole-story flow (`generate_video`)

1. Iterate `story.scenes` in order.
2. For each scene: if `video_path is not None` and file exists → skip (resume).
3. If `scene_filter` set and `scene.index not in scene_filter` → skip.
4. If `keyframe_path is None or not file.exists()` → halt with `VideoGenerationError("scene N has no keyframe")`.
5. Call `generate_video_for_scene` → updates Scene in-place.
6. After each scene: `story.save()` atomic write.

### 5.4 Dependencies on prior stages

- `Scene.keyframe_path` set by `KeyframeGeneratorStage` (S6+). S8 has nothing to do if keyframes absent — preflight catches this before any GPU time.
- `Scene.visual_prompt` populated by `VisualPromptsStage` (S4). Used as text conditioning for Wan I2V.

## 6. Error handling

### 6.1 Failure taxonomy

| Mode | Trigger | Response |
|---|---|---|
| Infrastructure | ComfyUI HTTP error, timeout, OOM, weights missing, workflow signature mismatch | Halt immediately. Raise `VideoGenerationError(scene_id, reason, retryable=False)`. No retry. Scene.video_path stays None — re-run resumes. |
| Content | Gates fail (motion below floor, >5% black frames, duration off by >0.2s) on first generation | Retry once with `seed = base_seed + 1`. Same workflow, same image, same prompt. |
| Content (post-retry) | Gates still fail on second generation | Halt with `VideoGenerationError(scene_id, reason, retryable=True)`. User decides: re-run for fresh seeds, edit visual_prompt, or relax track gate. |

### 6.2 Why halt-not-skip

Matches `KeyframeGenerationError` pattern S6.1/S6.2/S6.3 established. Halt makes failures visible immediately rather than silently producing a story with N missing clips. Per-scene atomic save means a halt at scene 7 leaves scenes 1-6 valid and resumable.

### 6.3 Retry semantics

```python
base_seed = scene.index * 1000          # deterministic
seeds = [base_seed, base_seed + 1]      # initial + retry-1
# Disjoint from keyframe_generator's seed scheme (scene.index * 1000 + candidate_idx),
# so re-runs are deterministic and user-recoverable.
```

### 6.4 Halt-on-precondition (run before any GPU work)

`generate_video` validates these before opening a ComfyUI session:

- All targeted scenes have `keyframe_path` set and file exists.
- Workflow JSON parses and contains all required `_meta.role` keys.
- `gates_cfg` (per-track YAML) has all four `video_gates` thresholds defined.

Any precondition fails → `VideoGenerationError` before any network call.

### 6.5 Resource lifecycle

Stage owns ComfyClient lifetime — `try / finally: await comfy.aclose()` (S6.1's lesson #6 fix). Test-injected ComfyClient skips aclose.

### 6.6 No silent failures

Every failure path either halts with a `VideoGenerationError` carrying scene_id + reason, or increments retry_used and tries again deterministically. A clean run terminates with every targeted scene's `video_path` set, or it raises.

## 7. Testing

### 7.1 Test pyramid

| Layer | File | Tests | What it covers |
|---|---|---|---|
| Unit (pure) | `tests/unit/test_video_generator.py` | ~12 | Seed determinism, gate ordering (duration → black → motion), retry path, halt rules, resume, precondition validation, `VideoReport` shape. |
| Unit (workflow) | `tests/unit/test_workflow.py` | +5 | New roles `image_in`, `frame_count`, `fps`, `width`, `height` (each optional). |
| Unit (comfyui) | `tests/unit/test_comfyui.py` | +6 | `upload_image` on Fake (deterministic filename) + Http (httpx.MockTransport asserting POST `/upload/image` multipart). |
| Integration (Stage) | `tests/integration/test_video_generator_stage.py` | ~4 | End-to-end with FakeComfyClient + synthetic MP4 fixtures: happy path (3 scenes), resume (1 pre-set), retry (Fake returns black MP4 first call → motion MP4 second), halt (Fake raises ComfyError). Stage closes ComfyClient via finally. |
| Integration (CLI) | `tests/integration/test_video_command.py` | ~5 | `platinum video <story>` dry-run; `--scenes 1,8,16`; missing story id; missing keyframe_path precondition; retry counter exposed in JSON output. |
| Integration (config) | `tests/integration/test_quality_gates_config.py` | +1 parametrized (5 cases) | Per-track `video_gates` block parses cleanly with thresholds in plausible ranges. |
| Unit (preflight) | `tests/unit/test_preflight_check.py` | +3 | Wan workflow JSON valid; Wan weights file >0 bytes (×4); WanVideoWrapper extension importable. |
| Unit (fixtures) | `tests/unit/test_fixture_helpers.py` | +2 | `make_synthetic_mp4(path, duration_s, motion=True)` produces valid MP4 of exact duration; motion=False produces single repeated frame (retry-fail-fixture path). |

### 7.2 Fakes vs real boundaries

- **FakeComfyClient** — extended with `upload_image` returning deterministic filename + per-call MP4-fixture rotation (sha256-keyed like S6 keyframe candidate rotation). Tests inject pre-generated MP4 fixtures from `tests/fixtures/videos/`.
- **httpx.MockTransport** — covers `HttpComfyClient.upload_image` wire shape (POST `/upload/image`, multipart with `image` field, returns `{"name": "...", "subfolder": "", "type": "input"}`).
- **Synthetic MP4 fixtures** — `make_synthetic_mp4` writes 5s 720p MP4s with optional motion (frame translation) using cv2.VideoWriter + `mp4v` codec. Same pattern as S5's `make_test_video_with_motion`. Stored in `tests/fixtures/videos/`.
- **No torch / Wan weights ever loaded in tests.** All Wan-specific behavior comes from fixture MP4s.

### 7.3 TDD ordering

1. Fixture helpers first (`make_synthetic_mp4`).
2. `workflow.inject()` new roles (failing tests → impl → green).
3. `ComfyClient.upload_image` (Fake first, Http+MockTransport second).
4. `VideoReport` + pure functions (`_seed_for_scene`, `generate_video_for_scene`, `generate_video`).
5. `VideoGeneratorStage` (impure shell + finally aclose).
6. CLI integration.
7. Track YAML `video_gates` blocks.
8. Preflight check additions.
9. Final cumulative review (S6.1 Lesson #6: catches resource-leak class bugs).

ruff / mypy clean required before each commit. Task-by-task commits expected ~25.

## 8. Box-side setup

### 8.1 vast_setup.sh additions

Replace the current single-file `wan22_i2v.safetensors` URL (wrong format — Wan 2.2 ships as MoE) with:

```bash
# --- ComfyUI extension for Wan 2.2 nodes ---
git_clone_or_update "https://github.com/kijai/ComfyUI-WanVideoWrapper" \
   "$COMFY_DIR/custom_nodes/ComfyUI-WanVideoWrapper"
pip install -r "$COMFY_DIR/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt"

# --- Wan 2.2 I2V-A14B weights (MoE: 2 experts + VAE + text encoder) ---
# NOTE: validate URLs during S8 probe — Wan-AI HF paths shift historically.
WAN_BASE="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main"
dl "$WAN_BASE/high_noise_model.safetensors" \
   "$MODELS_DIR/diffusion_models/wan2_2_i2v_high_noise.safetensors"
dl "$WAN_BASE/low_noise_model.safetensors" \
   "$MODELS_DIR/diffusion_models/wan2_2_i2v_low_noise.safetensors"
dl "$WAN_BASE/Wan2.1_VAE.pth" \
   "$MODELS_DIR/vae/wan2_2_vae.pth"
dl "$WAN_BASE/models_t5_umt5-xxl-enc-bf16.pth" \
   "$MODELS_DIR/text_encoders/umt5_xxl.pth"
```

### 8.2 Disk + VRAM budget on A6000

| Asset | Disk | Loaded VRAM |
|---|---|---|
| Flux Dev UNet (already there) | 23 GB | 23 GB (when loaded) |
| Flux ae + clip_l + t5xxl | ~6 GB | ~5 GB |
| Wan 2.2 high-noise expert | ~14 GB | ~14 GB (when loaded) |
| Wan 2.2 low-noise expert | ~14 GB | ~14 GB (when loaded) |
| Wan 2.2 VAE | ~250 MB | ~250 MB |
| UMT5-xxl text encoder | ~5 GB | ~5 GB |
| Other (LAION MLP, IPA leftovers) | ~3 GB | — |
| **Total disk** | ~65 GB | (vast.ai persistent volume) |
| **Peak VRAM during Wan inference** | — | ~33 GB (one expert + VAE + UMT5 active) |

A6000 (48 GB) has comfortable headroom for Wan 2.2 fp16 at 720p × 5s. The Flux + Wan total disk is ~65 GB; current vast_setup.sh already pre-cleans pip cache + whisper + chatterbox to fit.

### 8.3 Sequential expert switching

WanVideoWrapper handles the high-noise → low-noise expert switching internally during sampling (high-noise expert runs the first ~50% of denoising steps, low-noise the remaining). Workflow references both experts; wrapper swaps as needed. Memory peak ≈ one expert at a time.

### 8.4 Flux unload before Wan run

Workflow includes a `Comfy.UnloadAllModels` (or wrapper's `mm.soft_empty_cache`) node at the front of the graph. Sequential execution model: keyframe phase finishes → video phase starts → first Wan call evicts Flux → Wan loads cleanly.

### 8.5 Preflight check additions (`scripts/preflight_check.py`)

| Check | Implementation |
|---|---|
| Wan 2.2 weights present | `Path("/workspace/models/diffusion_models/wan2_2_i2v_high_noise.safetensors").stat().st_size > 1_000_000_000` and same for low_noise expert. |
| Wan VAE + UMT5 present | `stat().st_size > N` checks per file. |
| WanVideoWrapper extension importable | `import sys; sys.path.insert(0, ".../ComfyUI-WanVideoWrapper"); import nodes` or wrapper's specific entry point. |
| Wan workflow signature | `_workflow_signature(path)` sha256 prefix logged at preflight; mismatch with expected → halt (S6.3 lesson #8). |
| ComfyUI version | `git -C /workspace/ComfyUI rev-parse HEAD` for diagnostic; WanVideoWrapper requires recent core. |

### 8.6 Runbook

New `docs/runbooks/vast-ai-video-smoke.md`:

- **Step 1.** Provision A6000 + run `vast_setup.sh` (~15-20 min for Wan additions).
- **Step 2.** SSH in, run `preflight_check.py --video` (~30s).
- **Step 3.** **Rental 1 (probe).** `platinum video story_2026_04_25_001 --scenes 1,8,16`. Expect ~3 × 5min Wan calls = 15 min wall clock + ~1 min orchestration. Eye-check 3 MP4s on local. Closure: ≥2 of 3 viable (coherent 5s clip with motion, not corrupt, not still).
- **Step 4.** If probe passes: **Rental 2 (full).** `platinum video story_2026_04_25_001`. Expect ~16 × 5min = 80 min wall clock. Eye-check 16 MP4s locally. Closure: ≥14/16 viable on first eye-check.
- **Step 5.** Calibrate `video_gates` thresholds against actual-output metrics if false positives during the run.
- **Step 6.** `vastai destroy instance <id>` (no S9 chain — upscaling deferred).

### 8.7 Gotcha checklist (folded from prior sessions)

- HF token must be set on box (`HF_TOKEN`) — Wan-AI weights are gated. Same `huggingface-cli login --token` step Flux had in S6.
- `git archive` from Windows preserves CRLF in `.sh` files → `dos2unix vast_setup.sh` on box (S7.1 gap #11). Long-term fix via `.gitattributes` still pending.
- Workflow signature stamping in preflight catches stale-workflow-on-box (S6.3 lesson #8).
- Story.json POSIX paths on box (S7.1 gap #12 — already fixed in `Story.save`; regression-test still good).

## 9. Quality gates calibration

### 9.1 Per-track `video_gates` block

```yaml
# config/tracks/atmospheric_horror.yaml
track:
  quality_gates:
    # ... existing image gates (brightness_floor, subject_min_edge_density, ...)
    video_gates:
      duration_target_seconds: 5.0
      duration_tolerance_seconds: 0.2     # ±0.2s acceptable
      black_frame_max_ratio: 0.05         # max 5% near-black frames
      motion_min_flow: 0.3                # min mean optical flow magnitude
```

### 9.2 Per-track `motion_min_flow` initial values

| Track | motion_min_flow | Rationale |
|---|---|---|
| atmospheric_horror | 0.3 (low) | Static-mood scenes (candle in crypt, motionless figure) legitimately have low flow |
| folktales_world_myths | 0.5 | Mid — narrative motion expected |
| childrens_fables | 0.7 (high) | Active characters; static scenes uncommon |
| scifi_concept | 0.5 | Mid — varies by scene type |
| slice_of_life | 0.6 | Generally active |

These are deliberately permissive at first. The S6.3 calibration pattern: ship initial threshold → observe actual probe metrics → tighten if false-positive rate is high.

### 9.3 Constants across tracks

- `black_frame_max_ratio: 0.05` — Wan 2.2 occasionally produces a single-frame flash near transitions; >5% indicates real corruption.
- `duration_tolerance_seconds: 0.2` — Wan 2.2 frame counts are exact at 80 frames @ 16fps = 5.0s; tolerance covers ffmpeg metadata rounding.
- `duration_target_seconds: 5.0` — Phase A constant. S8.B / S13 may parameterize this per-scene if duration matching becomes an active concern.

## 10. CLI shape

```
Usage: platinum video [OPTIONS] STORY_ID

  Generate Wan 2.2 I2V clips for each scene's keyframe.

  All scenes with a populated keyframe_path produce a 5-second 720p clip
  written to data/stories/<id>/clips/scene_NNN_raw.mp4.

  Scenes with an existing video_path that points to an existing file are
  skipped (resume).

Options:
  --scenes TEXT      Comma-separated 1-indexed scene numbers (e.g., "1,8,16").
                     Default: all scenes with keyframe_path set.
  --dry-run          Print what would be generated; submit no workflows.
  --rerun-all        Force regeneration of all scenes (ignore existing video_path).
                     Mirrors --rerun-all-prompts from S7.1 adapt command.
  --help             Show this message and exit.
```

### 10.1 Resume modes

| Mode | Behavior |
|---|---|
| Default | Skip scenes with `video_path` populated and file exists; generate the rest. |
| `--scenes 1,8,16` | Generate only those scenes; respects existing `video_path` for skipping unless `--rerun-all`. |
| `--rerun-all` | Wipe `video_path` for every targeted scene, regenerate from scratch. Files on disk overwritten atomically. |

### 10.2 Output JSON

Printed at run end (mirrors keyframe stage):

```json
{
  "story_id": "story_2026_04_25_001",
  "scenes_total": 16,
  "scenes_succeeded": 16,
  "scenes_skipped_resume": 0,
  "scenes_failed": 0,
  "retries_used": 2,
  "wall_clock_seconds": 4830.5
}
```

## 11. Closure target

> **≥14 of 16 Cask scenes produce viable 5-second 720p clips on first eye-check after Rental 2's full run.**

"Viable" = not corrupt, has visible motion (not a still frame), keyframe content recognizably present. Same threshold S7.1 used for keyframe approveability.

If <14, scope an S8.A.1 follow-up before moving to S9 (upscaling). Likely root causes to investigate in that case: motion gate calibration, prompt-language for I2V (different from keyframe prompts?), Wan parameter tuning (frame count, sampling steps).

## 12. Out of scope (deferred)

| Item | Deferred to | Reason |
|---|---|---|
| Cross-scene last-frame chaining | S8.B | Architecturally distinct from per-scene I2V; fold in only if Rental 2 review shows discontinuity is the dominant quality issue. |
| Manual video review UI | S8.B or S15 | Phase A closure is filesystem eye-check, like S7.1.A. |
| Variable-duration clips matching narration | S13 assembly | Speed-stretch / pad / loop is assembly's job. |
| Multi-pass concatenation for >10s scenes | S8.B | Phase A is fixed 5s. |
| RealESRGAN 1080p upscale | S9 | Plan section 9. |
| Per-clip exposure normalization + LUT grading | S14 | Plan section 14. |
| Automated cross-clip continuity score | (no plan slot) | Defer indefinitely unless review surfaces a need. |

## 13. References

- **Plan:** `C:\Users\claws\.claude\plans\i-added-a-prd-concurrent-book.md` Section 8.
- **PRD:** `short_film_pipeline_prd.md`.
- **Predecessor session memo:** S7.1 verify complete — `C:\Users\claws\.claude\projects\C--Users-claws-OneDrive-Desktop-platinum\memory\project_s7_1_verify_complete.md`.
- **Pattern references:**
  - `keyframe_generator.py` — Stage shape, retry/halt, atomic save
  - `utils/comfyui.py` — ComfyClient Protocol, FakeComfyClient, HttpComfyClient
  - `utils/workflow.py` — `inject()` `_meta.role` pattern
  - `utils/validate.py` — `check_black_frames`, `check_motion`, `check_duration_match`
  - S6.3 design (`docs/plans/2026-04-26-session-6.3-flux-workflow-rebuild-design.md`) — preflight + workflow signature pattern.
- **Existing keyframes (input to S8):** `data/stories/story_2026_04_25_001/keyframes/scene_NNN/candidate_{0,1,2}.png` (16 scenes × 3 candidates, ~54 MB local-only).
