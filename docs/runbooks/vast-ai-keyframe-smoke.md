# Runbook: vast.ai keyframe live smoke

Drives a live keyframe-generation pass against a curator-approved + adapted
story (default: Cask of Amontillado, scenes 1/8/16) on a freshly rented vast.ai
A6000 48GB box (or A6000 Ada / A100 / H100 — anything >=48GB VRAM AND >=64GB
system RAM). Mirrors the data-flow diagram in
`docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-design.md` Section 5;
S6.3 workflow rebuild details in
`docs/plans/2026-04-26-session-6.3-flux-workflow-rebuild-design.md`. The S6.3
rebuild added `FluxGuidance` + `ModelSamplingFlux` nodes to produce detailed
subjects (fixing the S6.2 mood-only-render issue).

**Time budget:** ~50-75 min total
- Provisioning: 30-45 min (mostly weight downloads)
- Smoke run + verify: 10-15 min
- Tear down: ~30s (or skip when chaining into Session 7)

**Cost budget:** $1.50-2.50 (one rental window of an A6000 at ~$0.60-0.90/hr)

---

## 1. Prerequisites

One-time setup on your local machine and vast.ai account:

- [ ] vast.ai account with at least $5 credit
- [ ] Public SSH key registered with vast.ai (Account → SSH Keys)
- [ ] vastai CLI installed locally:
  ```
  pip install vastai
  vastai set api-key <YOUR_VAST_AI_KEY>
  ```
- [ ] platinum repo pushed to a public GitHub URL (so the box can `git clone` it).
  - If your platinum repo is currently private, either make it public before
    proceeding, or push to a public mirror, or configure an SSH deploy key on
    the box. The runbook below assumes public HTTPS clone.
- [ ] A curator-approved + `platinum adapt`-completed story on local disk.
  Typically Cask of Amontillado from the Session 4 smoke; verify with
  `python -m platinum status --story <CASK_ID>` — `visual_prompts` should
  show COMPLETE.
- [ ] Local `.env` does NOT yet have `PLATINUM_*_HOST` values set (we add
  them in Step 6).

## 2. Pick + rent the box (~3 min)

```bash
# A6000 48GB VRAM + >=64GB system RAM (drops FP8 collapse risk; see S6.2 design).
# Acceptable alternates if no A6000 offers: A6000 Ada -> A100 40GB -> A100 80GB -> H100.
vastai search offers 'gpu_name=RTX_A6000 cpu_ram>=64 disk_space>=80 verified=true' -o 'dph_total'

# Pick the cheapest verified offer (top of the sorted list)
# Note the OFFER_ID, then create the instance:
vastai create instance <OFFER_ID> --image pytorch/pytorch:latest --disk 80 --ssh

# Verify the instance came up; note its SSH host + port
vastai show instances
```

Capture two values for later:
- `<HOST>` — e.g. `ssh4.vast.ai` or a public IP
- `<PORT>` — the SSH port assigned by vast.ai

Verify SSH works:
```bash
ssh root@<HOST> -p <PORT> 'hostname && nvidia-smi | head -5'
```
Expected: connects without password prompt; prints `Tesla RTX 4090` info.

## 3. Provision the box (~30-45 min, mostly downloads)

SSH into the box and clone the platinum repo + run setup:

```bash
ssh root@<HOST> -p <PORT>

# Inside the box:
git clone https://github.com/<your-user>/platinum /workspace/platinum
bash /workspace/platinum/scripts/vast_setup.sh 2>&1 | tee /workspace/setup.log
```

Watch the log; `vast_setup.sh` is idempotent (re-runnable on the same persistent
volume — weights skip re-download). Expect ~30-45 min on a clean instance.

When complete, the script prints:
```
[setup] Setup complete.
[setup] Start ComfyUI:        bash /workspace/launch_comfyui.sh
[setup] Start score_server:   bash /workspace/launch_score_server.sh
[setup] Verify ComfyUI:       curl http://localhost:8188/system_stats
[setup] Verify score_server:  curl http://localhost:8189/health
[setup] Set PLATINUM_COMFYUI_HOST and PLATINUM_AESTHETICS_HOST in your local .env
```

Verify weights landed:
```bash
ls -lh /workspace/models/unet/flux1-dev.safetensors \
        /workspace/models/aesthetic/sac+logos+ava1-l14-linearMSE.pth
```
Expected: `flux1-dev.safetensors` ~24GB, MLP head ~6MB.

### Pre-flight check (~30s)

After `vast_setup.sh` completes, run the pre-flight script to verify the box
is fully ready before any GPU work:

```bash
ssh root@<HOST> -p <PORT> \
  'export HF_TOKEN=hf_... \
          PLATINUM_COMFYUI_HOST=http://localhost:8188 \
          PLATINUM_AESTHETICS_HOST=http://localhost:8189; \
   cd /workspace/platinum && /opt/conda/envs/p311/bin/python scripts/preflight_check.py'
```

Expected: 5 OK lines + workflow signature + "preflight OK". If any check
FAILs, fix it before proceeding (re-run vast_setup.sh, accept HF license,
restart ComfyUI/score_server, etc.). Pre-flight catches the "whoami passes
but token lacks gated access" gotcha and the "stale workflow on box" gotcha.

## 4. Launch services in tmux (~2 min)

Both ComfyUI and the score_server are long-running. Use tmux so they survive
SSH disconnect:

After provisioning, set this once for use in subsequent steps (it points
at the conda env's Python binary; using it directly avoids `conda activate`
shell-init fragility under SSH):

```bash
PLATINUM_PY=/opt/conda/envs/p311/bin/python
```

Still inside the box:

```bash
# Still inside the box:
tmux new -s platinum

# Window 0: ComfyUI
bash /workspace/launch_comfyui.sh
# Wait for: "To see the GUI go to: http://0.0.0.0:8188"

# New tmux window: Ctrl-b c
bash /workspace/launch_score_server.sh
# Wait for: "Application startup complete." + "Uvicorn running on http://0.0.0.0:8189"

# Detach from tmux: Ctrl-b d
# (services keep running)
exit  # close the SSH session
```

## 5. Verify endpoints from local (~30s)

Step 3's pre-flight already verified ComfyUI + score-server alive on the box
itself. From your local terminal, just confirm the SSH tunnel is forwarding:

```bash
curl -sf http://<HOST>:8188/system_stats | head -5
curl -sf http://<HOST>:8189/health
```

Both should return 200. If either fails, the issue is the SSH tunnel/proxy,
not the services themselves (since pre-flight succeeded on the box). See
Step 12's "vast.ai SSH proxy resets" entry for the self-healing tunnel script.

## 6. Wire platinum to the box

Edit your local `.env` (in the platinum project root or `secrets/.env`):

```bash
PLATINUM_COMFYUI_HOST=http://<HOST>:8188
PLATINUM_AESTHETICS_HOST=http://<HOST>:8189
```

Verify config loads:
```bash
python -c "from platinum.config import Config; c = Config(); print(c.settings['comfyui']['host'], c.settings['aesthetics']['host'])"
```
Expected: prints both URLs.

## 7. Run workflow probes (S6.3 first run only)

S6.3 added FluxGuidance + ModelSamplingFlux nodes to the workflow. Before
the full Cask validation in Step 8, run two diagnostic probes to confirm
the rebuild produces detailed subjects (the S6.2 issue was mood-only renders
with no recognizable subjects).

### 7a. Probe A: cfg=1.0 alone (~3 min)

This probe temporarily removes the new nodes to isolate whether `cfg=1.0`
alone fixes the issue. If it does, the workflow rebuild was over-engineered
and we can ship a simpler config.

On the box, edit the workflow JSON to point KSampler.model back at node 1
and KSampler.positive back at node 3 (skipping nodes 10 and 11). Keep cfg=1.0.

```bash
# (One-line sed or inline Python -- the implementer chose at runtime)
ssh root@<HOST> -p <PORT> 'cd /workspace/platinum && \
  python scripts/keyframe_quality_smoke.py \
    --prompt "<Tuscan vineyard prompt>" --label tuscan-A \
    --output-dir /tmp/smoke-A
  python scripts/keyframe_quality_smoke.py \
    --prompt "<Cask scene 1 chiaroscuro prompt>" --label cask1-A \
    --output-dir /tmp/smoke-A'
```

scp the 6 PNGs locally and eye-check:
- Tuscan: vineyard rendered? cypress trees? farmhouse? Or yellow/dark blob?
- Cask 1: nobleman + goblet + furniture? Or amber smudge in black?

### 7b. Probe B: full reference (only if 7a inadequate)

If 7a's outputs are mood-only blobs, restore Phase 1's full reference workflow:

```bash
ssh root@<HOST> -p <PORT> 'cd /workspace/platinum && \
  git checkout config/workflows/flux_dev_keyframe.json && \
  python scripts/keyframe_quality_smoke.py \
    --prompt "<Tuscan>" --label tuscan-B --output-dir /tmp/smoke-B
  python scripts/keyframe_quality_smoke.py \
    --prompt "<Cask 1>" --label cask1-B --output-dir /tmp/smoke-B'
```

Eye-check 6 more PNGs. If 7b STILL produces mood-only output, the workflow
hypothesis is wrong; halt session and open a new diagnostic ladder (likely
visual_prompts revision, or escape to flux_pro_api).

**Decision:** ship the FULL REFERENCE workflow regardless of which probe wins
the eye-check. Probe A is a diagnostic, not a config we'd ship — "cfg=1.0
with no FluxGuidance" gives Flux zero guidance signal.

## 8. Cask 1/8/16 validation + threshold calibration

### 8a. Reset + Cask validation (~6 min)

Phase 2 must re-run scenes 1, 8, 16 against the rebuilt workflow. The
orchestrator skips scenes whose `keyframe_path` is already set, so we
need to clear stale state from S6.2 Phase 2:

```bash
# scp the local Cask story.json to the box (avoids re-paying Claude $1
# to re-run platinum adapt; visual_prompts are deterministic-enough).
scp -P <PORT> \
    data/stories/<CASK_STORY_ID>/story.json \
    root@<HOST>:/workspace/platinum/data/stories/<CASK_STORY_ID>/story.json

# Reset stale keyframe state.
ssh root@<HOST> -p <PORT> \
    'cd /workspace/platinum && \
     python scripts/reset_scene_keyframes.py <CASK_STORY_ID> \
       --scenes 1,8,16 --delete-files'

# Run the actual keyframe stage.
time $PLATINUM_PY -m platinum keyframes <CASK_STORY_ID> --scenes 1,8,16 \
    2>&1 | tee /tmp/cask-smoke.log
```

Expected: 9 PNGs (3 candidates x 3 scenes), all passing brightness AND
subject gates, LAION 5.5-7+, ~0-2 fallbacks total, exit 0. Wall-clock
~6 min on A6000.

### 8b. Threshold calibration

Print per-candidate edge densities to tune `subject_min_edge_density` for
atmospheric_horror:

```bash
python -c "
from pathlib import Path
from platinum.utils.validate import check_image_subject
for p in sorted(Path('data/stories/<CASK_STORY_ID>/keyframes').glob('scene_*/candidate_*.png')):
    r = check_image_subject(p, min_edge_density=0.0)
    print(f'{p}  edge_density={r.metric:.4f}')
"
```

If a viable Cask 1 scores edge_density 0.015 (below the 0.020 floor), drop
the floor for atmospheric_horror to 0.012. If a non-viable candidate scores
0.025, raise to 0.030. Edit `config/tracks/atmospheric_horror.yaml` locally
and commit.

## 9. Eye-check (~1 min)

Open the 9 Cask PNGs (3 scenes x 3 candidates from Step 8a) and sanity-check:

```bash
# Windows
explorer data/stories/<CASK_STORY_ID>/keyframes/scene_001

# macOS / Linux
open data/stories/<CASK_STORY_ID>/keyframes/scene_001
```

Look for:
- **On-theme**: vaguely matches scene prompts (wine cellar / catacombs /
  chained skeleton for Cask). If wildly off, the visual_prompts from
  Session 4's adapt may need re-running with adjusted templates.
- **Distinct candidates**: the 3 candidates per scene are visibly different
  compositions (different seeds → different crops, lighting, framing).
  If they look identical, deterministic seeding may not be wired — check
  `_seeds_for_scene` in `keyframe_generator.py`.
- **Selected vs rejected**: the candidate marked "selected" in the table
  from Step 8 should look subjectively better than the rejected ones.
  If the selection seems random, the LAION model may not be discriminating
  well — record the example for future tuning.
- **Closure criterion (≥2 of 3 viable):** if at least 2 of the 3 Cask scenes
  (1, 8, 16) show recognizable subjects on eye-check, ship S6.3. If 0 or 1
  are viable, S6.3 doesn't close — escalate to S6.4 (visual_prompts revision).

## 10. Tear down

S6.3 closes with explicit teardown. S7 spins a fresh box.

```bash
vastai destroy instance <ID>
```

Verify all rentals are cleaned up:
```bash
vastai show instances
```

Both rentals (Probe rental from Step 7 and Cask validation rental from Step 8)
should be destroyed. S7 begins from a clean slate to limit hang exposure
across rentals.

## 11. Commit smoke artifacts

**DO NOT** commit:
- `data/stories/*` — regeneratable workspace data (per project memory)
- `.env` — contains live host URLs

**DO** commit:
- `docs/runbooks/vast-ai-keyframe-smoke.md` — backfill the troubleshooting
  section (Step 12) with anti-patterns observed during this run
- Any code fixes that came out of the smoke (treat as their own commits with
  `fix(...)` messages, NOT lumped with the runbook)

```bash
git status
# CONFIRM nothing under data/stories/ is staged
# CONFIRM .env is NOT staged
git add docs/runbooks/vast-ai-keyframe-smoke.md
git commit -m "docs(runbooks): vast-ai-keyframe-smoke -- backfill troubleshooting from first live run"
```

Also remove the live `PLATINUM_*_HOST` values from your local `.env`
(or comment them out) so subsequent local runs don't try to reach the
torn-down box.

## 12. Troubleshooting

(This section is populated DURING the live smoke as anti-patterns are
encountered. Each entry: **Symptom** → **Fix**. Adding bullets here is the
high-value artifact future-you will appreciate when re-running this runbook
months later for Sessions 8/9/10/13.)

- **Symptom:** `vast_setup.sh` aborts at "Cloning ComfyUI" with
  `fatal: destination path '/workspace/ComfyUI' already exists and is not an empty directory.`
  → **Fix:** vast.ai's `pytorch/pytorch:latest` base image pre-creates an
  empty `/workspace/ComfyUI/models` scaffold that confuses git clone.
  Resolved permanently in `scripts/vast_setup.sh` (commit `3475c5a`):
  the script now `rm -rf "$COMFYUI_DIR"` before cloning when no `.git`
  subdir is present. If you hit this on an old script revision, manually
  `rm -rf /workspace/ComfyUI && bash /workspace/platinum/scripts/vast_setup.sh`.

- **Symptom:** `vast_setup.sh` exits silently after "Downloading
  flux1-dev.safetensors" with a 0-byte file at
  `/workspace/models/unet/flux1-dev.safetensors` and no error message.
  → **Cause:** Flux.1 Dev is a **gated** HuggingFace repo. Without an
  `HF_TOKEN` Authorization header, HF returns a 401 redirect that wget
  doesn't surface as an error -- it writes a 0-byte file and exits 0.
  → **Fix:** Resolved in `scripts/vast_setup.sh`: the `dl()` function
  now (a) auto-attaches `HF_TOKEN` as a Bearer header for `huggingface.co`
  URLs, (b) treats 0-byte downloads as failure with `rm -f` cleanup,
  (c) uses `--timeout=120 --tries=2` so future hangs fail in 2 min, not
  forever. **Before running `vast_setup.sh`, export `HF_TOKEN=hf_...`** --
  the script logs a WARN if it's missing.

- **Symptom:** `HF_TOKEN` is exported, but `vast_setup.sh` still logs
  `WARN: Flux UNet download failed (HF_TOKEN missing or no license acceptance?)`
  and produces a 0-byte `flux1-dev.safetensors`. Verify with:
  `curl -sI -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors`
  → if you see `HTTP/1.1 401 Unauthorized` and the body contains "Access
  to model black-forest-labs/FLUX.1-dev is restricted", the token is fine
  but the **HF account has not accepted the model license**.
  → **Cause:** Gated HF repos require BOTH a valid token AND a one-time
  license click in the web UI per HF account. The token alone reaches the
  metadata endpoint (`/api/models/...` returns 200) but the actual file URL
  (`/resolve/main/...`) returns 401 until the license is accepted.
  → **Fix:** Visit https://huggingface.co/black-forest-labs/FLUX.1-dev
  in a browser logged in as the token owner, click **Agree and access
  repository**, then re-run the two failed downloads:
  ```
  HF_TOKEN=hf_... wget --header="Authorization: Bearer $HF_TOKEN" \
    -O /workspace/models/unet/flux1-dev.safetensors \
    https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
  HF_TOKEN=hf_... wget --header="Authorization: Bearer $HF_TOKEN" \
    -O /workspace/models/vae/ae.safetensors \
    https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors
  ```
  No need to re-run the full `vast_setup.sh`.

- **Symptom:** `bash /workspace/launch_comfyui.sh` exits with
  `RuntimeError: The NVIDIA driver on your system is too old (found version 12020). Please update your GPU driver`.
  → **Cause:** vast.ai's `pytorch/pytorch:latest` base image ships with
  NVIDIA driver 535.154.05 (CUDA runtime 12.2). The default pip wheel for
  `torch >= 2.10` is the **cu13** variant, which requires driver ≥ 12.5.
  ComfyUI's `pip install -r requirements.txt` pulls latest torch by default,
  so the venv ends up with `torch 2.11.0+cu130` and crashes at startup
  when `_lazy_init` calls `_cuda_init`.
  → **Fix:** Resolved in `scripts/vast_setup.sh` by pinning the ComfyUI
  venv's torch to **cu121 wheels** (compatible with driver 12.2) before
  the requirements.txt install:
  ```
  pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio
  pip install -r "$COMFYUI_DIR/requirements.txt"
  pip install --index-url https://download.pytorch.org/whl/cu121 xformers
  pip install triton
  ```
  If you hit this on an old script revision, repair the existing venv:
  ```
  source /workspace/ComfyUI/venv/bin/activate
  pip install --index-url https://download.pytorch.org/whl/cu121 \
      --upgrade --force-reinstall torch torchvision torchaudio
  pip install --index-url https://download.pytorch.org/whl/cu121 \
      --upgrade --force-reinstall xformers
  ```

- **Symptom:** ComfyUI gets killed silently mid-Flux-load, log ends at
  "Requested to load Flux" with no traceback. `nvidia-smi` shows 1MB GPU
  used after; `pgrep -af "python main.py"` returns nothing.
  → **Cause:** Container OOM. RTX 4090 has 24GB VRAM and the vast.ai
  pytorch image gives the container ~32GB system RAM. Flux1-dev fp16
  (24GB safetensors) + t5xxl_fp16 (9GB) + clip_l (~250MB) overflow the
  RAM at load -- the host OOM-killer reaps the process before `set -e`
  can react and tmux's per-window `command exits -> close` reaps the
  pane too, hiding the death.
  → **Fix:** Resolved in `scripts/vast_setup.sh`'s launcher: stack
  `--mmap-torch-files --fp8_e4m3fn-unet --fp8_e4m3fn-text-enc --cpu-vae --lowvram`.
  Brings Flux in at ~12GB and t5xxl at ~4.5GB, well under the 32GB cap.
  Trade-off: image quality drops on FP8 (see next entry).

- **Symptom:** End-to-end smoke completes and `story.json` shows
  `keyframe_path` populated for every selected scene, but every
  `candidate_*.png` is essentially black (mean RGB ~0-3 out of 255 per
  channel; LAION still scores them ~3.9-4.6 because the MLP head
  doesn't penalise low-content imagery hard enough to halt selection).
  [ROOT CAUSE WAS WORKFLOW, NOT FP8 -- FIXED IN S6.3 commit <find via git log>; preserved as cautionary tale]
  → **Cause:** S6.2 initially hypothesized FP8 quantization, but S6.2 Phase 2
  reproduced the same black-PNG output on A6000 fp16 (no FP8). The true root
  cause was the missing `FluxGuidance` + `ModelSamplingFlux` nodes in the
  Flux workflow. Without those nodes, Flux produces mood-only renders with
  zero subject detail for dark scenes like Cask catacombs.
  → **Fix:** S6.3 rebuild added `FluxGuidance` + `ModelSamplingFlux` nodes
  with `KSampler.cfg=1.0`. This forces Flux to render detailed subjects.
  The FP8 + `--cpu-vae` flag stack was never the problem -- merely a red
  herring chased by misdiagnosis. The "Fix options" list below is now
  historical, kept as a reminder of the diagnostic blind alley.

- **Symptom:** `--scenes 0,7,15` run starts, processes scene index 7,
  then halts. Scene index 0 never appears in the log; scene index 15
  never starts because of the halt.
  → **Cause:** The `scene_breakdown` stage emits 1-indexed `scene.index`
  values (1..N), not 0-indexed. The CLI's old `--scenes` validator
  treated values as array offsets (0..N-1) but the keyframe generator
  filters by direct match against `scene.index`. So `0` matched no
  scene (silently), `7` matched scene 7, and the runbook examples for
  "start/middle/end of a 16-scene story" should be `1, 8, 16` not
  `0, 7, 15`.
  → **Fix:** Resolved in `src/platinum/cli.py`: `--scenes` is now
  validated against `{scene.index for scene in s.scenes}` (the actual
  indices the story uses) rather than `range(len(s.scenes))`. Use
  `--scenes 1,8,16` for a 16-scene 1-indexed story.

- **Symptom:** ComfyUI returns `400 Bad Request` for every workflow
  submission with log entry
  `invalid prompt: missing_node_type Node 'ID #_meta'`.
  → **Cause:** `utils/workflow.py` stores a top-level `_meta.role`
  block as part of every loaded workflow JSON for role-name lookup
  ("which node id is the positive prompt", etc.). Posting that block
  intact to ComfyUI's `/prompt` endpoint trips ComfyUI's "every
  top-level key is a node, missing class_type means corrupt" check.
  → **Fix:** Resolved in `src/platinum/utils/comfyui.py`:
  `HttpComfyClient._submit` strips any `_meta` key from the workflow
  dict before serialising to the request body. Regression test:
  `tests/unit/test_comfyui.py::test_http_comfy_client_strips_meta_from_prompt_payload`.

- **Symptom:** `bash launch_score_server.sh` starts cleanly, `/health`
  returns 200, but `POST /score` returns 500 with
  `ModuleNotFoundError: No module named 'score_server'` in the server log.
  → **Cause:** `server.py`'s lazy import is
  `from score_server.model import score_image_bytes` (package-style),
  which works locally because `tests/conftest.py` adds `scripts/` to
  `sys.path`. But the launcher previously did `cd /workspace/score_server`
  before `uvicorn server:app`, putting the package directory itself on
  `sys.path` so `model` is importable but `score_server.model` isn't.
  → **Fix:** Resolved in `scripts/vast_setup.sh`'s launcher: cwd is now
  `/workspace` (parent of the package), and the entrypoint is
  `score_server.server:app`. Both the local test path and the box path
  now resolve `from score_server.model import ...` consistently.

- **Symptom:** A live keyframe run works through Flux generation and
  scoring, then halts with
  `ImportError: Matplotlib requires dateutil>=2.7; you have 2.6.1`
  or `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
  → **Cause:** `check_hand_anomalies` (the keyframe anatomy gate) calls
  MediaPipe Hands. MediaPipe imports matplotlib at init, and on Python
  3.14 the new mediapipe wheel restructured the `solutions` namespace.
  An old local dateutil also blocks matplotlib's import.
  → **Fix:** Resolved in `src/platinum/utils/validate.py`:
  `check_hand_anomalies` now catches `ImportError`/`AttributeError`
  from the factory call and returns a passing `CheckResult` with reason
  `"skipped: hand detector unavailable"` rather than halting the
  pipeline. Tooling drift (Python upgrade, mediapipe major bump,
  dateutil age) no longer blocks keyframe generation -- the score-based
  selection stays correct.

- **Symptom:** Keyframe run dies mid-flight with
  `RemoteProtocolError: Server disconnected without sending a response`
  on candidate generation, but ComfyUI is still alive on the box and
  `curl localhost:8188/system_stats` from the box succeeds.
  → **Cause:** vast.ai's SSH proxy (`ssh<N>.vast.ai`) periodically
  RST-resets connections that have been idle a few minutes -- including
  long-lived port-forwards opened with `ssh -L`. A single `ssh -L` will
  silently die after a hiccup and your local `curl` returns
  `connection refused` even though the box is healthy.
  → **Fix:** Run a self-healing wrapper instead of a single ssh:
  ```bash
  cat > /tmp/tunnel.sh <<'EOF'
  #!/usr/bin/env bash
  while true; do
    ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
        -i ~/.ssh/id_ed25519 -p <PORT> -N \
        -L 8188:localhost:8188 -L 8189:localhost:8189 \
        root@<HOST>
    echo "[tunnel] dropped at $(date), retry in 5s"
    sleep 5
  done
  EOF
  bash /tmp/tunnel.sh &
  ```
  Verifies with `curl localhost:8188/system_stats && curl localhost:8189/health`
  before kicking off the keyframe run.

- **Symptom:** vast.ai's `pytorch/pytorch:latest` base image ships Python 3.10.
  `pip install -e .` fails with `meson-python: error: package requires
  Python >=3.11` (numpy 2.4+ build requirement), and after fixing that,
  imports fail with `ImportError: cannot import name 'StrEnum' from 'enum'`
  and `cannot import name 'UTC' from 'datetime'` (both 3.11-only).
  → **Cause:** Mismatch between platinum's `requires-python = ">=3.11"` and
  the box's Python 3.10. Even with `--ignore-requires-python`, transitive
  deps (numpy 2.x, contourpy 1.4+, matplotlib 3.10+) demand 3.11.
  → **Fix (workaround used in S6.3 Phase 2):** Lean install with version
  constraints + sitecustomize polyfill for the two stdlib gaps:
  ```bash
  pip install --ignore-requires-python "numpy<2" "opencv-python<4.11" \
      "pillow<11" "httpx>=0.27" "pyyaml>=6" "python-dotenv>=1" \
      "typer>=0.12" "jinja2>=3.1" "rich>=13.7" "sqlalchemy>=2.0" \
      "aiosqlite>=0.20" "alembic>=1.13" "anthropic>=0.40"
  pip install -e . --no-deps --ignore-requires-python
  SP=$(python -c 'import site; print(site.getsitepackages()[0])')
  cat > "$SP/_strenum_polyfill.py" <<'EOF'
  import enum, datetime as _dt
  if not hasattr(enum, "StrEnum"):
      class StrEnum(str, enum.Enum): pass
      enum.StrEnum = StrEnum
  if not hasattr(_dt, "UTC"):
      _dt.UTC = _dt.timezone.utc
  EOF
  echo "import _strenum_polyfill" > "$SP/sitecustomize.py"
  ```
  → **Proper fix (S6.4 — RESOLVED):** `vast_setup.sh` now provisions a `p311`
  conda env via `conda create -n p311 python=3.11 -y` and installs platinum
  into it. All subsequent platinum invocations use `/opt/conda/envs/p311/bin/python`.
  The polyfill workaround above is preserved as a cautionary tale of how an
  ergonomically-bad workaround (sitecustomize.py + version-pinned deps +
  --ignore-requires-python) accumulates if the right fix gets deferred.

- **Symptom:** `python -m platinum keyframes <story> --scenes X,Y` exits in
  <5s with "Keyframes complete" but generates zero PNGs. story.json shows
  `keyframe_path: None` for the requested scenes after.
  → **Cause:** Orchestrator's `Stage.is_complete()` checks
  `story.stages` (in-memory list of StageRun records, persisted to story.json)
  for the latest entry matching the stage name. A prior COMPLETE run leaves
  that record behind. `reset_scene_keyframes.py` clears per-scene
  `keyframe_path` + candidate files but does NOT strip the StageRun history,
  so the next invocation's orchestrator sees "keyframe_generator already
  COMPLETE, skip stage entirely" and never enters `generate()`.
  → **Fix (manual):** Before rerunning, strip the stage entries:
  ```bash
  python -c "
  import json
  from pathlib import Path
  p = Path('data/stories/<id>/story.json')
  d = json.loads(p.read_text())
  d['stages'] = [s for s in d['stages']
                 if s.get('stage') != 'keyframe_generator']
  p.write_text(json.dumps(d, indent=2))
  "
  ```
  Note: deleting `data/platinum.db` is NOT enough — the resume state lives
  in story.json, not the SQLite projection. → **Proper fix (S6.4):**
  `reset_scene_keyframes.py` should accept `--reset-stage` to strip
  matching `keyframe_generator` StageRuns from story.json. Test:
  rerun-after-reset produces fresh PNGs.

- **Symptom:** `image_model.candidates_per_scene: 6` set in
  `config/tracks/atmospheric_horror.yaml`, but `platinum keyframes` still
  generates exactly 3 candidates per scene.
  → **Cause:** `generate()` in `keyframe_generator.py` calls
  `generate_for_scene(scene, ...)` without passing `n_candidates`. The
  function's default (`n_candidates: int = 3`) wins. The yaml setting is
  read for documentation but never forwarded.
  → **Fix (S6.4):** In `generate()` (or `KeyframeGeneratorStage.run`),
  read `track_cfg["image_model"]["candidates_per_scene"]` and forward
  to `generate_for_scene(n_candidates=N)`. Add integration test asserting
  a yaml override of 5 produces 5 candidate PNGs on disk.

- **Symptom:** Cask scene 1 (Venetian nobleman portrait, single oil lamp)
  consistently produces output below the brightness floor (mean_rgb 2.6-3.0)
  regardless of FluxGuidance + ModelSamplingFlux + sampler tweaks. Subjects
  ARE rendered (a face IS in the latent) but invisibly dim — unviewable as
  shippable keyframes.
  → **Cause:** The visual_prompt for scene 1 is over-loaded with darkness
  modifiers — "dark velvet doublet, candlelit study at night, single oil
  lamp casting deep amber glow across half his face, the other half lost in
  shadow, deep oxblood velvet drapery, vaulted dark oak ceiling". Flux
  faithfully renders the prompt; the prompt itself describes a
  near-imperceptible image. Negative prompt also fights brightness ("bright
  daylight", "multiple light sources").
  → **Fix (S6.4):** Re-run `platinum adapt` with a revised visual_prompts
  template that caps darkness-modifier density (rule of thumb: at most one
  explicit darkness cue per clause, primary subjects described in lit
  terms — e.g. "amber-lit face" rather than "half-lit, half lost in
  shadow"). Test with Cask scene 1 specifically. Catacomb scenes (8 + 16)
  with torchlight + bone walls + ambient amber render cleanly without this
  treatment.

- **Symptom (positive lesson, not a failure):** Bumping KSampler
  `steps: 30 → 60` materially improves Flux Dev output quality on
  chiaroscuro scenes — less "AI slop" texture, sharper subject definition,
  cleaner figure rendering. Selected candidates' LAION scores moved from
  ~5.7-6.2 to 6.2-6.5 in S6.3 Phase 2 Cask 8/16 testing.
  → **Decision (shipped in S6.3 Phase 2):** `steps: 60` is the default in
  `flux_dev_keyframe.json`. Generation time per candidate ~50s on A6000
  (was ~25s), still within budget for 3 candidates × 16 scenes ≈ 40 min
  per Cask story.

---

## Reference

- S6.1 design doc: `docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-design.md`
- S6.1 plan doc: `docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-plan.md`
- S6.3 design doc: `docs/plans/2026-04-26-session-6.3-flux-workflow-rebuild-design.md`
- S6.3 plan doc: `docs/plans/2026-04-26-session-6.3-flux-workflow-rebuild-plan.md`
- Rebuilt Flux Dev workflow: `config/workflows/flux_dev_keyframe.json`
- score_server source: `scripts/score_server/`
- Provisioning script: `scripts/vast_setup.sh`
- Pre-flight script: `scripts/preflight_check.py`
- Reset script: `scripts/reset_scene_keyframes.py`
- CLI command: `src/platinum/cli.py` — `keyframes`
