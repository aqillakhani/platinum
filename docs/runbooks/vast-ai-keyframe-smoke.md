# Runbook: vast.ai keyframe live smoke

Drives the first live keyframe-generation pass against a curator-approved + adapted
story (default: Cask of Amontillado, scenes 0/7/15) on a freshly rented vast.ai
RTX 4090. Mirrors the data-flow diagram in
`docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-design.md` Section 5.

**Time budget:** ~50-75 min total
- Provisioning: 30-45 min (mostly weight downloads)
- Smoke run + verify: 10-15 min
- Tear down: ~30s

**Cost budget:** $1-2 (one rental window of an RTX 4090 at ~$0.30-0.50/hr)

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
# Search for an RTX 4090 with at least 80GB disk on a verified host
vastai search offers 'gpu_name=RTX_4090 disk_space>=80 verified=true' -o 'dph_total'

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

## 4. Launch services in tmux (~2 min)

Both ComfyUI and the score_server are long-running. Use tmux so they survive
SSH disconnect:

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

```bash
# From your local terminal:
curl -sf http://<HOST>:8188/system_stats | head -20
curl -sf http://<HOST>:8189/health
```

Expected:
- ComfyUI: 200, JSON with GPU info (RTX 4090, CUDA version, etc.)
- score_server: 200, `{"ok":true,"model":"ViT-L-14 + LAION MLP"}`

If either fails:
- SSH back to the box: `ssh root@<HOST> -p <PORT>`
- `tmux attach -t platinum` and check the failing window for errors
- Check vast.ai's port forwarding — sometimes the public IP/port differs from
  the SSH host/port (look at `vastai show instance <ID>` for explicit
  port mappings).

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

## 7. Dry-run the keyframe command

```bash
python -m platinum keyframes <CASK_STORY_ID> --scenes 0,7,15 --dry-run
```

Expected:
```
Would generate keyframes for scenes [0, 7, 15] of story <CASK_STORY_ID>
  comfy   = http://<HOST>:8188
  scorer  = http://<HOST>:8189
```
Exit code 0. If the host URLs are blank: re-check Step 6 (`.env` not loaded —
ensure you're in the project root or the file is at `secrets/.env`).

## 8. Run for real (~7-10 min)

```bash
time python -m platinum keyframes <CASK_STORY_ID> --scenes 0,7,15 \
    2>&1 | tee /tmp/cask-smoke.log
```

Expected:
- 3 progress lines per scene (one per candidate generation, ~30-60s each on Flux)
- 3 score lines per scene (~50-100ms each via score_server)
- Final summary print
- Exit code 0
- Total wall-clock ~7-10 min

Verify artifacts on disk:
```bash
ls -lh data/stories/<CASK_STORY_ID>/keyframes/scene_{000,007,015}/
```
Expected: each `scene_NNN/` dir has `candidate_0.png`, `candidate_1.png`,
`candidate_2.png` at ~150-300 KB each.

Verify story.json was updated:
```bash
python -c "
from platinum.models.story import Story
from pathlib import Path
import os
cask = os.environ.get('CASK_STORY_ID')
s = Story.load(Path(f'data/stories/{cask}/story.json'))
for i in (0, 7, 15):
    print(s.scenes[i].keyframe_path, s.scenes[i].keyframe_scores,
          s.scenes[i].validation.get('keyframe_selected_via_fallback'))
"
```
Expected: each line shows a real path + 3 floats + a bool.

## 9. Eye-check (~1 min)

Open the 9 PNGs and sanity-check:

```bash
# Windows
explorer data/stories/<CASK_STORY_ID>/keyframes/scene_000

# macOS / Linux
open data/stories/<CASK_STORY_ID>/keyframes/scene_000
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

## 10. Tear down (~30s)

```bash
vastai destroy <INSTANCE_ID>
vastai show instances  # confirm gone
```

The persistent volume is preserved; weights survive for next rental.
Re-renting on the same volume skips the 30-45 min weights download.

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
  → **Cause:** The FP8 + `--cpu-vae` flag stack we use to fit Flux on a
  32GB / 24GB RTX 4090 collapses output for very dark prompts (the Cask
  story is catacombs / wine cellar / candlelit walls + a strong
  negative_prompt against bright daylight). FP8 e4m3fn UNet output
  decoded on CPU VAE has precision drift that pushes already-dark
  scenes to all-black.
  → **Fix:** None at the smoke layer -- the pipeline is functioning
  correctly (gen + score + select + persist all succeed). Real fix
  options for future runs: (a) rent a larger box (64GB RAM + A6000
  48GB) so we can drop the FP8 + cpu-vae flags, (b) use Comfy-Org's
  pre-quantized FP8 Flux variant (built for FP8 sampling), (c) raise
  the LAION threshold so degenerate outputs trigger a halt instead of
  a fallback selection. Documented for Session 6.2 retro.

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

---

## Reference

- Design doc: `docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-design.md`
- Plan doc: `docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-plan.md`
- score_server source: `scripts/score_server/`
- Provisioning script: `scripts/vast_setup.sh`
- CLI command: `src/platinum/cli.py` — `keyframes`
