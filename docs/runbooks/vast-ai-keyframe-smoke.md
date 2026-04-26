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
  the script logs a WARN if it's missing. Also: visit
  https://huggingface.co/black-forest-labs/FLUX.1-dev once on the web to
  accept the license terms (one-time per HF account).

---

## Reference

- Design doc: `docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-design.md`
- Plan doc: `docs/plans/2026-04-25-session-6.1-keyframe-live-smoke-plan.md`
- score_server source: `scripts/score_server/`
- Provisioning script: `scripts/vast_setup.sh`
- CLI command: `src/platinum/cli.py` — `keyframes`
