# vast.ai Wan 2.2 I2V smoke runbook (S8 Phase A)

Two-rental probe -> full pattern. Closure: >=14/16 Cask scenes produce
viable 5s clips on first eye-check after Rental 2.

## Prerequisites

- Local main at S8 closeout commit (workflow + Stage + CLI shipped).
- Existing keyframes for `story_2026_04_25_001` under
  `data/stories/story_2026_04_25_001/keyframes/` (16 scenes x 3 candidates).
- HF_TOKEN with FLUX.1-dev + Wan-AI gated access.
- `vastai` CLI authenticated, `~/.config/vastai/vast_api_key` on disk.
- `~/.ssh/id_ed25519` registered with vast.ai.

## Step 1: Provision A6000

```bash
vastai search offers \
  'gpu_name=RTX_A6000 cpu_ram>=64 disk_space>=80 verified=true' \
  -o 'dph_total'
vastai create instance <id> \
  --image nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04 \
  --disk 80
# wait for instance running, get ssh host:port from `vastai show instance <id>`
```

## Step 2: Run vast_setup.sh

```bash
git archive HEAD | ssh -p <port> root@<host> 'mkdir -p /workspace/platinum && tar -x -C /workspace/platinum'
ssh -p <port> root@<host> 'apt-get install -y dos2unix && dos2unix /workspace/platinum/scripts/*.sh && bash /workspace/platinum/scripts/vast_setup.sh'
# expect ~15-20 min for Flux + Wan additions
```

## Step 3: Preflight

```bash
ssh -p <port> root@<host> 'cd /workspace/platinum && python scripts/preflight_check.py --workflow config/workflows/wan22_i2v.json'
```

Expect 5 checks GREEN: HF token, ComfyUI alive, score-server alive, Wan
workflow valid, Wan weights present + extension importable. If any fail,
investigate before continuing.

## Step 4: Rental 1 (probe -- scenes 1, 8, 16)

On local:

```bash
export PLATINUM_COMFYUI_HOST=http://<host>:<comfy_port>
platinum video story_2026_04_25_001 --scenes 1,8,16
```

Expected ~15 min wall clock (3 x 5 min Wan calls + ~1 min orchestration).
Outputs at:

```
data/stories/story_2026_04_25_001/clips/scene_001_raw.mp4
data/stories/story_2026_04_25_001/clips/scene_008_raw.mp4
data/stories/story_2026_04_25_001/clips/scene_016_raw.mp4
```

Eye-check on local (open in VLC / ffplay):

- Each clip is exactly 5 seconds (gate already enforces this; eye-check is for content).
- Has visible motion -- not a still frame, not a slideshow.
- Scene content recognizable from keyframe.
- No corruption, no excessive black frames, no obvious artifacts.

**Closure target Step 4:** >=2 of 3 viable. If <2, investigate before
Rental 2:

- Wan node `class_type` drift? -- check ComfyUI logs on box for unknown
  node-class errors.
- Model load OOM? -- check `nvidia-smi` during sampling.
- Sampler params off? -- the workflow ships with `cfg=6.0`,
  `scheduler=unipc`, `steps=30` as defaults; tune if outputs are
  uniformly bad.

## Step 5: Rental 2 (full -- 16 scenes)

If Step 4 closure met:

```bash
platinum video story_2026_04_25_001
```

Expected ~80-100 min wall clock. Outputs:
`data/stories/story_2026_04_25_001/clips/scene_*_raw.mp4` (16 files).

## Step 6: Eye-check + closure

Local eye-check 16 clips. Closure: >=14/16 viable on first eye-check.
If <14, scope an S8.A.1 follow-up before moving to S9 (upscaling).

## Step 7: Calibration (if needed)

If `video_gates` fired false-positives during Rental 2 (rejected viable
clips), tighten thresholds in track YAMLs based on actual metrics from
the run. Common axes:

- `motion_min_flow` (Cask scenes are sometimes too static for 0.3 floor).
- `black_frame_max_ratio` (transition flashes can briefly cross threshold).

Re-run failed scenes only via `--scenes <csv>` after threshold update.

## Step 8: Teardown

```bash
vastai destroy instance <id>
```

S8 ends here. S9 (RealESRGAN upscale) is its own session, will rent its
own A6000 / H100.

## Gotcha checklist

- HF_TOKEN must be on box AND have Wan-AI gated access. Existing Flux
  HF token may not include Wan-AI access -- verify on HF web UI.
- `dos2unix` on `.sh` files (Windows CRLF strikes again -- S7.1 gap #11).
- Wan node `class_type` may drift; preflight workflow signature check
  catches stale on-box workflow (S6.3 lesson #8).
- A6000 has been infrastructurally flaky during heavy GPU work
  (S6.2 retro). Reboot if host hangs >30s.
- `--scenes` flag uses 1-indexed scene numbers (matching `scene.index`
  field). `--scenes 1,8,16` targets the same Cask scenes as the
  S6.3 keyframe runbook.
