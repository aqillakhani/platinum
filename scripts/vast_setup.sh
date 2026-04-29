#!/usr/bin/env bash
# Provision a vast.ai RTX 4090 instance with everything the platinum pipeline
# needs at runtime. Run this once on a fresh instance; weights persist on the
# attached volume. Re-running is safe (idempotent).
#
# Prerequisites on the local machine:
#   - vast.ai instance rented with RTX 4090 24GB and a persistent volume mounted at /workspace
#   - SSH access configured (ssh root@HOST -p PORT)
#   - This file rsynced or scp'd onto the instance, then run as: bash vast_setup.sh
#
# What gets installed:
#   ComfyUI + Manager
#   Flux.1 Dev weights
#   Wan 2.2 I2V weights
#   IP-Adapter FaceID
#   ControlNet Depth (Flux variant)
#   RealESRGAN (ncnn-vulkan binary)
#   Chatterbox-Turbo (Resemble AI)
#   Whisper large-v3
#
# Disk: ~60GB of weights cached to /workspace/models (the persistent volume).
# Run time on a clean instance: ~30-45 min depending on download bandwidth.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/platinum}"
MODELS_DIR="${MODELS_DIR:-/workspace/models}"
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/ComfyUI}"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }

log "Installing system packages"
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl wget unzip \
    ffmpeg \
    python3 python3-venv python3-pip \
    libsndfile1 \
    build-essential

mkdir -p "$WORKDIR" "$MODELS_DIR" "$COMFYUI_DIR/models"

# ---- Python 3.11 conda env (for platinum the package) ---------------------
#
# vast.ai's pytorch/pytorch:latest base image ships Python 3.10 but
# platinum's pyproject requires >=3.11 (StrEnum, datetime.UTC, numpy 2.x,
# matplotlib 3.10+). S6.3 Phase 2 burned ~30 min on a polyfill workaround;
# this conda env eliminates that dance. ComfyUI / score_server / Chatterbox
# / Whisper venvs are independent and stay on Python 3.10 (their deps don't
# need 3.11 and ComfyUI's torch is cu121-pinned for driver-12.2 reasons).

log "Creating Python 3.11 conda env for platinum"
if ! command -v conda >/dev/null 2>&1; then
    log "ERROR: conda not found on PATH. vast.ai's pytorch base image ships"
    log "       conda; this script assumes that. If you're on a different"
    log "       base image, install miniconda manually before re-running."
    exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx 'p311'; then
    conda create -n p311 python=3.11 -y
else
    log "p311 conda env already present (idempotent)"
fi
P311_PY=/opt/conda/envs/p311/bin/python
P311_PIP=/opt/conda/envs/p311/bin/pip

# ---- ComfyUI ---------------------------------------------------------------

if [ ! -d "$COMFYUI_DIR/.git" ]; then
    log "Cloning ComfyUI"
    # Some base images (vast.ai's pytorch image) pre-create an empty
    # /workspace/ComfyUI/models scaffold that's not a git repo. Clear it before
    # cloning -- otherwise `git clone` aborts with "destination not empty".
    rm -rf "$COMFYUI_DIR"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
else
    log "ComfyUI already present, pulling latest"
    git -C "$COMFYUI_DIR" pull --ff-only || true
fi

python3 -m venv "$COMFYUI_DIR/venv"
# shellcheck source=/dev/null
source "$COMFYUI_DIR/venv/bin/activate"
pip install --upgrade pip
# Pin torch to cu121 wheels -- vast.ai's pytorch base image ships NVIDIA driver
# 535.x (CUDA 12.2 runtime), and the default pip wheel for torch >=2.10 is cu13
# which requires driver >=12.5 and crashes ComfyUI at startup with
# "RuntimeError: NVIDIA driver on your system is too old". Pinning cu121
# (compatible with driver 12.2) keeps ComfyUI starting cleanly.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio
pip install -r "$COMFYUI_DIR/requirements.txt"
# xformers/triton wheels must match the installed torch ABI; build with the same
# index-url so they pick the cu121 variant rather than reinstalling cu13 torch.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    xformers
pip install triton                     # triton has no cu-tagged wheels

# Manager (custom node manager — useful for IP-Adapter & ControlNet nodes)
mkdir -p "$COMFYUI_DIR/custom_nodes"
if [ ! -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"
fi

# ---- Symlink models dir to the persistent volume --------------------------

# Move ComfyUI's stock models dir aside, symlink to /workspace/models so weights
# survive instance restarts.
if [ -d "$COMFYUI_DIR/models" ] && [ ! -L "$COMFYUI_DIR/models" ]; then
    log "Linking ComfyUI/models -> $MODELS_DIR"
    rm -rf "$COMFYUI_DIR/models"
    ln -s "$MODELS_DIR" "$COMFYUI_DIR/models"
fi

# ---- Model weights --------------------------------------------------------

# Flux.1 Dev (~24GB): T5 + CLIP + UNet + VAE
mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/vae" "$MODELS_DIR/clip" "$MODELS_DIR/unet" \
         "$MODELS_DIR/controlnet" "$MODELS_DIR/ipadapter" "$MODELS_DIR/upscale_models"

# Download with timeouts + retries; pass HF_TOKEN bearer header on huggingface URLs.
# Treats 0-byte target as a failed download (gated repo without auth returns 0 bytes
# while wget exits 0). Caller is responsible for cleaning the partial file before retry.
dl() {
    local url="$1"
    local dest="$2"
    if [ -s "$dest" ]; then return 0; fi
    log "Downloading $(basename "$dest")"
    local extra_headers=()
    if [[ "$url" == *"huggingface.co"* ]] && [ -n "${HF_TOKEN:-}" ]; then
        extra_headers+=(--header="Authorization: Bearer $HF_TOKEN")
    fi
    rm -f "$dest"  # avoid 0-byte stub from a previous failed run blocking retry
    wget -q --show-progress \
        --timeout=120 --tries=2 --waitretry=15 --continue \
        "${extra_headers[@]}" -O "$dest" "$url" || {
        log "WARN: wget failed for $(basename "$dest") (exit $?)"
        rm -f "$dest"
        return 1
    }
    if [ ! -s "$dest" ]; then
        log "WARN: $(basename "$dest") downloaded as 0 bytes (gated repo? bad URL?)"
        rm -f "$dest"
        return 1
    fi
    return 0
}

# Validate HF_TOKEN -- gated downloads (Flux.1 Dev) fail silently without it.
if [ -z "${HF_TOKEN:-}" ]; then
    log "WARN: HF_TOKEN not set; gated HuggingFace downloads (Flux.1 Dev, etc.) will fail."
    log "      Export it before running: HF_TOKEN=hf_... bash vast_setup.sh"
fi

# Flux Dev — main UNet (FP8 build to fit comfortably alongside Wan 2.2). GATED — needs HF_TOKEN.
dl "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
   "$MODELS_DIR/unet/flux1-dev.safetensors" || \
    log "WARN: Flux UNet download failed (HF_TOKEN missing or no license acceptance?)"
dl "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" \
   "$MODELS_DIR/vae/ae.safetensors" || \
    log "WARN: Flux VAE download failed"
dl "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
   "$MODELS_DIR/clip/clip_l.safetensors" || log "WARN: clip_l download failed"
dl "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
   "$MODELS_DIR/clip/t5xxl_fp16.safetensors" || log "WARN: t5xxl_fp16 download failed"

# S7.1.B7.1 -- Flux IP-Adapter (Redux fp8 build) for cross-scene face lock.
# References config/workflows/flux_dev_keyframe.json:12.ipadapter_file.
dl "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors" \
   "$MODELS_DIR/ipadapter/flux1-redux-dev-fp8_e4m3fn.safetensors" || \
    log "WARN: FLUX.1-Redux-dev download failed (HF_TOKEN missing or no license acceptance?)"

# S7.1.B7.1 -- Flux Depth ControlNet (used by config/workflows/flux_dev_keyframe.json:15).
dl "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors" \
   "$MODELS_DIR/controlnet/flux-controlnet-depth-v3.safetensors" || \
    log "WARN: ControlNet Depth download failed"

# S7.1.B7.1 -- Flux Canny+Pose Union ControlNet (used by config/workflows/flux_dev_keyframe.json:18).
dl "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors" \
   "$MODELS_DIR/controlnet/flux-controlnet-canny-pose-union.safetensors" || \
    log "WARN: ControlNet Pose download failed"

# S7.1.B7.1 -- DepthAnythingV2 preprocessor (used by config/workflows/pose_depth_map.json:3).
mkdir -p "$MODELS_DIR/depthanything"
dl "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
   "$MODELS_DIR/depthanything/depth_anything_v2_vitl.pth" || \
    log "WARN: DepthAnythingV2 download failed"

# S7.1.B7.1 -- DWPose preprocessor (used by config/workflows/pose_depth_map.json:2).
mkdir -p "$MODELS_DIR/dwpose"
dl "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx" \
   "$MODELS_DIR/dwpose/yolox_l.onnx" || \
    log "WARN: DWPose yolox download failed"
dl "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx" \
   "$MODELS_DIR/dwpose/dw-ll_ucoco_384.onnx" || \
    log "WARN: DWPose dw-ll download failed"

# S7.1.B7.1 -- ComfyUI extensions for IP-Adapter + ControlNet + preprocessors.
# Pinned-revision clones; later upgrades happen via Manager so the verify run
# is reproducible from a fresh box.
declare -A COMFY_EXTS=(
    [ComfyUI-IPAdapter_plus]="https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"
    [ComfyUI-Advanced-ControlNet]="https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git"
    [comfyui_controlnet_aux]="https://github.com/Fannovel16/comfyui_controlnet_aux.git"
)
for ext_name in "${!COMFY_EXTS[@]}"; do
    ext_dir="$COMFYUI_DIR/custom_nodes/$ext_name"
    if [ ! -d "$ext_dir" ]; then
        log "Cloning $ext_name"
        git clone "${COMFY_EXTS[$ext_name]}" "$ext_dir"
    else
        log "$ext_name already present (idempotent)"
    fi
    if [ -f "$ext_dir/requirements.txt" ]; then
        # Run inside ComfyUI's venv so the deps land where ComfyUI imports.
        # shellcheck source=/dev/null
        source "$COMFYUI_DIR/venv/bin/activate"
        pip install -r "$ext_dir/requirements.txt"
        deactivate
    fi
done

# Wan 2.2 I2V weights -- not used by S6.1, deferred OK.
dl "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/resolve/main/diffusion_pytorch_model.safetensors" \
   "$MODELS_DIR/checkpoints/wan22_i2v.safetensors" || \
    log "Wan 2.2 weights -- adjust URL if HuggingFace path changed (deferred to S8)"

# RealESRGAN upscaler (ncnn-vulkan binary build for speed)
if [ ! -d "$MODELS_DIR/realesrgan-ncnn-vulkan" ]; then
    log "Installing realesrgan-ncnn-vulkan"
    cd "$MODELS_DIR"
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
    unzip -q realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d realesrgan-ncnn-vulkan
    rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip
fi

# ---- Chatterbox-Turbo -----------------------------------------------------

CHATTERBOX_DIR="/workspace/chatterbox"
if [ ! -d "$CHATTERBOX_DIR" ]; then
    log "Installing Chatterbox-Turbo"
    git clone https://github.com/resemble-ai/chatterbox.git "$CHATTERBOX_DIR"
fi
python3 -m venv "$CHATTERBOX_DIR/venv"
# shellcheck source=/dev/null
source "$CHATTERBOX_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$CHATTERBOX_DIR/requirements.txt" || pip install chatterbox-tts
deactivate

# ---- Whisper large-v3 -----------------------------------------------------

WHISPER_DIR="/workspace/whisper"
mkdir -p "$WHISPER_DIR"
python3 -m venv "$WHISPER_DIR/venv"
# shellcheck source=/dev/null
source "$WHISPER_DIR/venv/bin/activate"
pip install --upgrade pip
pip install openai-whisper
# Pre-download the large-v3 weights (~3GB)
python3 - <<'PY'
import whisper
whisper.load_model("large-v3", download_root="/workspace/models/whisper")
print("Whisper large-v3 cached.")
PY
deactivate

# ---- LAION-Aesthetics v2 score server (Session 6.1) -----------------------

SCORE_DIR="/workspace/score_server"
mkdir -p "$SCORE_DIR" "$MODELS_DIR/aesthetic"

# Assumes /workspace/platinum is git-cloned (see docs/runbooks/vast-ai-keyframe-smoke.md).
if [ -d "/workspace/platinum/scripts/score_server" ]; then
    log "Copying score_server tree from /workspace/platinum"
    cp -r /workspace/platinum/scripts/score_server/. "$SCORE_DIR/"
else
    log "WARNING: /workspace/platinum/scripts/score_server not found; skipping score_server copy"
fi

if [ -f "$SCORE_DIR/requirements.txt" ]; then
    python3 -m venv "$SCORE_DIR/venv"
    # shellcheck source=/dev/null
    source "$SCORE_DIR/venv/bin/activate"
    pip install --upgrade pip
    pip install -r "$SCORE_DIR/requirements.txt"
    deactivate
fi

# LAION MLP head (~6MB)
dl "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth" \
   "$MODELS_DIR/aesthetic/sac+logos+ava1-l14-linearMSE.pth"

# ---- ComfyUI launch helper ------------------------------------------------

# Launcher script writes a small wrapper so the runbook can do a single
# `bash launch_comfyui.sh` to start the server.
#
# Box class: A6000 48GB VRAM + >=64GB system RAM (or A6000 Ada / A100 / H100).
# fp16 Flux + GPU VAE fits with margin; no FP8/lowvram/cpu-vae flags needed.
#   --mmap-torch-files     load weights via mmap (no full RAM spike at startup)
#
# Historical note: Session 6.1 ran on a 4090 32GB-RAM box with
# `--fp8_e4m3fn-unet --fp8_e4m3fn-text-enc --cpu-vae --lowvram` to fit
# Flux fp16 (24GB) + t5xxl_fp16 (9GB) under the 32GB system RAM cap. That
# combination produced perceptually degenerate output (mean RGB 0-3) for
# Cask of Amontillado scenes -- LAION-Aesthetics doesn't reject black
# images, so the score-based fallback persisted them. Session 6.2 fixes
# the root cause by dropping FP8 entirely on a properly-resourced box.
# See docs/plans/2026-04-26-session-6.2-keyframe-quality-fix-design.md.

cat > /workspace/launch_comfyui.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ComfyUI
source venv/bin/activate
exec python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --mmap-torch-files
EOF
chmod +x /workspace/launch_comfyui.sh

# ---- score_server launch helper -------------------------------------------

cat > /workspace/launch_score_server.sh <<'EOF'
#!/usr/bin/env bash
# Launch LAION-Aesthetics v2 score server. Loads CLIP+MLP into GPU at startup.
# cwd is /workspace so `score_server` resolves as a Python package and
# server.py's `from score_server.model import ...` import works the same way
# it does in tests/conftest.py (which adds scripts/ to sys.path).
set -euo pipefail
cd /workspace
source /workspace/score_server/venv/bin/activate
exec uvicorn score_server.server:app --host 0.0.0.0 --port 8189 --workers 1
EOF
chmod +x /workspace/launch_score_server.sh

# ---- Install platinum into p311 -------------------------------------------
#
# Done late so a platinum install failure (e.g., transient pypi flake)
# doesn't block the heavy ComfyUI / score-server / weights bring-up above.
# Idempotent: pip install -e updates an existing install in place.
#
# The runbook expects the user to git-clone platinum to /workspace/platinum
# BEFORE running this script (see runbook Step 3). Do not add a git-clone
# block here -- it would clone the script's own parent over itself.

if [ -d /workspace/platinum ]; then
    log "Installing platinum into p311"
    "$P311_PIP" install --upgrade pip
    "$P311_PIP" install -e /workspace/platinum
    "$P311_PY" -m platinum --help >/dev/null 2>&1 && \
        log "platinum CLI ready: $P311_PY -m platinum" || \
        log "WARN: platinum --help failed post-install; investigate"
else
    log "WARN: /workspace/platinum not present; skipping platinum install."
    log "      git-clone platinum and rerun this script (idempotent)."
fi

log "Setup complete."
log "Start ComfyUI:        bash /workspace/launch_comfyui.sh"
log "Start score_server:   bash /workspace/launch_score_server.sh"
log "Verify ComfyUI:       curl http://localhost:8188/system_stats"
log "Verify score_server:  curl http://localhost:8189/health"
log "Set PLATINUM_COMFYUI_HOST and PLATINUM_AESTHETICS_HOST in your local .env"
log "Platinum CLI:        $P311_PY -m platinum <subcommand>"
log "(or activate first:  source activate p311 && python -m platinum ...)"
