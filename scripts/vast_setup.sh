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

# IP-Adapter FaceID (for cross-scene character lock) -- not used by S6.1, deferred OK.
dl "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_flux.bin" \
   "$MODELS_DIR/ipadapter/ip-adapter-faceid_flux.bin" || \
    log "IP-Adapter FaceID Flux variant not yet hosted at expected path -- fetch manually if needed"

# ControlNet Depth (Flux variant) -- not used by S6.1, deferred OK.
dl "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors" \
   "$MODELS_DIR/controlnet/flux-depth.safetensors" || \
    log "WARN: ControlNet Depth download failed -- deferred for S6.2"

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

cat > /workspace/launch_comfyui.sh <<'EOF'
#!/usr/bin/env bash
# Launch ComfyUI listening on all interfaces so the orchestrator can hit it.
# vast.ai pytorch container = ~32GB system RAM, RTX 4090 = 24GB VRAM.
# Flux1-dev fp16 (24GB safetensors) + t5xxl_fp16 (9GB) + clip_l (~250MB)
# overflow system RAM at load time -> host OOM-killer reaps the process
# silently mid-load. --lowvram alone is not enough because ComfyUI still
# loads the t5xxl encoder fully (~9GB) before the unet load even starts.
# Stack these flags to stay safely under both RAM and VRAM caps:
#   --mmap-torch-files     load weights via mmap (no full 24GB RAM spike)
#   --fp8_e4m3fn-unet      Flux UNet stored in fp8 -> ~12GB instead of 24GB
#   --fp8_e4m3fn-text-enc  t5xxl in fp8 -> ~4.5GB instead of 9GB
#   --cpu-vae              VAE on CPU, frees ~1GB VRAM during decode
#   --lowvram              stream unet chunks to GPU rather than full load
set -euo pipefail
cd /workspace/ComfyUI
source venv/bin/activate
exec python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch \
    --mmap-torch-files \
    --fp8_e4m3fn-unet \
    --fp8_e4m3fn-text-enc \
    --cpu-vae \
    --lowvram
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

log "Setup complete."
log "Start ComfyUI:        bash /workspace/launch_comfyui.sh"
log "Start score_server:   bash /workspace/launch_score_server.sh"
log "Verify ComfyUI:       curl http://localhost:8188/system_stats"
log "Verify score_server:  curl http://localhost:8189/health"
log "Set PLATINUM_COMFYUI_HOST and PLATINUM_AESTHETICS_HOST in your local .env"
