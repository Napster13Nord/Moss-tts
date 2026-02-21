#!/bin/bash
# =============================================================================
# MOSS-TTS RunPod Network Storage Setup Script
# Run this ONCE after attaching your network storage volume.
# Everything is installed to /workspace so it persists across pod restarts.
# =============================================================================

set -e  # Exit on error

WORKSPACE=/workspace
VENV_DIR=$WORKSPACE/venv
REPO_DIR=$WORKSPACE/MOSS-TTS
HF_CACHE=$WORKSPACE/hf_cache
APP_DIR=$WORKSPACE/moss-tts-app

echo "=============================================="
echo " MOSS-TTS RunPod Setup"
echo " Network storage: $WORKSPACE"
echo "=============================================="

# ── 1. Create directories ─────────────────────────────────────────────────────
echo ""
echo "[1/5] Creating directories..."
mkdir -p "$HF_CACHE"
mkdir -p "$WORKSPACE/outputs"
mkdir -p "$APP_DIR"

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
else
    echo "  Venv already exists, skipping creation."
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet

# Detect CUDA version to pick the right PyTorch wheel
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "  Detected CUDA: ${CUDA_VER:-unknown}"

if [[ "$CUDA_VER" == 11.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "  Installing PyTorch for CUDA 11.x..."
elif [[ "$CUDA_VER" == 12.4* ]] || [[ "$CUDA_VER" == 12.5* ]] || [[ "$CUDA_VER" == 12.6* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "  Installing PyTorch for CUDA 12.4+..."
else
    # Default: CUDA 12.1 (most common RunPod template)
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    echo "  Installing PyTorch for CUDA 12.1 (default)..."
fi

pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --quiet
pip install transformers accelerate librosa soundfile gradio --quiet
pip install einops omegaconf pyyaml scipy datasets sentencepiece protobuf --quiet

echo "  Python packages installed."

# ── 3. Clone MOSS-TTS repository ──────────────────────────────────────────────
echo ""
echo "[3/5] Cloning MOSS-TTS repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/OpenMOSS/MOSS-TTS.git "$REPO_DIR"
    echo "  Cloned to $REPO_DIR"
else
    echo "  Repo already exists. Updating..."
    git -C "$REPO_DIR" pull --quiet
fi

# ── 4. Copy app files to workspace ────────────────────────────────────────────
echo ""
echo "[4/5] Copying app files to workspace..."

# Copy app.py if it exists next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/app.py" ]; then
    cp "$SCRIPT_DIR/app.py" "$APP_DIR/app.py"
    echo "  Copied app.py to $APP_DIR/"
fi
if [ -f "$SCRIPT_DIR/start.sh" ]; then
    cp "$SCRIPT_DIR/start.sh" "$APP_DIR/start.sh"
    chmod +x "$APP_DIR/start.sh"
    echo "  Copied start.sh to $APP_DIR/"
fi

# ── 5. Pre-download model weights to network storage ──────────────────────────
echo ""
echo "[5/5] Pre-downloading MOSS-TTS model weights (~13 GB)..."
echo "  This only happens ONCE. Weights will be cached in $HF_CACHE"
echo "  This may take 10-20 minutes depending on your connection..."

HF_HOME="$HF_CACHE" python3 - <<'PYEOF'
import os
os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/hf_cache"))

print("  Downloading processor...")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    trust_remote_code=True,
)
print("  Processor ready.")

print("  Downloading model weights (large file, please wait)...")
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print("  Model weights cached successfully.")
PYEOF

echo ""
echo "=============================================="
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Every time you start a new pod, run:"
echo "      bash $APP_DIR/start.sh"
echo "   2. Open port 7860 in RunPod to access Gradio UI"
echo "      OR use the public share link printed in the logs."
echo "=============================================="
