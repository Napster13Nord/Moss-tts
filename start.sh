#!/bin/bash
# =============================================================================
# MOSS-TTS RunPod Quick-Start Script
# Run this EVERY TIME you start a new pod (after setup.sh has been run once).
# Takes ~30 seconds to start (no reinstalling needed).
# =============================================================================

WORKSPACE=/workspace
VENV_DIR=$WORKSPACE/venv
HF_CACHE=$WORKSPACE/hf_cache
APP_DIR=$WORKSPACE/moss-tts-app

# Check that setup has been done
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please run setup.sh first!"
    exit 1
fi

if [ ! -d "$HF_CACHE" ]; then
    echo "ERROR: Model cache not found at $HF_CACHE"
    echo "Please run setup.sh first!"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Point HuggingFace to network storage cache (no re-download needed)
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"

# Output directory on network storage
export MOSS_OUTPUT_DIR="$WORKSPACE/outputs"
mkdir -p "$MOSS_OUTPUT_DIR"

echo "=============================================="
echo " MOSS-TTS Starting..."
echo " Model cache: $HF_HOME"
echo " Outputs: $MOSS_OUTPUT_DIR"
echo " Gradio port: 7860"
echo "=============================================="
echo ""
echo " Open RunPod's port 7860 to access the UI."
echo " Or use the public Gradio share link below."
echo "=============================================="
echo ""

# Determine app location
if [ -f "$APP_DIR/app.py" ]; then
    cd "$APP_DIR"
elif [ -f "$(dirname "${BASH_SOURCE[0]}")/app.py" ]; then
    cd "$(dirname "${BASH_SOURCE[0]}")"
else
    echo "ERROR: app.py not found. Run setup.sh again to copy app files."
    exit 1
fi

python3 app.py
