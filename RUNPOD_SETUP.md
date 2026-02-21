# MOSS-TTS on RunPod with Network Storage

Run MOSS-TTS 1.7B voice cloning on RunPod. After the first setup, pods start in ~30 seconds with no re-downloading.

---

## How It Works

| Location | What's stored |
|---|---|
| `/workspace/venv` | Python packages (torch, transformers, gradio, …) |
| `/workspace/hf_cache` | Model weights (~13 GB, downloaded once) |
| `/workspace/MOSS-TTS` | Cloned repo |
| `/workspace/outputs` | Your generated audio files |
| `/workspace/moss-tts-app` | `app.py` and `start.sh` |

Everything lives on the **network volume**, so terminating and restarting the pod keeps it all intact.

---

## Step-by-Step Setup

### Step 1 — Create a Network Volume on RunPod

1. Go to **RunPod → Storage → Network Volumes**
2. Click **+ New Network Volume**
3. Name it (e.g. `moss-tts`), set size to **30 GB** (safe margin for weights + outputs)
4. Choose the **same datacenter region** you will use for pods

### Step 2 — Create a Pod and Attach the Volume

1. Go to **Pods → + Deploy**
2. Choose a GPU template — recommended options:
   - **RTX 3090 / 4090** (24 GB VRAM) — best quality, all RVQ settings work
   - **RTX 3080** (10 GB VRAM) — works on Fast/Balanced presets
   - **A100 / A40** — overkill but fast
3. Select the **PyTorch** template (it has CUDA pre-installed)
4. Under **Volume**, attach your network volume → set mount path to `/workspace`
5. Under **Expose Ports**, add `7860` (for Gradio)
6. Deploy the pod

### Step 3 — First-Time Setup (run once)

Open a terminal in the pod (via RunPod web terminal or SSH):

```bash
# Clone this repo to get the scripts
git clone https://github.com/Napster13Nord/Moss-tts.git /tmp/moss-repo
cd /tmp/moss-repo

# Run the setup script — takes 15-25 min (downloads ~13 GB model)
bash setup.sh
```

The setup script:
- Creates a Python venv at `/workspace/venv`
- Installs all packages
- Clones the MOSS-TTS repo
- Downloads model weights to `/workspace/hf_cache`
- Copies `app.py` and `start.sh` to `/workspace/moss-tts-app/`

### Step 4 — Every Time You Restart a Pod

```bash
bash /workspace/moss-tts-app/start.sh
```

That's it. No reinstalling. No re-downloading. Starts in ~30 seconds.

---

## Accessing the Gradio UI

Two options:

**Option A — RunPod port forwarding (recommended)**
1. In the RunPod dashboard, open your pod
2. Click **Connect → HTTP Service → Port 7860**
3. A URL like `https://xxxx-7860.proxy.runpod.net` opens the UI

**Option B — Public Gradio link**
- When the app starts it prints a `gradio.live` URL
- Works without any port setup, valid for 72 hours

---

## GPU VRAM Guide

| GPU | VRAM | Best preset |
|---|---|---|
| RTX 3080 | 10 GB | Fast (8 RVQ) or Balanced (16 RVQ) |
| RTX 3090 / 4090 | 24 GB | Maximum (32 RVQ) |
| A100 40 GB | 40 GB | Maximum (32 RVQ) |

If you get **Out of Memory**: lower Max Tokens, switch to a lower RVQ preset, or click "Clear GPU" and retry.

---

## File Structure After Setup

```
/workspace/
├── venv/               Python virtual environment (persistent)
├── hf_cache/           HuggingFace model weights (~13 GB, persistent)
├── MOSS-TTS/           Cloned OpenMOSS repo (persistent)
├── outputs/            Your generated .wav files (persistent)
└── moss-tts-app/
    ├── app.py          Gradio application
    └── start.sh        Quick-start script
```

---

## Troubleshooting

**"Virtual environment not found"**
→ `setup.sh` hasn't been run yet, or the network volume wasn't mounted. Verify the volume is attached at `/workspace` and re-run setup.

**"CUDA version mismatch" or PyTorch errors**
→ The setup script auto-detects CUDA version. If it picks the wrong one, edit `setup.sh` and hardcode `TORCH_INDEX` to the right wheel URL.

**Model loads slowly on first start after setup**
→ Normal — first load moves weights from disk to VRAM (~1-2 min). Subsequent generations in the same session are fast.

**Audio saved where?**
→ `/workspace/outputs/moss_tts_YYYYMMDD_HHMMSS.wav` — persistent across pod restarts.
