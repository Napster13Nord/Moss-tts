"""
MOSS-TTS 1.7B - Zero-Shot Voice Cloning
RunPod / local server version (converted from Google Colab notebook)
Original notebook by AIQUEST | Model by OpenMOSS Team
"""

import os
import gc
import time
import atexit
import traceback
import warnings
from datetime import datetime

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor, GenerationConfig
import gradio as gr

# ── Environment ───────────────────────────────────────────────────────────────
# These are set by start.sh; fall back to sensible defaults for local runs.
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
OUTPUT_DIR = os.environ.get("MOSS_OUTPUT_DIR", "/workspace/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure HuggingFace uses network-storage cache
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME

# ── Warnings & backend settings ───────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device} | dtype: {dtype}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM total: {vram_total:.1f} GB")


# ── Custom GenerationConfig ───────────────────────────────────────────────────
class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32


# ── Global model state ────────────────────────────────────────────────────────
model = None
processor = None


def cleanup_model():
    """Unload model and free GPU memory."""
    global model, processor
    if model is not None:
        print("Cleaning up model from GPU...")
        del model
        model = None
    if processor is not None:
        if hasattr(processor, "audio_tokenizer"):
            del processor.audio_tokenizer
        del processor
        processor = None
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared.")


atexit.register(cleanup_model)


def resolve_attn_implementation() -> str:
    return "sdpa" if device == "cuda" else "eager"


def load_model():
    """Load model from HuggingFace cache (no download if already cached)."""
    global model, processor

    if model is None:
        print("Loading MOSS-TTS model from cache...")
        attn_implementation = resolve_attn_implementation()

        processor = AutoProcessor.from_pretrained(
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            trust_remote_code=True,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        model = AutoModel.from_pretrained(
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        model.eval()

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"Model loaded. VRAM used: {vram_used:.2f} GB")

    return model, processor


# ── Quality presets ───────────────────────────────────────────────────────────
PRESETS = {
    "Fast (8 RVQ)": {
        "n_vq": 8, "text_temp": 1.5, "audio_temp": 0.95,
        "text_top_p": 1.0, "audio_top_p": 0.95,
        "text_top_k": 50, "audio_top_k": 50, "audio_rep_pen": 1.1,
    },
    "Balanced (16 RVQ)": {
        "n_vq": 16, "text_temp": 1.5, "audio_temp": 0.95,
        "text_top_p": 1.0, "audio_top_p": 0.95,
        "text_top_k": 50, "audio_top_k": 50, "audio_rep_pen": 1.1,
    },
    "High Quality (24 RVQ)": {
        "n_vq": 24, "text_temp": 1.5, "audio_temp": 0.95,
        "text_top_p": 1.0, "audio_top_p": 0.95,
        "text_top_k": 50, "audio_top_k": 50, "audio_rep_pen": 1.1,
    },
    "Maximum (32 RVQ)": {
        "n_vq": 32, "text_temp": 1.5, "audio_temp": 0.95,
        "text_top_p": 1.0, "audio_top_p": 0.95,
        "text_top_k": 50, "audio_top_k": 50, "audio_rep_pen": 1.1,
    },
}


def apply_preset(preset_name):
    p = PRESETS[preset_name]
    return (
        p["n_vq"], p["text_temp"], p["text_top_p"], p["text_top_k"],
        p["audio_temp"], p["audio_top_p"], p["audio_top_k"], p["audio_rep_pen"],
    )


# ── Speech generation ─────────────────────────────────────────────────────────
def generate_speech(
    text, reference_audio, max_new_tokens, speed,
    text_temp, text_top_p, text_top_k,
    audio_temp, audio_top_p, audio_top_k,
    audio_repetition_penalty, n_vq,
    progress=gr.Progress(),
):
    if not text or not text.strip():
        return None, "Please enter some text."

    try:
        progress(0, desc="Loading model...")
        model, processor = load_model()

        estimated_duration = max_new_tokens / 12.5
        status = (
            f"Text length: {len(text):,} chars\n"
            f"Target: {max_new_tokens} tokens (~{estimated_duration / 60:.1f} min)\n\n"
        )
        yield None, status

        # Build conversation
        progress(0.1, desc="Processing input...")
        if reference_audio is not None:
            status += f"Voice cloning: {os.path.basename(reference_audio)}\n"
            conversations = [[
                processor.build_user_message(text=text, reference=[reference_audio])
            ]]
        else:
            status += "Using default voice\n"
            conversations = [[processor.build_user_message(text=text)]]

        yield None, status

        batch = processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Avoid exact 1.0 temperature (known bug in the model)
        if text_temp == 1.0:
            text_temp = 1.001
        if audio_temp == 1.0:
            audio_temp = 1.001

        # Build generation config
        generation_config = DelayGenerationConfig()
        generation_config.pad_token_id = processor.tokenizer.pad_token_id
        generation_config.eos_token_id = 151653
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = True
        generation_config.do_sample = True
        generation_config.num_beams = 1
        generation_config.n_vq_for_inference = n_vq
        generation_config.do_samples = [True] * (n_vq + 1)
        generation_config.layers = [
            {"repetition_penalty": 1.0, "temperature": text_temp,
             "top_p": text_top_p, "top_k": text_top_k}
        ] + [
            {"repetition_penalty": audio_repetition_penalty, "temperature": audio_temp,
             "top_p": audio_top_p, "top_k": audio_top_k}
        ] * n_vq

        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        status += "\nGenerating audio...\n"
        yield None, status

        start_time = time.time()
        progress(0.3, desc="Generating...")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        gen_time = time.time() - start_time

        progress(0.85, desc="Decoding audio...")
        status += f"Generated in {gen_time:.1f}s\nDecoding...\n"
        yield None, status

        decoded_messages = processor.decode(outputs)
        audio = decoded_messages[0].audio_codes_list[0]

        if device == "cuda":
            del outputs, input_ids, attention_mask, batch, decoded_messages
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        # Speed adjustment
        progress(0.94, desc="Adjusting speed...")
        if speed != 1.0:
            sample_rate = processor.model_config.sampling_rate
            new_sr = int(sample_rate * speed)
            audio = torchaudio.transforms.Resample(new_sr, sample_rate)(
                torchaudio.transforms.Resample(sample_rate, new_sr)(audio.unsqueeze(0))
            ).squeeze(0)

        # Save to network storage
        progress(0.97, desc="Saving...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"moss_tts_{timestamp}.wav")
        torchaudio.save(
            output_path,
            audio.unsqueeze(0),
            processor.model_config.sampling_rate,
        )

        duration = len(audio) / processor.model_config.sampling_rate
        vram_used = torch.cuda.memory_allocated() / 1024**3 if device == "cuda" else 0
        rtf = gen_time / duration if duration > 0 else 0

        progress(1.0, desc="Done!")
        status += (
            f"\nDone!\n"
            f"Audio: {duration:.1f}s ({duration / 60:.2f} min)\n"
            f"Generation time: {gen_time:.1f}s\n"
            f"RTF: {rtf:.2f}x\n"
            f"Speed: {speed}x\n"
            f"VRAM: {vram_used:.2f} GB\n"
            f"RVQ: {n_vq}/32\n"
            f"Saved: {output_path}"
        )
        yield output_path, status

    except torch.cuda.OutOfMemoryError:
        yield None, (
            "OUT OF MEMORY\n\n"
            f"Tried: {max_new_tokens} tokens with {n_vq} RVQ\n\n"
            "Solutions:\n"
            "1. Reduce Max Tokens\n"
            "2. Switch to Fast (8 RVQ) preset\n"
            "3. Click 'Clear GPU' then retry\n\n"
            "Approximate limits per RVQ setting:\n"
            "  8 RVQ  -> ~7200 tokens (~12 min)\n"
            "  16 RVQ -> ~4800 tokens (~8 min)\n"
            "  24 RVQ -> ~3000 tokens (~5 min)\n"
            "  32 RVQ -> ~2400 tokens (~4 min)"
        )
    except Exception as e:
        yield None, f"Error: {e}\n\n{traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
custom_css = """
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px; border-radius: 12px; margin-bottom: 16px;
    text-align: center; color: white;
}
.header h1 { margin: 0 0 8px 0; font-size: 1.8em; color: white !important; }
.header p  { margin: 4px 0; opacity: 0.95; color: white !important; }
.footer {
    text-align: center; padding: 14px; margin-top: 16px;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; font-size: 0.9em;
}
.footer a { color: #ffd700 !important; text-decoration: none; font-weight: 600; }
"""

with gr.Blocks(title="MOSS-TTS", theme=gr.themes.Soft(), css=custom_css) as demo:

    gr.HTML("""
    <div class="header">
        <h1>MOSS-TTS 1.7B - Zero-Shot Voice Cloning</h1>
        <p>Running on RunPod with persistent network storage</p>
        <p style="font-size:0.85em; opacity:0.8;">
            Model by OpenMOSS &nbsp;|&nbsp; Notebook by AIQUEST
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter the text you want to convert to speech...",
                lines=10,
                value="Hello! This is MOSS text-to-speech running on RunPod with network storage.",
            )

            reference_audio = gr.Audio(
                label="Reference Voice (optional - upload for voice cloning)",
                type="filepath",
                sources=["upload"],
            )

            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Balanced (16 RVQ)",
                label="Quality Preset",
            )

            with gr.Row():
                max_tokens = gr.Slider(50, 5000, 2500, step=100, label="Max Tokens")
                speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")

            with gr.Accordion("Advanced Settings", open=False):
                n_vq = gr.Slider(8, 32, 8, step=1, label="RVQ Layers")
                with gr.Row():
                    text_temp  = gr.Slider(0.1, 2.0, 1.5, step=0.1,  label="Text Temp")
                    text_top_p = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="Text Top-P")
                    text_top_k = gr.Slider(1, 100, 50,   step=1,    label="Text Top-K")
                with gr.Row():
                    audio_temp  = gr.Slider(0.1, 2.0, 0.95, step=0.05, label="Audio Temp")
                    audio_top_p = gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Audio Top-P")
                with gr.Row():
                    audio_top_k  = gr.Slider(1, 100, 50, step=1,    label="Audio Top-K")
                    audio_rep_pen = gr.Slider(1.0, 1.5, 1.1, step=0.05, label="Rep Penalty")

            with gr.Row():
                generate_btn = gr.Button("Generate Speech", variant="primary",  size="lg", scale=3)
                clear_btn    = gr.Button("Clear GPU",        variant="secondary", size="lg", scale=1)

        with gr.Column(scale=1):
            audio_output  = gr.Audio(label="Generated Audio", type="filepath")
            status_output = gr.Textbox(label="Status", lines=18, interactive=False)

    gr.HTML("""
    <div class="footer">
        Made with love by <b>AIQUEST</b> &nbsp;|&nbsp;
        Running on <b>RunPod</b> with persistent network storage
    </div>
    """)

    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[n_vq, text_temp, text_top_p, text_top_k,
                 audio_temp, audio_top_p, audio_top_k, audio_rep_pen],
    )

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, reference_audio, max_tokens, speed,
                text_temp, text_top_p, text_top_k,
                audio_temp, audio_top_p, audio_top_k,
                audio_rep_pen, n_vq],
        outputs=[audio_output, status_output],
    )

    def clear_memory():
        cleanup_model()
        return "GPU cleared. Ready for next generation."

    clear_btn.click(fn=clear_memory, inputs=[], outputs=[status_output])


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MOSS-TTS Gradio interface starting...")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")
    demo.launch(
        server_name="0.0.0.0",  # Bind to all interfaces so RunPod can expose the port
        server_port=7860,
        share=True,             # Also generate a public gradio.live link
        debug=False,
    )
