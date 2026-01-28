# TTS Hub — Implementation Summary (What’s Done So Far)

This document explains what was implemented in `tts-hub/` so far, how it works, and how it maps to each of your existing model repos.

## 1) What You Asked For (restated)

You asked for:

1. A review of these TTS / voice-cloning stacks:
   - `chatterbox-multilingual`
   - `voxcpm-ane`
   - `f5-hindi-urdu`
   - `pocket-tts`
   - `cosyvoice3-mlx`
   - `index-tts`
2. A single platform / one Web UI to access all of them.
3. Preservation of each repo’s optimizations and special requirements:
   - Chatterbox multilingual long-form generation chunking (≤150 chars) + stitching
   - IndexTTS2 MPS cache clearing / stability patterns
   - F5 Hindi/Urdu Roman→Hindi pipeline and pronunciation overrides
   - Model-specific requirements like “reference transcript required” (CosyVoice3 zero_shot, VoxCPM prompt cache)
4. Ensure ffmpeg is available globally so conversions to WAV/etc happen consistently.

## 2) What Was Delivered

### 2.1 New Folder

All work is in: `tts-hub/`

Key files:

- Spec sheet: `tts-hub/SPEC_SHEET.md` (updated to match implementation)
- Hub server: `tts-hub/webui.py`
- UI: `tts-hub/custom_ui/index.html`, `tts-hub/custom_ui/static/app.js`, `tts-hub/custom_ui/static/styles.css`
- Hub core utilities: `tts-hub/hub/`
- Per-model workers/adapters: `tts-hub/workers/worker_*.py`
- Diagnostics: `tts-hub/tools/doctor.py`

### 2.2 One UI, Multiple Backends (isolated)

The hub is implemented as:

- A **single FastAPI server** (the hub) that hosts one UI and exposes simple `/api/*` endpoints.
- A **separate worker process per model**, launched on-demand.

This isolation is important because each repo has different dependency sets and “best” runtime:

- PyTorch/MPS repos (IndexTTS2, Chatterbox, F5) want different pinned stacks and different MPS memory settings.
- MLX (CosyVoice3) has its own dependencies + cache behavior.
- CoreML/ANE (VoxCPM-ANE) loads compiled CoreML bundles and has its own constraints.

Instead of trying to install everything into one Python environment (guaranteed to become fragile), each worker runs under the model repo’s own virtualenv:

- `<repo>/.venv/bin/python`

The hub launches the correct interpreter automatically (see `tts-hub/hub/paths.py`).

Note on “internet research”:

- This implementation is based on the **local repos present in your workspace** (their code + READMEs). If you want the hub to include remote-model download UX or model-card/license summaries, that’s a follow-up item (outlined in `tts-hub/ROADMAP_AND_IMPROVEMENTS.md`).

## 3) How It Works (Architecture)

### 3.1 Hub HTTP API

Implemented endpoints in `tts-hub/webui.py`:

- `GET /` — serves the UI
- `GET /api/models` — returns model list from `tts-hub/hub/model_registry.py`
- `GET /api/info` — returns ffmpeg availability
- `POST /api/generate` — receives inputs (multipart form), converts uploads to WAV, delegates generation to the correct worker, returns the final audio file
- `POST /api/unload` — unloads/recycles a model worker process

### 3.2 Worker IPC Protocol

The hub talks to workers over stdin/stdout using newline-delimited JSON:

- Worker prints handshake on startup:
  - `{"ok": true, "msg": "..."}`
- Hub sends:
  - `{"cmd":"gen","model_id":"...","request":{...}}`
  - `{"cmd":"unload"}`
  - `{"cmd":"shutdown"}`
- Worker replies:
  - `{"ok": true, "result": {"output_path": "...", "meta": {...}}}`
  - or `{"ok": false, "error": "..."}` on error

Implementation: `tts-hub/hub/subprocess_worker.py`

### 3.3 Repo Path + Python Resolution

Workers are launched with:

- `cwd=<repo_root>` — so relative repo paths work (e.g., `checkpoints/`, local `models/`)
- `PYTHONPATH` injected — so imports resolve without pip-installing repos into each venv

Resolver logic is in: `tts-hub/hub/paths.py`

### 3.4 File Outputs + Upload Staging

Uploads:

- Saved under `tts-hub/outputs/uploads/<task_id>/`
- Converted to WAV using ffmpeg:
  - `prompt.wav`
  - `emo.wav` (if provided)

Outputs:

- Worker writes output WAV to:
  - `tts-hub/outputs/<model_id>/...`
- Hub optionally converts to mp3/flac (if user selects) and streams the final file back to the browser.

ffmpeg helper functions: `tts-hub/hub/audio_utils.py`

## 4) The Unified Web UI

### 4.1 Common UI Features

Implemented in `tts-hub/custom_ui/index.html` + `tts-hub/custom_ui/static/app.js`:

- Model picker dropdown
- Reference audio upload + in-browser microphone recording (saved as WAV)
- Text prompt
- Output format selector (`wav`, `mp3`, `flac`)
- “Unload” button (recycles the selected model worker)

### 4.2 Model-Specific UI Panels (implemented)

The UI shows/hides panels depending on the selected model.

#### IndexTTS2

Exposes:

- Emotion mode: `speaker | emo_ref | emo_vector | emo_text`
- Emotion weight (`emo_alpha`)
- Random sampling toggle
- Token segmentation knobs: `max_text_tokens_per_segment`, `max_mel_tokens`
- Fast mode (forces greedy decoding and smaller mel limits)
- Sampling knobs: `do_sample`, `temperature`, `top_p`, `top_k`, `num_beams`, `repetition_penalty`, `length_penalty`

#### Chatterbox Multilingual

Exposes:

- `language_id`
- Cloning toggle: “use ref audio”
- `cfg_weight`, `temperature`, `exaggeration`
- `fast_mode`
- Long-form: enable chunking + `max_chunk_chars` (default 150) + `crossfade_ms`
- Optional enhancement toggles:
  - DeepFilterNet (denoise)
  - NovaSR (48kHz upscaling)

#### F5 Hindi/Urdu

Exposes:

- Roman input toggle (Roman Urdu → Devanagari before synthesis)
- Pronunciation overrides toggle + override mapping text area
- `cross_fade_duration`, `nfe_step`, `speed`
- Optional remove silence
- Seed

#### CosyVoice3-MLX

Exposes:

- Model: `8bit | 4bit | fp16`
- Mode: `zero_shot | cross_lingual | instruct`
- Language and speed
- Instruction text (required for `instruct`)
- Transcript input is required for `zero_shot` (UI enforces this)

#### Pocket TTS

Exposes:

- “voice” HF URL field (used when no reference audio is uploaded)
- `temperature`, `lsd_decode_steps`, `eos_threshold`, optional `noise_clamp`
- Optional truncate prompt audio

#### VoxCPM-ANE

Exposes:

- Optional cached voice name (if provided, prompt audio+transcript can be omitted)
- `cfg_value`, `inference_timesteps`, `max_length`
- Transcript is required when using prompt audio (UI enforces this)

## 5) What Each Worker Actually Does (Repo Mapping)

### 5.1 IndexTTS2 Worker

File: `tts-hub/workers/worker_index_tts2.py`

- Imports `indextts.infer_v2.IndexTTS2` and calls `infer(...)`.
- Preserves MPS stability patterns:
  - MPS watermark env defaults are set before torch import
  - calls `indextts.memory_utils.clear_memory()` after generation (fallbacks to gc + `torch.mps.empty_cache()` if needed)
  - sets per-process MPS memory fraction if available
- Output sample rate is whatever IndexTTS2 produces (~22050 Hz).

### 5.2 Chatterbox Multilingual Worker

File: `tts-hub/workers/worker_chatterbox_mtl.py`

- Imports `chatterbox.mtl_tts.ChatterboxMultilingualTTS`.
- Preserves the most important optimization you called out:
  - **long-form ≤150 character chunking** + **crossfade stitching**
- Optional post-processing pipeline (toggled):
  - DeepFilterNet denoise
  - NovaSR 48k upscaling
- Clears MPS cache after generation/unload.

### 5.3 F5 Hindi/Urdu Worker

File: `tts-hub/workers/worker_f5_hindi_urdu.py`

- Re-implements the Roman → Devanagari pipeline and override system (from your `roman_gradio.py`) so the hub does not depend on running Gradio.
- Loads the local Hindi checkpoint:
  - `f5-hindi-urdu/models/hindi/model_2500000.safetensors`
  - `f5-hindi-urdu/models/hindi/vocab.txt`
- Uses `f5_tts.infer.utils_infer.infer_process(...)` for generation.

### 5.4 CosyVoice3-MLX Worker

File: `tts-hub/workers/worker_cosyvoice3_mlx.py`

- Uses `mlx_audio`’s `load_model()` and `model.generate(**kwargs)`.
- Preserves the strict “prompt formatting” requirement:
  - For `zero_shot`: `ref_text` is formatted as `You are a helpful assistant.<|endofprompt|>...`
  - For `cross_lingual`: the synthesized text is prefixed similarly
  - For `instruct`: uses the instruction formatting pattern from your CosyVoice3 UI
- Clears MLX cache after each run/unload.

### 5.5 Pocket TTS Worker

File: `tts-hub/workers/worker_pocket_tts.py`

- Uses the repo’s public API:
  - `pocket_tts.TTSModel.load_model()`
  - `get_state_for_audio_prompt(...)`
  - `generate_audio(...)`
- Supports either:
  - uploaded reference audio (voice cloning), or
  - a “voice” HF URL (catalog voice)

Important operational note:

- PocketTTS voice-cloning weights can require HF terms acceptance/login depending on your cache state; errors will surface directly if HF access is blocked.

### 5.6 VoxCPM-ANE Worker

File: `tts-hub/workers/worker_voxcpm_ane.py`

- Imports `voxcpmane.server` and calls `generate_audio_chunks(...)`.
  - This mirrors the repo’s own server-side inference path (including text normalization, punctuation fixes, etc.).
- Requires prompt transcript when using prompt audio (same as the repo).
- Writes output WAV at 16kHz.

## 6) ffmpeg Status

ffmpeg is required and is used in two places:

- Uploaded audio → WAV (mono) conversion
- Output WAV → mp3/flac conversion (when requested)

Diagnostics tool checks it:

- `python3 tts-hub/tools/doctor.py`

## 7) Diagnostics (Implemented)

`tts-hub/tools/doctor.py` verifies:

- `ffmpeg` exists on `PATH`
- Each worker can be spawned using the target repo’s `.venv` interpreter
- Each worker responds to handshake + shutdown

This confirms the wiring and process launch assumptions are correct.

## 8) Current Limitations (Not “broken”, just not done yet)

- The hub UI does not yet show per-model runtime stats (loaded/unloaded, device, mem usage, last gen time).
- There is no output history list or output cleanup UI.
- “One-click” voice cache management for VoxCPM (list voices, create cache, etc.) is not implemented (only a “voice name” textbox exists).
- No streaming HTTP audio endpoints are exposed yet (everything returns a finished file).
- Concurrency is intentionally conservative: one generation at a time per model worker (safer for MPS / non-thread-safe inference).

## 9) Watermark Module (Current Status)

The provenance watermarking system (`watermark/`) is now fully functional and integrated.

### 9.1 Architecture (Multiclass)
- **Design:** Switched from legacy 32-bit payload to a simplified **(N+1)-class attribution** system (Clean + N Models).
- **Detector:** Predicts `P(clean)` vs `P(watermarked)`.
- **Identifier:** `K`-way classifier (via `id` head) to attribute audio to specific TTS models (IndexTTS, Chatterbox, etc.).
- **Localization:** Uses a "watermarkness" score over time (`loc` head) to handle truncated or concatenated audio.

### 9.2 Training Tools
- **Quick Smoke Train (`watermark/scripts/quick_voice_smoke_train.py`):**
  - The main training loop.
  - Supports 3-stage training: Decoder Pretrain -> Encoder Train -> End-to-End Finetune.
  - Handles on-the-fly dataset generation/augmentation (reverb, noise).
  - **Smart Checkpointing:** Auto-saves "best" model, prioritizing robust (reverb) metrics and guarding against detector collapse.
  
- **Overnight Tuner (`watermark/scripts/overnight_tune_s1.py`):**
  - Automates hyperparameter search for weight balancing (`detect_weight` vs `id_weight`).
  - Uses a shared manifest for fair comparisons across trials.

### 9.3 Visualization & Monitoring
- **Live Dashboard (`watermark/scripts/live_dashboard.py`):**
  - Real-time FastAPI + Chart.js dashboard.
  - Visualizes Detection AUC, ID Accuracy, Separation, and Loss curves.
  - Access at `http://localhost:8765` after launching.
  
### 9.4 Benchmarking
- **Datasets:** Scripts to generate `mini_benchmark_data` (small, fast) and `medium_benchmark_data` (LibriSpeech subset).
- **Evaluation:** Integrated probe steps during training ensure continuous evaluation against "unseen" data.
