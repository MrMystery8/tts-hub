# Unified MLX/MPS TTS + Voice Cloning Hub — Spec Sheet

This spec started as a design brief. It has now been updated to reflect the **current implemented state** in `tts-hub/` (v0.1), based on repository exploration and the adapters actually written.

## 1) Objective

Build **one local Web UI** (single “platform”) that can run and switch between these existing local repos/models from `../`:

- `chatterbox-multilingual` (Chatterbox Multilingual)
- `voxcpm-ane` (VoxCPM-ANE / CoreML+ANE)
- `f5-hindi-urdu` (F5-TTS Hindi/Urdu + Roman→Devanagari pipeline)
- `pocket-tts` (Kyutai Pocket TTS)
- `cosyvoice3-mlx` (CosyVoice3 on MLX)
- `index-tts` (IndexTTS2; treat as baseline implementation/optimizations)

Primary target: **Apple Silicon** with **PyTorch MPS / MLX / CoreML (ANE)**.

## 2) Constraints & Assumptions

- Runs locally; no cloud deployment required.
- Model weights are expected to be present locally or downloadable (HF cache/Hub); the hub must support **local-only** operation when weights are already cached.
- Each model may have different Python dependencies; the hub should tolerate this by running models in **isolated worker processes**.
- `ffmpeg` must be available on `PATH` for reliable audio conversion to WAV and optional output conversion (wav→mp3/flac).

## 3) Supported Models — Capability Matrix

| Model | Runtime | Primary Use | Voice Cloning Inputs | Ref Transcript Required | Languages | Output SR (typical) | Notable Optimizations / Notes |
|---|---|---|---|---|---|---:|---|
| **IndexTTS2** (`index-tts`) | PyTorch (MPS/CPU/CUDA) | High quality AR zero-shot TTS + emotion control | Speaker ref audio; optional emotion ref audio | No (speaker ref), **optional** for emotion text; supports emotion vectors/text modes | zh/en focus (repo supports mixed zh+pinyin + en improvements) | ~22.05kHz | **Worker mode**, MPS cache clearing, segment token chunking, emotion controls |
| **Chatterbox Multilingual** (`chatterbox-multilingual`) | PyTorch (MPS/CPU/CUDA) | Multilingual TTS + optional zero-shot cloning | Optional ref audio prompt (`audio_prompt_path`) | No | 23+ (incl. hi, zh, ar, …) | 24kHz | **<150 char chunking** + **crossfade stitching**; optional artifact cleanup pipeline; MPS env tuning |
| **F5 Hindi/Urdu** (`f5-hindi-urdu`) | PyTorch (MPS/CPU/CUDA) | Hindi/Urdu cloning with Roman→Hindi assist | Ref audio + ref text (can be Roman→Devanagari); gen text | Often yes (prompt text improves), depends on pipeline | hi/ur focused | (vocoder-dependent) | Roman Urdu → Devanagari pipeline + override engine; caching; speed/nfe controls |
| **CosyVoice3-MLX** (`cosyvoice3-mlx`) | MLX | Voice cloning (zero-shot / cross-lingual / instruct) | Ref audio; mode-dependent ref transcript/instruct text | **Yes for zero_shot** (strict formatting) | multi (mode-dependent) | 24kHz | MLX model caching + `mx.clear_cache()`; strict prompt prefix handling |
| **Pocket TTS** (`pocket-tts`) | PyTorch (CPU-first) | Low-latency CPU TTS + streaming | Optional ref audio (or predefined voices) | No | English (currently) | 24kHz | Streaming generation; KV-cache slicing for voice-state memory; long-text chunking built-in |
| **VoxCPM-ANE** (`voxcpm-ane`) | CoreMLTools (CPU+ANE) | Fast Apple Neural Engine TTS | Ref audio **and** ref transcript (prompt cache) | **Yes** (prompt_wav + prompt_text required together) | zh/en TN supported | 16kHz | ANE inference; prompt-cache reuse; single-worker generation queue recommended |

## 4) Unified Hub — Current Implementation (v0.1)

### 4.1 Folder Layout

- `tts-hub/webui.py`: FastAPI server + UI routes.
- `tts-hub/custom_ui/`: single front-end (HTML/JS/CSS) with model picker and per-model settings panels.
- `tts-hub/hub/`: hub core (ffmpeg helpers, worker launcher, model registry, path resolver).
- `tts-hub/workers/worker_*.py`: one worker per model, each running under that repo’s own `.venv` python.
- `tts-hub/outputs/`: generated audio outputs per model.
- `tts-hub/outputs/uploads/`: per-request upload staging area (converted to WAV).
- `tts-hub/tools/doctor.py`: sanity checks (ffmpeg + worker handshake using each repo’s `.venv`).

### 4.2 API Surface (what the UI calls)

- `GET /`: UI
- `GET /api/models`: list models
- `GET /api/info`: ffmpeg availability
- `POST /api/generate`: run selected model (multipart form)
- `POST /api/unload`: unload/recycle selected model worker

### 4.3 Worker Process Model

- One subprocess per model, started lazily on first request.
- Worker IPC: newline-delimited JSON over stdin/stdout:
  - first line is handshake: `{"ok": true, "msg": "..."}`
  - commands: `gen`, `unload`, `shutdown`
- Workers run with:
  - `cwd=<repo_root>` (critical for local relative paths)
  - `PYTHONPATH` set per repo so imports resolve (e.g., Chatterbox uses `chatterbox/src`).

### 4.4 Audio Conversion (implemented)

- On upload, hub converts `prompt_audio` / `emo_audio` to **mono WAV** via ffmpeg (`hub/audio_utils.py`).
- Sample-rate normalization is intentionally delegated to each model’s existing pipeline (each repo already resamples internally as needed).
- Outputs are always written as WAV first by the worker, then optionally converted by ffmpeg to `mp3`/`flac` in the hub server.

## 5) UI Requirements (implemented vs planned)

### 5.1 Web UI (implemented)

- **Model picker** (dropdown) + per-model description + “Unload” button (recycles worker process).
- Common inputs:
  - `text` (required)
  - `output_format` (wav/mp3/flac)
  - reference audio upload/record (optional for some models)
- Model-specific panels that appear only when that model is selected:
  - **Voice cloning**: ref audio upload/record; optional transcript; optional voice-cache selection
  - **Language** selection where applicable
  - **Model variant** selection where applicable (e.g., CosyVoice3 fp16/8bit/4bit)
  - **Advanced controls** (exposed as toggles/sliders) mapped from each repo’s WebUI knobs

### 5.2 Web UI (not yet implemented but desirable)

- Per-model “status” panel (device, loaded/unloaded, last generation time, memory stats).
- Output history list + cleanup controls.
- A global “save outputs” toggle and/or configurable output root.
- A global “post-normalize” toggle (peak normalize / loudness normalize).

## 6) Long Text Handling

- Preserve each repo’s optimized strategy:
  - Chatterbox: **split into ≤150 char chunks** at language-aware sentence boundaries + **crossfade stitch**
  - Pocket TTS: use built-in sentence chunking/streaming
  - IndexTTS2: use token-based segmentation (`max_text_tokens_per_segment`)
- Hub UI exposes model-specific long-form settings where implemented (Chatterbox chunking; Index token segmentation).

## 7) Caching & Memory Management

- Default to **one in-flight generation per model** (queue/lock) to avoid thread-safety and MPS issues.
- Provide explicit controls:
  - “Unload model” (free RAM / clear MLX cache / clear MPS cache)
  - “Recycle worker” (restart process) — implemented via `/api/unload`
- IndexTTS2 worker explicitly runs cross-platform cache clearing after generation (`indextts.memory_utils.clear_memory()` when available).
- Chatterbox worker clears MPS cache after generation and unload.
- CosyVoice3 worker clears MLX cache after generation and unload.

## 8) Non-Functional Requirements

- **Apple Silicon-first**: prefer MPS for PyTorch models, MLX for MLX models, ANE for CoreML models.
- **Stability**: isolate each model in its own process to prevent cross-model dependency conflicts and to contain MPS/Metal memory growth.
- **Offline-friendly**: allow specifying local checkpoint dirs; prefer local HF cache if present.
- **Reproducibility**: expose `seed` inputs when supported.

## 9) Model Adapters — Current Inputs/Controls (what’s wired)

### 9.1 IndexTTS2 (`workers/worker_index_tts2.py`)

- Requires: `prompt_audio`
- Supports: `emo_mode` (`speaker|emo_ref|emo_vector|emo_text`), `emo_alpha`, `emo_audio`, `emo_vector`, `emo_text`, `use_random`
- Supports: `max_text_tokens_per_segment`, `max_mel_tokens`, `fast_mode`
- Supports: sampling params (`do_sample`, `temperature`, `top_p`, `top_k`, `num_beams`, `repetition_penalty`, `length_penalty`)

### 9.2 Chatterbox Multilingual (`workers/worker_chatterbox_mtl.py`)

- Supports: optional `prompt_audio` cloning
- Supports: `language_id`, `cfg_weight`, `temperature`, `exaggeration`, `fast_mode`
- Long-form: `enable_chunking`, `max_chunk_chars` (default 150), `crossfade_ms` (default 50)
- Optional enhancement: `enable_df` (DeepFilterNet), `enable_novasr` (NovaSR 48k)

### 9.3 F5 Hindi/Urdu (`workers/worker_f5_hindi_urdu.py`)

- Requires: `prompt_audio`
- Supports: `prompt_text` (recommended), `roman_mode`, `overrides_enabled`, `overrides_text`
- Supports: `cross_fade_duration`, `nfe_step`, `speed`, `remove_silence`, `seed`

### 9.4 CosyVoice3-MLX (`workers/worker_cosyvoice3_mlx.py`)

- Requires: `prompt_audio`
- Supports: `cosy_model` (`8bit|4bit|fp16`), `mode` (`zero_shot|cross_lingual|instruct`), `language`, `speed`
- zero_shot requires `prompt_text` (reference transcript); instruct requires `instruct_text`
- Implements strict CosyVoice3 prompt-prefix formatting internally

### 9.5 Pocket TTS (`workers/worker_pocket_tts.py`)

- Supports: `prompt_audio` (voice cloning) OR `voice` (HF voice URL)
- Supports: generation params `temperature`, `lsd_decode_steps`, `noise_clamp`, `eos_threshold`, `truncate_prompt`

### 9.6 VoxCPM-ANE (`workers/worker_voxcpm_ane.py`)

- Supports: `voice` (cached voice name) OR (`prompt_audio` + `prompt_text`)
- Supports: `cfg_value`, `inference_timesteps`, `max_length`

## 7) Setup Requirements

- Install `ffmpeg` globally (macOS recommended):
  - `brew install ffmpeg`
- Ensure each repo’s environment is available (either their `.venv` or a shared env that satisfies imports).

## 8) Known Model-Specific UI Requirements (Must Be Supported)

- **CosyVoice3-MLX**: `zero_shot` requires a **reference transcript**, and it must be formatted with the required prefix (`You are a helpful assistant.<|endofprompt|>...`).
- **VoxCPM-ANE**: `prompt_wav_path` and `prompt_text` must both be provided for cloning; expose `cfg_value`, `inference_timesteps`, `max_length`.
- **IndexTTS2**: expose emotion controls (speaker / emo_ref / emo_vector / emo_text), and segment controls; preserve MPS worker-mode behavior.
- **Chatterbox Multilingual**: preserve **≤150 char chunking** + **stitching**; expose `cfg_weight`, `temperature`, `exaggeration`, `fast_mode`, and optional enhancement toggles.
- **F5 Hindi/Urdu**: preserve Roman→Devanagari conversion and override system; expose `nfe_step`, `cross_fade_duration`, `speed`, `remove_silence`, and seed.

## 9) What’s Still “Spec” (not yet built)

- Automatic worker recycling based on memory thresholds (IndexTTS2’s worker-manager style).
- Listing/creating/managing VoxCPM voice caches from the hub UI (currently only “voice name” input).
- Streaming endpoints (PocketTTS streaming, Index streaming, etc.).
- One-click “download required weights” flows and full offline install story.
