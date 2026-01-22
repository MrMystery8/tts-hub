# TTS Hub — What’s Remaining + Improvement Opportunities

This document lists what’s still missing (relative to a “complete” unified platform) and what improvements are worth adding next. It is written against the current state of `tts-hub/` (v0.1).

## 1) Highest-Priority Remaining Work (Recommended Next Steps)

### 1.1 Add a Real “Model Status” System

Goal: make the hub feel like one coherent product, not just a router.

Suggested additions:

- `/api/status?model_id=...`:
  - worker running/not running
  - device (mps/cpu/cuda/mlx/ane)
  - last generation time + duration
  - worker PID
  - (where possible) memory stats
- UI badges:
  - “Loaded” / “Unloaded”
  - device type
  - last run time

Implementation hint:

- The hub can track status on the server side (timestamps, last errors).
- For PyTorch MPS, you can expose:
  - `torch.mps.current_allocated_memory()`
  - `torch.mps.driver_allocated_memory()`
- For MLX, you can expose at least “model loaded/unloaded”; MLX doesn’t provide the same memory counters.

### 1.2 Add Worker Recycling Policy (IndexTTS-style Stability)

IndexTTS’s repo includes a worker-recycling idea based on memory growth. The hub currently has manual “Unload”.

Add:

- Configurable recycling:
  - recycle after `N` generations
  - recycle if MPS driver memory > threshold

Where to implement:

- `tts-hub/hub/subprocess_worker.py` or `tts-hub/hub/hub_manager.py` can enforce a policy.
- Alternatively, each worker can self-report memory stats and request a recycle.

### 1.3 Add “VoxCPM Voice Cache Management” in Hub

VoxCPM-ANE supports cached voices (npy + txt). The hub currently only accepts a typed voice name.

Add endpoints + UI:

- List voices:
  - default cache voices (in the model snapshot cache)
  - custom cache voices (`~/.cache/ane_tts` or equivalent)
- Create voice cache:
  - upload audio + transcript + voice name → run `create_custom_voice(...)`
- Delete voice cache:
  - remove `.npy` + `.txt` (and optional raw audio)

You already have the underlying implementation in `voxcpmane.server` and `VoxCPMANE.create_custom_voice()`.

### 1.4 Add Streaming APIs (Optional but High Value)

Several stacks support streaming or chunked generation:

- PocketTTS has `generate_audio_stream()`
- IndexTTS2 has a `stream_return` mode in `infer()`
- VoxCPM server already generates audio chunks internally

Add:

- `/api/generate-stream` that returns:
  - either chunked WAV (hard) or raw PCM stream + final WAV (pragmatic)
  - or websockets (best UX for “voice agent” style)

This is a product-level improvement: users see first audio quickly.

## 2) Model-Specific Gaps vs Their Original UIs

### 2.1 IndexTTS2

Not yet exposed in the hub UI:

- Glossary management (`glossary.yaml` editing + “enable glossary” toggle)
- Segment preview UI (tokenization preview table from Gradio)
- “Experimental” controls (depending on what you want to keep)

Potential improvement:

- Add “preset” buttons (Quality/Balanced/Fast) like Index’s custom UI.

### 2.2 Chatterbox Multilingual

Currently preserved:

- ≤150 char chunking + crossfade stitching

Not yet preserved (from your more advanced optimized script):

- chunk-level artifact detection / cleaning heuristics
- silence trimming between chunks
- more configurable chunk logic (sentence boundaries per language, min chunk size)
- seed input
- selectable “default prompt audio per language” and/or built-in language demo prompts

Recommendation:

- Add `seed` + chunking advanced knobs (min chunk length, punctuation splitting options).

### 2.3 F5 Hindi/Urdu

Currently preserved:

- Roman→Devanagari + overrides
- generation core knobs (`nfe_step`, `speed`, seed, remove_silence)

Not integrated:

- multi-speech-type UI sections
- Qwen chat helper (“generate text”) flows
- caching strategy used by Gradio app (LRU caching of inference results)

Recommendation:

- Add optional caching of “reference preprocessing” outputs (ref audio tensor + ref text tokens) for faster repeated generations with the same voice.

### 2.4 CosyVoice3-MLX

Currently preserved:

- required prompt-prefix formatting
- MLX cache clearing
- mode-specific required fields

Not yet exposed:

- any STT helper for transcript extraction (if you want it)
- model unload status displayed in UI

Recommendation:

- Add optional “auto-transcribe ref audio” using a local ASR model (only if you want this; it adds dependencies).

### 2.5 Pocket TTS

Currently preserved:

- can run with HF voice URL (no prompt audio required)
- can use prompt audio for cloning when available

Not yet implemented:

- streaming audio endpoint
- improved “voice picker” UI (list of known voices + licenses)
- better UX when HF access is blocked for voice-cloning weights (clear error message + instruction)

Recommendation:

- Add `/api/pocket/voices` that returns a curated list + license info (either static JSON or pulled from their docs).

### 2.6 VoxCPM-ANE

Currently preserved:

- same inference pathway as `voxcpmane.server.generate_audio_chunks`

Not yet integrated:

- “serve” mode UI from VoxCPM-ANE repo
- voice cache compilation flows (see §1.3)

Recommendation:

- Add voice cache flows first; it’s the biggest UX win.

## 3) Hub-Level Product Improvements (Model-Independent)

### 3.1 Central Config File

Add something like:

- `tts-hub/config.yaml` or `tts-hub/config.json`

Use it to control:

- repo roots (if folder names change)
- python interpreter paths (if `.venv/` is missing)
- default model parameters per model
- output directory
- worker recycling thresholds

### 3.2 Better Error Surfacing

Currently, worker errors are returned as strings.

Improve by returning structured errors:

- `error_code` (e.g., `MISSING_TRANSCRIPT`, `MISSING_PROMPT_AUDIO`, `HF_AUTH_REQUIRED`)
- `hint` with actionable fix

### 3.3 Output Management

Add:

- “Recent generations” list in UI with:
  - model, timestamp, duration, download link
- “Clear outputs” button per model
- optional auto-clean old uploads in `outputs/uploads/`

### 3.4 Post-Processing Options

Add optional, model-independent audio post-processing:

- peak normalize
- loudness normalize (EBU R128)
- trim leading/trailing silence
- format conversion is already present via ffmpeg

## 4) Performance/Memory Improvements (Apple Silicon Focus)

### 4.1 MPS/Metal Memory

IndexTTS2 already has good patterns (clear cache + optional worker isolation).

Add:

- periodic `torch.mps.synchronize()` + `empty_cache()` in other torch workers (where safe)
- per-model “MPS watermark ratio” settings exposed via config

### 4.2 Reduce Reload Time

For large models:

- keep workers warm (default)
- add “Unload after X minutes idle” (optional)
- show loading progress/state in UI

## 5) Testing + Verification Improvements

Right now, wiring is verified via:

- `tts-hub/tools/doctor.py` (ffmpeg + worker handshake)

Add:

- per-model “smoke test” script that generates 1 second of audio using existing small prompts (if local weights exist)
- CI-friendly unit tests for:
  - request validation
  - field mapping
  - ffmpeg conversion helpers

## 6) UX/Workflow Enhancements

Suggestions:

- Presets per model (“Fast / Balanced / Quality”)
- Clear “required inputs” hints per model
- Auto-hide irrelevant sections (mostly done) + contextual help text (expandable)
- Add an “Advanced” toggle to reduce clutter for casual use

## 7) Packaging / Installation Improvements

Currently, the hub has a simple `requirements.txt`.

Improvements:

- Add `tts-hub/pyproject.toml` (uv/poetry) for reproducible installs
- Provide a single `run_hub.sh` that:
  - checks ffmpeg
  - runs doctor
  - launches hub using a known-good python env

