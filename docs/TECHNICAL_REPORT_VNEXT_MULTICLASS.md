# Technical Report (Current Repo State): TTS Hub + Watermark (Multiclass Attribution)

Last updated: 2026-01-26

This repository contains two related systems that share a workspace:

1. **TTS Hub**: a FastAPI server + static Web UI that orchestrates multiple TTS / voice-cloning stacks via subprocess “workers”.
2. **Watermark**: a research/training package for **audio provenance watermarking**, currently implemented as a **multiclass attribution** problem (not a bit payload).

This document focuses on:

- What runs today and how to run it.
- The repo architecture (hub + workers + watermark).
- How the **quick smoke training** is configured.
- How the **live dashboard** is wired and which metrics it expects.

---

## Table of Contents

- [1. Repo Layout and Entrypoints](#1-repo-layout-and-entrypoints)
- [2. TTS Hub Architecture](#2-tts-hub-architecture)
- [3. Worker Architecture (Subprocess IPC)](#3-worker-architecture-subprocess-ipc)
- [4. Watermark Architecture (Multiclass Attribution)](#4-watermark-architecture-multiclass-attribution)
- [5. Training Stages and Losses](#5-training-stages-and-losses)
- [6. Evaluation, Attacks, and Probe Metrics](#6-evaluation-attacks-and-probe-metrics)
- [7. Smoke Tests and “Quick Smoke Train”](#7-smoke-tests-and-quick-smoke-train)
- [8. Live Dashboard (Single Log + Controller)](#8-live-dashboard-single-log--controller)
- [9. Outputs, Artifacts, and File Conventions](#9-outputs-artifacts-and-file-conventions)
- [10. Extending the System](#10-extending-the-system)
- [11. Legacy Code](#11-legacy-code)

---

## 1. Repo Layout and Entrypoints

Top-level structure (important directories only):

```
tts-hub/
├── webui.py                     # TTS Hub FastAPI entrypoint (serves UI + API)
├── run.sh                       # Creates `.venv` + runs webui.py
├── custom_ui/                   # Static frontend (HTML/JS/CSS)
├── hub/                         # Hub backend (model registry, subprocess workers, ffmpeg helpers)
├── workers/                     # Per-model worker adapters (one subprocess per model)
├── watermark/                   # Watermark training/eval package (encoder/decoder/training/dashboard)
├── tests/                       # Repo-level engineering-contract smoke tests
├── tools/doctor.py              # Worker handshake / environment diagnostics
├── outputs/                     # Hub outputs + training runs (logs, audio, metrics)
└── docs/                        # Design logs and technical reports (this file)
```

### 1.1 Primary entrypoints

- **Run the Hub (UI + API):** `./run.sh` (wraps `.venv/bin/python3 webui.py --port 7891`)
- **Check workers and ffmpeg:** `python3 tools/doctor.py`
- **Watermark quick smoke train:** `python -m watermark.scripts.quick_voice_smoke_train ...`
- **Watermark full train:** `python -m watermark.scripts.train_full ...`
- **Watermark dashboard:** `python -m watermark.scripts.live_dashboard ...`

### 1.2 Dependency split (important)

There are multiple “dependency domains”:

- **Hub runtime** (FastAPI + uvicorn) is installed via `requirements.txt` into `.venv/` (created by `run.sh`).
- **Watermark training** is a Python package defined by `pyproject.toml` (torch/torchaudio/sklearn/soundfile).
- **Each model worker** is expected to run using its model repo’s own venv under a sibling repo directory (see `hub/paths.py`).

---

## 2. TTS Hub Architecture

The hub is a thin orchestrator:

- A **FastAPI server** (`webui.py`) serves:
  - static UI assets from `custom_ui/`
  - a small JSON/HTTP API under `/api/*`
- A **HubManager** (`hub/hub_manager.py`) maintains a registry of models and lazily spawns one worker subprocess per model.
- **Workers** (in `workers/`) are “adapters”: they import and run code from sibling model repos and implement a small JSON-over-stdio protocol.

### 2.1 `webui.py`: HTTP API + static UI

The hub mounts the frontend and exposes endpoints:

- `GET /` → `custom_ui/index.html`
- `GET /api/models` → list available models (from `hub/model_registry.py`)
- `GET /api/info` → ffmpeg availability + server time
- `GET /api/status` → per-model process status + last generation stats
- `POST /api/generate` → normalize uploads → delegate generation to a worker → return audio file
- `POST /api/unload` → unload a model (shutdown its worker)

Key mechanics in `webui.py`:

- Uploads are saved under `outputs/uploads/<uuid>/...`.
- Reference audio uploads are converted to canonical WAV via ffmpeg (if present) before being passed to workers.

Snippet (upload normalization + worker delegation pattern):

```py
# webui.py
prompt_audio_in = _save_upload("prompt_audio")
prompt_audio_wav = uploads_root / "prompt.wav"
ffmpeg_convert_to_wav(input_path=prompt_audio_in, output_path=prompt_audio_wav)

result = manager.generate(model_id=model_id, request={...})
return FileResponse(str(result.output_path), media_type="audio/wav")
```

### 2.2 `hub/hub_manager.py`: model lifecycle + stats

`HubManager`:

- Loads model specs from `hub/model_registry.py`.
- Resolves worker runtime paths (repo root + venv python + PYTHONPATH) via `hub/paths.py`.
- Spawns and caches `SubprocessWorker` instances.
- Tracks basic generation statistics (count, timing, inferred device string).

Worker boot is lazy (first request starts the subprocess).

### 2.3 `custom_ui/`: frontend behavior

The UI is plain HTML/JS/CSS:

- `custom_ui/index.html`: layout and model-specific panels.
- `custom_ui/static/app.js`: client logic.
- `custom_ui/static/styles.css`: styling tokens and components.

Frontend highlights:

- Fetches models from `GET /api/models`, displays clickable cards.
- Records or uploads reference audio; stores session data in `localStorage` + `IndexedDB`.
- Submits multipart form to `POST /api/generate`.
- Supports `POST /api/unload` to unload an in-memory model.

---

## 3. Worker Architecture (Subprocess IPC)

Workers are separate Python subprocesses that:

- execute under the **model repo’s venv** (sibling repo), not the hub venv
- implement a strict **newline-delimited JSON protocol** on stdin/stdout

### 3.1 Hub ↔ worker protocol

Implemented by `hub/subprocess_worker.py` and `workers/_worker_protocol.py`.

Lifecycle:

1. Hub starts worker: `[python, "-u", worker_script]`
2. Worker prints a one-line JSON handshake: `{"ok": true, "msg": "..."}`
3. Hub sends JSON requests to stdin:
   - `{"cmd":"gen","model_id":"...","request":{...}}`
   - `{"cmd":"unload"}`
   - `{"cmd":"shutdown"}`
4. Worker responds with:
   - `{"ok": true, "result": {...}}`
   - or `{"ok": false, "error": "..."}`

### 3.2 Critical detail: stdout protection

Many ML stacks print to stdout, which would corrupt the JSON protocol. Workers solve this by redirecting `sys.stdout` to stderr and using the original stdout only for protocol messages:

```py
# workers/_worker_protocol.py
_real_stdout = sys.stdout
sys.stdout = sys.stderr

def send(obj): _real_stdout.write(json.dumps(obj) + "\n")
```

### 3.3 Worker scripts are adapters, not model implementations

Example: `workers/worker_index_tts2.py`:

- owns model singleton caching (`_tts`)
- loads from a model-specific directory (often `./checkpoints`)
- reads request fields from `request["fields"]`
- writes output into `outputs/<model_id>/...`

---

## 4. Watermark Architecture (Multiclass Attribution)

The watermark subsystem lives under `watermark/` and is packaged as a separate Python package (see `pyproject.toml`).

### 4.1 Goal and “codec simplification”

The current design is **attribution**, not general “message embedding”.

Instead of embedding/decoding a bit payload (e.g., `id + version + CRC`), the system is simplified to a **(K+1)-class classification** objective:

- `class 0`: clean / not watermarked
- `class 1..K`: model attribution classes

This eliminates CRC logic and reduces watermark “capacity” to what the product needs: “which model made it?”

### 4.2 Configuration (`watermark/config.py`)

Core audio/window constants:

- `SAMPLE_RATE = 16000`
- `SEGMENT_SECONDS = 3.0` → training segments are fixed at 3s (`SEGMENT_SAMPLES = 48000`)
- sliding-window decoder parameters:
  - `WINDOW_SAMPLES = 16000` (1s)
  - `HOP_RATIO = 0.5` (50% overlap)
  - `TOP_K = 3` (aggregation)

Multiclass constants:

```py
# watermark/config.py
WM_MODE = "multiclass"
N_MODELS = 8
N_CLASSES = N_MODELS + 1
CLASS_CLEAN = 0
```

### 4.3 Data model (manifest + dataset)

Watermark training uses a JSON manifest (array of objects). The current code expects at least:

- `path`: path to an audio file
- `has_watermark`: `0/1`
- `model_id`: `0..N_MODELS-1` for positives; `-1` or missing for clean
- `version`: optional metadata (not watermarked in multiclass mode)

Dataset implementation: `watermark/training/dataset.py`

Key behavior:

- Loads audio via `watermark/utils/io.py::load_audio` and enforces canonical `(1, T)` mono float32 @ 16k.
- Crops/pads to `SEGMENT_SAMPLES`.
- Produces a **single label**: `y_class`:
  - clean → `0`
  - watermarked with `model_id` → `model_id + 1`

Snippet:

```py
# watermark/training/dataset.py
if has_watermark >= 0.5:
    y_class = int(model_id) + 1
else:
    y_class = int(CLASS_CLEAN)
```

### 4.4 Encoder: class-conditioned FiLM + bounded perturbation

`watermark/models/encoder.py` implements:

- `WatermarkEncoder`: per-window embedder
- `OverlapAddEncoder`: overlap-add wrapper for arbitrary-length audio

Encoder design:

- Inputs:
  - `audio`: `(B, 1, T)`
  - `class_id`: `(B,)` (LongTensor)
- Conditioning:
  - `nn.Embedding(N_CLASSES, embed_dim)`
  - FiLM modulation (`gamma, beta`) per conv block from embedding
- Output:
  - bounded perturbation: `x_wm = x + alpha * tanh(delta)`

Snippet:

```py
# watermark/models/encoder.py
emb = self.class_embed(class_id)  # (B, D)
delta = self.out(...)
watermark = torch.tanh(delta)
return audio + self.alpha * watermark
```

`OverlapAddEncoder`:

- slices audio into 1s windows with 50% overlap
- encodes each window
- reconstructs using Hann weighting + fold + normalization
- guarantees output length equals input length

### 4.5 Decoder: pure-torch mel frontend + CNN + softmax head

`watermark/models/decoder.py` implements:

- `WatermarkDecoder`: window classifier
- `SlidingWindowDecoder`: clip-level wrapper with top-k aggregation

#### 4.5.1 Loc-gated decoder (loc + detect + ID)

The decoder is intentionally **loc-gated** to avoid early collapse to “clean” (class 0) and to stop attribution gradients being diluted by non-watermarked regions:

- `loc` head: localized watermarkness over the window time axis
- `detect` head: auxiliary global watermark presence (kept for compatibility and optional regularization)
- `id` head: K-way model attribution using loc-weighted pooling

The combined `(K+1)` distribution is derived as:

```py
P(clean) = 1 - P(wm_loc)
P(class=i+1) = P(wm_loc) * P(id=i)
```

This makes “detection is the major goal” explicit (and tunable via weights), while still enabling attribution once detection is reliable.

Window-level decoder:

- computes STFT + mel filterbank (no librosa; safe on MPS)
- runs a CNN backbone
- produces `loc_logits`, `detect_logit`, and `id_logits`
- derives `(K+1)` class probabilities/logits for backward compatibility

Clip-level aggregation (`SlidingWindowDecoder`):

- runs the window decoder over all windows
- selects top-k windows by **localization pooled watermark probability**
- averages their logits to produce stable clip-level outputs

Return contract (clip-level):

- `clip_detect_logit`: `(B,)`
- `clip_wm_prob` / `clip_detect_prob`: `(B,)` derived from localization pooling (same value)
- `clip_id_logits`: `(B, K)`
- `clip_id_probs`: `(B, K)`
- `clip_class_probs`: `(B, K+1)` (derived)
- `clip_class_logits`: `(B, K+1)` (derived log-probs)
- window-level outputs for training and debugging:
  - `all_window_detect_logits`: `(B, n_win)`
  - `all_window_loc_logits`: `(B, n_win, Tloc)`
  - `all_window_wm_prob_loc`: `(B, n_win)`
  - `all_window_id_logits`: `(B, n_win, K)`

Snippet:

```py
# watermark/models/decoder.py (clip-level)
top_wm_probs = torch.gather(wm_prob_loc, 1, top_idx)  # (B, k)
clip_wm_prob = top_wm_probs.mean(dim=1)
```

---

## 5. Training Stages and Losses

Training is organized as staged optimization over encoder/decoder:

- **Stage 1 (`s1`)**: decoder pretraining (encoder frozen)
- **Stage 2 (`s2_encoder`)**: train encoder only (decoder frozen)
- **Stage 3 (`s3_finetune`)**: finetune encoder + decoder together (optional)

These stages exist as functions in:

- `watermark/training/stage1.py`
- `watermark/training/stage2.py`

### 5.1 Stage 1: decoder pretraining (`watermark/training/stage1.py`)

Key idea: the dataset contains *clean carriers*, and watermark is applied **on-the-fly** to positives using the frozen encoder.

Loss:

- Detect loss (all samples): BCE on per-window + clip `wm_prob_loc` (localization pooled)
- Optional consistency: MSE between `sigmoid(clip_detect_logit)` and `clip_wm_prob` (helps keep detect head aligned)
- ID loss (positives only): CE on per-window + clip ID logits
- Total: `loss = detect_weight * loss_detect + id_weight * loss_id`

This directly matches the product decomposition:

1) detect watermark reliably  
2) if detected, attribute model ID

### 5.2 Stage 2/3: encoder training and finetuning (`watermark/training/stage2.py`)

Stage 2 (encoder only):

- encoder learns to embed class-specific watermark robustly under differentiable augmentations
- decoder remains fixed and provides gradients through attribution CE

Stage 3 (finetune):

- decoder is unfrozen
- adds an explicit **clean detect** loss on negative (clean) samples to keep calibration stable
- supports `freeze_detect_head` to prevent detector drift while improving attribution

Core losses:

- `loss_detect`: BCE on window + clip detect logits (positives)
- `loss_id`: CE on window + clip ID logits (positives)
- `loss_qual`: `CachedSTFTLoss` (multi-resolution STFT)
- `loss_budget`: `EnergyBudgetLoss` (hard limit around target dB)
- (Stage 3 only) `loss_clean_detect`: BCE on clean samples (target=0)

Multi-loss weighting:

- `UncertaintyLossWrapper` learns log-variance weights per term (Kendall et al.).

Augmentations:

- differentiable noise, EQ, volume change
- optional differentiable reverb approximation

---

## 6. Evaluation, Attacks, and Probe Metrics

### 6.1 Attacks (`watermark/evaluation/attacks.py`)

Attacks are registered in a dict `ATTACKS` and include:

- `clean`
- `noise_white_20db`
- `resample_8k`
- `mp3_128k` (best-effort using torchaudio backends)
- `aac_128k` (best-effort)
- `reverb`

All attacks are wrapped by `apply_attack_safe` which enforces:

- input/output shape `(1, T)`
- CPU execution (to avoid MPS kernel edge cases)
- preserved length (crop/pad)
- finite values

### 6.2 Probe metrics (`watermark/evaluation/probe.py`)

The “probe” is a lightweight evaluation run used during training and in dashboards. It operates on a cached set of `ProbeItem(audio, y_class)`:

- If `y_class != 0`, it first applies the encoder to the carrier (on-the-fly).
- It runs the decoder and collects:
  - binary detection score = `clip_wm_prob = 1 - P(clean)`
  - predicted class = `argmax(clip_class_logits)`

Reported metrics (core):

- Detection:
  - `mini_auc`: ROC-AUC for watermark presence using `clip_wm_prob`
  - `tpr_at_fpr_1pct`: detection TPR when threshold is set at the 99th percentile of negative scores
  - `thr_at_fpr_1pct`: that threshold (helps explain “everything predicted clean”)
- Localization pooled score diagnostics (optional):
  - `wm_prob_loc_mean`, `wm_prob_loc_pos_mean`, `wm_prob_loc_neg_mean`
- Attribution:
  - `id_acc_pos`: ID accuracy on positives (ignores detection threshold; uses argmax over ID head)
  - `wm_acc`: “product-style” accuracy on positives, after thresholding
  - `attr_acc`: “product-style” accuracy over all clips, after thresholding
- Imperceptibility / budget diagnostics (positives only):
  - `wm_snr_db_mean`: mean SNR between carrier and encoded audio (higher is quieter)
  - `wm_budget_ok_frac`: fraction meeting the configured budget target (see `BUDGET_TARGET_DB` in `watermark/config.py`)
- Confusions:
  - `confusion`: full `(K+1)×(K+1)` confusion **after thresholding at FPR=1%**
  - `confusion_attr`: `K×K` confusion on **watermarked-only** subset (no clean row/col)
- Calibration signals:
  - `pred_clean_rate`, `p_clean_pos_mean`, `p_clean_neg_mean`

Important interpretation note:

- If detection is not yet strong at `FPR=1%`, then `thr_at_fpr_1pct` becomes very high.
- That yields `pred_clean_rate` near 1.0, and `confusion` appears dominated by `0/0`.
- In that case, use `id_acc_pos` and `confusion_attr` to evaluate attribution independent of the detection threshold.

Optional reverb probe:

- `*_reverb` variants are computed by applying the `reverb` attack to the already-encoded carrier and decoding again.
- Additional attacks can be computed by passing `extra_attacks` into `compute_probe_metrics`, producing suffix keys like `tpr_at_fpr_1pct_resample_8k`.

---

## 7. Smoke Tests and “Quick Smoke Train”

There are two “smoke” concepts in this repo:

1. **Engineering-contract smoke tests** (unit tests): verify I/O shapes, attack invariants, and encoder length.
2. **Quick smoke training**: run a tiny end-to-end training loop and produce dashboard-ready JSONL logs.

### 7.1 Engineering-contract tests (`tests/smoke_test.py`)

What it verifies:

- Audio load/save contract: always `(1, T)` float32 @ target SR.
- All attacks preserve `(1, T)` and finite values.
- `OverlapAddEncoder` preserves length for multiple “awkward” sizes.

This is intended to catch the most common runtime failures early: shape drift, resample bugs, MPS padding issues, and length off-by-one errors.

### 7.2 Strict verification script (`watermark/scripts/verify_strict.py`)

What it verifies:

- `EnergyBudgetLoss` activates when watermark energy violates budget.
- Encoder/decoder multiclass output shapes match expectations.

### 7.3 Quick voice smoke training (`watermark/scripts/quick_voice_smoke_train.py`)

This is the main “quick smoke” training flow for watermarking.

Inputs:

- either:
  - a folder of real audio clips (`--source_dir`, defaults to `mini_benchmark_data`), or
  - an existing manifest (`--manifest`)

What it does:

1. Collects audio files recursively under `--source_dir`.
2. Creates a synthetic manifest in `--out/manifest.json` by alternating:
   - positive sample (watermarked): assigns `model_id` round-robin in `[0..N_MODELS-1]`
   - negative sample (clean): `model_id = -1`
   - If `--manifest` is provided, this step is skipped.
3. Builds models:
   - `encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES))`
   - `decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES))`
   - Optionally loads weights:
     - `--load_encoder /path/to/encoder.pt`
     - `--load_decoder /path/to/decoder.pt`
4. Runs stage schedule:
   - Stage 1: decoder pretrain (epochs = `epochs_s1 + epochs_s1b`)
   - Stage 2: encoder train (epochs = `epochs_s2`)
   - Stage 3: finetune (epochs = `epochs_s1b_post`)
5. Writes `metrics.jsonl` (append-only) for the live dashboard.
6. Logs a final held-out `test_probe` (attack suite controlled by `--test_attacks`, default: `resample_8k,noise_white_20db`).

Important notes:

- Many CLI flags remain for dashboard compatibility (legacy), but multiclass mode only uses a subset (epochs, probe cadence, reverb probability, etc.).
- Probe clips are cached once (center crop) to make probe evaluation stable across epochs.
- This script is intended to support a **two-phase workflow**:
  1) detection-first (focus on `detect_weight`, moderate `epochs_s2`)
  2) ID-finetune (load prior weights, increase `id_weight`, optionally `--freeze_detect_head_in_s3`)

---

## 8. Live Dashboard (Single Log + Controller)

The dashboard is implemented as a FastAPI app in `watermark/scripts/live_dashboard.py`.

It visualizes **JSONL metrics logs** written by the training scripts (`JSONLMetricsLogger`).

### 8.1 Metrics log format (`watermark/utils/metrics_logger.py`)

Each line is a JSON object. Training scripts emit (at minimum):

- `{"type":"meta", ...}` (first few lines)
- `{"type":"step", "stage":..., "epoch":..., "loss":..., ...}`
- `{"type":"epoch", "stage":..., "epoch":..., "loss":..., ...}`
- `{"type":"probe", "stage":..., "epoch":..., "mini_auc":..., "wm_acc":..., ...}`

Every event gets an automatic `ts` timestamp.

Stop signal:

- if a file named `STOP` exists next to the metrics file, `JSONLMetricsLogger` raises `KeyboardInterrupt` and training exits.

### 8.2 Single-log mode

Run:

```bash
python -m watermark.scripts.live_dashboard --log outputs/<run>/metrics.jsonl
```

The server:

- serves an HTML dashboard at `/`
- polls the log file tail via an API endpoint and renders charts in the browser (Chart.js)

Charts/metrics are multiclass-aware:

- detection score comes from the detect head (`clip_wm_prob`)
- attribution metrics shown as:
  - `id_acc_pos` (ID head accuracy on positives, independent of detection threshold)
  - `attr_acc` / `wm_acc` (thresholded “product-style” metrics)
- confusion matrices:
  - default view is `confusion_attr` (watermarked-only, K×K)
  - optional view is full `confusion` (thresholded, (K+1)×(K+1))

### 8.3 Controller mode (multi-run launcher)

If `--log` is omitted, the dashboard runs in controller mode and manages sessions under `outputs/dashboard_runs/` (configurable via `--runs_dir`).

Controller features:

- create sessions (write `session.json`)
- spawn training processes for:
  - `watermark.scripts.quick_voice_smoke_train`
  - `watermark.scripts.train_full`
- capture stdout to per-session log files
- stop runs (SIGTERM / STOP mechanism depending on session)
- view session artifacts (metrics, stdout, sample decode reports if present)

Safety:

- the controller sanitizes pasted commands and only allows known `-m watermark.scripts.<...>` modules.
- it strips user-provided `--out/--output/--log_metrics` so it can manage paths under its runs directory.
- it also strips whitespace-only “positional” tokens from pasted commands (prevents a common paste error where `"\n"` becomes an “unrecognized argument”).

---

## 9. Outputs, Artifacts, and File Conventions

### 9.1 Hub outputs

- Upload staging: `outputs/uploads/<task_id>/prompt.wav`, `emo.wav`, etc.
- Model outputs: `outputs/<model_id>/...` (generated by workers)

### 9.2 Watermark outputs

Typical run directory contains:

- `manifest.json` (if generated by quick smoke, copied from source for self-containment)
- `config.json` (full CLI args + derived settings)
- `metrics.jsonl` (dashboard log)
- `encoder.pt` / `decoder.pt` (best model weights, always saved for reuse)
- `checkpoints/` directory (durable checkpoint artifacts):
  - `last.pt` - crash-safe checkpoint (always overwritten)
  - `best.pt` - best by chosen metric
  - `best_meta.json` - metric name/value/epoch/stage
  - `last_meta.json` - epoch/stage + minimal info

**Note**: Both `quick_voice_smoke_train.py` and `train_full.py` now produce the same durable artifacts for consistent runbook workflows.

---

## 10. Extending the System

### 10.1 Adding a new hub model

1. Add a `ModelSpec` entry in `hub/model_registry.py`.
2. Update `hub/paths.py` to map the `model_id` to:
   - sibling repo directory
   - venv python path
   - PYTHONPATH list
3. Implement a worker adapter in `workers/worker_<name>.py`:
   - handshake on startup
   - handle `{cmd:"gen"}` requests
   - write outputs to `outputs/<model_id>/...`

### 10.2 Adding a new watermark attribution class

1. Increment `N_MODELS` in `watermark/config.py` (or refactor to dynamic mapping).
2. Ensure your manifest uses `model_id` in `[0..N_MODELS-1]` for positives.
3. Retrain (class embedding + decoder head dimensions depend on `N_CLASSES`).

### 10.3 Changing the watermark detection score

Current watermark score is:

- `clip_wm_prob = sigmoid(clip_detect_logit)` (detect head)

This is intentionally decoupled from attribution so you can prioritize Goal 1 (detection) without starving Goal 2 (ID).

---

## 12. Recent Runs (Progress Log)

This section captures “known good / known bad” runs observed during iteration on MPS.

### 12.1 Detection-first runs

- `outputs/dashboard_runs/1769406908_b1389f` (1024 clips; detect-heavy weights + reverb probing)
  - Best `mini_auc_reverb≈0.896` and `tpr@fpr1%_reverb≈0.562` (S1 e4).
  - Takeaway: Stage 1 can produce a strong detector quickly; later stages can still regress if encoder training is not well balanced.
- `outputs/dashboard_runs/1769409052_f961e5` (512 clips; detect-first small)
  - Best `mini_auc_reverb≈0.892` and `tpr@fpr1%_reverb≈0.461` (S1 e4/e6).
  - Stage 2/3 reduced detection metrics vs the S1 peak.
  - Takeaway: decoder learns an easy-to-detect pattern, but encoder training can learn a less robust watermark under the quality/budget constraints.

### 12.2 ID-finetune run (loaded from detection-first)

- `outputs/dashboard_runs/1769409407_ee6038` (S3-only finetune; loaded encoder/decoder + manifest)
  - `id_acc_pos` improved vs chance (best ≈0.293 where chance is 0.125 for 8 IDs),
    but detection robustness dropped (best `tpr@fpr1%_reverb≈0.262`).
  - Takeaway: this run is a valid “Goal 2 attempt”, but Goal 1 degraded; next iterations should keep more detection pressure during finetune and/or increase Stage 2 detection-first epochs before ID finetune.

### 12.3 Known failure mode: thresholded confusion collapses to 0/0

When `thr_at_fpr_1pct` becomes extremely high (because the detector is weak), almost everything is predicted clean:

- `pred_clean_rate → 1.0`
- full `confusion` dominated by `0/0`

In this regime, prefer `confusion_attr` and `id_acc_pos` to judge attribution signal.

---

## 11. Legacy Code

Legacy, bit-payload watermarking code is preserved under `watermark/legacy/` and is not part of the primary training path.

Compatibility shims exist:

- `watermark/models/codec.py` re-exports the old `MessageCodec` for older experiments.
- `watermark/scripts/quick_smoke_train.py` is a stub that points to the new smoke run and the legacy module.

If you are reading older docs or logs that mention “CRC”, “payload bits”, or “preamble bits”, those refer to the legacy path.
