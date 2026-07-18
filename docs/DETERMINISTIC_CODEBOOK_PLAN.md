# Deterministic Codebook + Decoder-Only Watermarking (Branch Plan)

This document tracks the implementation of the deterministic codebook watermarking approach on the `deterministic-codebook` branch.

## Goals (Phase 1)

1. **Stable model attribution IDs across runs**
   - IndexTTS2 and Chatterbox multilingual must always embed the same attribution IDs across retrains.
2. **Decoder-only training**
   - Remove dependence on a learned encoder for the baseline system; the encoder becomes a deterministic embedder with a fixed codebook.
3. **Hub + Dashboard parity**
   - Training runs created via the dashboard can be selected in TTS Hub and used for post-generation watermarking and detection.

## Non-goals (Phase 1)

- Robustness to strong adversarial attacks (codec, heavy reverb, time-stretch, etc.). We will still report metrics under a small suite of attacks, but we will not optimize for them yet.
- Protection against watermark removal.

## Design Overview

### Deterministic encoder

- The encoder is not trained.
- For each attribution ID (pred_model_id in `0..K-1`), we define a fixed **time-domain signature** (codeword) at 16kHz.
- Embedding is overlap-add:
  - Split the audio into 1s windows (same as existing `WINDOW_SAMPLES` / `HOP_RATIO`).
  - Add `delta = scale(audio_window) * signature[id]` per window.
  - Use Hann windowing and overlap-add to avoid boundary artifacts.

### Equal-strength constraint (by construction)

- Every signature is normalized to equal RMS/energy.
- The per-window embedding scale is set from a shared budget target (e.g., `BUDGET_TARGET_DB`) relative to the carrier window energy.
- This makes all IDs equally detectable in expectation and avoids “dead IDs” caused by random class embeddings.

### Decoder-only training

- Stage 1 remains the main training loop, but the on-the-fly watermarking step uses the deterministic encoder instead of the learned encoder.
- Stages 2/3 are disabled for this training mode.

## Run Artifacts / Hub Compatibility

### Required files in a run directory

- `decoder.pt` (required)
- `codebook.json` (required for deterministic mode; includes key, K, windowing params, budget target, etc.)
- Optional:
  - `hub_mapping.json` (TTS model -> pred_model_id mapping override)
  - `metrics.jsonl`, `report.md`, `session.json` (same as current)

### Hub behavior

- If a run has `codebook.json`, the hub uses the deterministic encoder for embedding.
- Detection uses `decoder.pt` (same as current), and attribution uses the decoder output.

## Dashboard changes

- Add an “Encoder mode” selector: `learned` (existing) vs `deterministic_codebook`.
- When deterministic mode is selected:
  - `epochs_s2` and `epochs_s1b_post` are forced to `0` in the dashboard command generator.
  - The run exports `codebook.json` instead of `encoder.pt` (or in addition, if needed for backward-compat).

## Validation / Acceptance Criteria

1. **Unit tests**
   - Deterministic codebook generation is stable across processes.
   - Embedding respects budget normalization (within tolerance).
2. **Smoke test**
   - Hub can embed and detect watermark using a deterministic run (IndexTTS2 + Chatterbox multilingual).
3. **Sanity metrics**
   - On held-out medium benchmark:
     - Lower false negatives for the two mapped IDs than the current S1-only “random encoder” behavior.
     - ID attribution for mapped IDs is consistent across restarts/runs.

## Implementation Steps

1. Implement deterministic codebook generator and encoder (MPS-safe).
2. Add a training flag to `quick_voice_smoke_train.py` to use deterministic mode and export `codebook.json`.
3. Update `live_dashboard.py` to surface deterministic mode + key and generate correct commands.
4. Update `hub/watermark_service.py` to accept deterministic runs (no `encoder.pt` required).
5. Add tests + run the existing smoke integration script.

