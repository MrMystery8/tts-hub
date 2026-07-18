# Deterministic Codebook vs Learned Encoder (S1-only) — Findings & Recommendations

This document summarizes observed behavior in TTS Hub + dashboard runs, and compares:

1. **Learned encoder architecture (current default)** used in many dashboard runs where Stage 1 trains the decoder and the encoder is **random/frozen** (S1-only), and
2. **Deterministic codebook + decoder-only** (new approach on branch `deterministic-codebook`).

It also records why early deterministic-codebook runs performed poorly and what changed to fix them.

## Executive summary

- **S1-only learned-encoder runs are unstable across IDs** because the encoder is randomly initialized and frozen: some IDs become “strong”, some “weak/dead”. In TTS Hub this shows up as flaky detection and “ID drift” (the model embeds one ID but the detector predicts another).
- **Deterministic codebook + decoder-only is stable** (given a fixed `codebook_key`) and removes the “dead ID” problem because all IDs are constructed to have equal strength by design.
- Early deterministic runs were bad because the codewords were **noise-like bandpass** signals that are **magnitude-spectrum indistinguishable**, while the decoder uses mel/magnitude features. Switching the codebook to **multitone** (magnitude-spectrum distinct per ID) fixed attribution.

## What “architecture” means here

### A) Learned encoder (S1-only) — prior 20k run

- Encoder: `WatermarkEncoder` (FiLM-conditioned on class_id) wrapped by `OverlapAddEncoder`.
- Decoder: `WatermarkDecoder` wrapped by `SlidingWindowDecoder`.
- Training behavior in S1-only:
  - **Decoder is trained**.
  - **Encoder is frozen** and acts as a synthetic watermark generator.

This is the setup used by the 20k S1-only run:
- Run: `outputs/dashboard_runs/1769924317_97e1dc`
- Stages: `epochs_s1=100`, `epochs_s2=0`, `epochs_s1b_post=0`

### B) Deterministic codebook + decoder-only — new approach

- Encoder: `DeterministicCodebookEncoder` (non-trainable) created from a `codebook_key` and configuration, and exported as `codebook.json`.
- Decoder: same decoder as above.
- Training behavior:
  - **Decoder is trained**.
  - Encoder is **not trained**.

This setup produced the latest strong run:
- Run: `outputs/dashboard_runs/1770012678_514ceb`
- `encoder_mode=deterministic_codebook`
- `codebook_style=multitone`

## Key observed problems (learned encoder, S1-only)

### 1) “ID drift” and inconsistent mapping in TTS Hub

S1-only learned-encoder runs can show “ID drift” because the encoder’s class conditioning is random/frozen.
If class_id 2 happens to produce a weak watermark while class_id 7 produces a strong watermark, the detector will more often predict 7 because it is the one it can reliably see.

This is why “mapping” was required in some runs: not as a *training fix*, but as a way to pick stable/strong IDs in a given run.

### 2) Practical usability issues

- Two runs trained on the same data with the same S1 settings can behave dramatically differently in TTS Hub.
- Adding more clips (2k → 20k) does not guarantee better hub behavior if the “IDs you care about” are weak codewords in that run’s frozen encoder.

## Why early deterministic-codebook runs performed badly

Early deterministic-codebook runs used a **noise-bandpass** signature per ID.

That looks different in phase/time-domain, but the decoder’s frontend is mel/magnitude-based, so what matters is the **magnitude spectrum**.
We measured that the magnitude spectra across IDs were almost identical (cosine similarity ~0.996 between different IDs), meaning:

- Detection can be good (watermark present vs clean),
- Attribution is near-random because the decoder cannot distinguish IDs reliably.

Additionally, the initial scaling logic produced a louder watermark than intended in some runs (e.g. `wm_snr_db_mean ~26.6dB` vs a -30dB target), and `wm_budget_ok_frac` was often 0.0.

## What changed to get the new “good” results

### 1) Codewords became magnitude-spectrum distinct (`codebook_style=multitone`)

We switched the deterministic signatures to a **multitone** codebook:

- Each ID gets a small set of distinct tone frequencies (within a bandpass).
- This makes different IDs easy to separate from mel/magnitude features, which is what the current decoder uses.

### 2) Budget scaling was corrected (global scaling after overlap-add)

We changed scaling so the final watermark delta meets the target budget **after** overlap-add normalization.
This avoids skew introduced by per-window scaling before overlap-add.

### 3) Dashboard + hub support for deterministic runs

- Training exports `codebook.json` for deterministic runs.
- Hub can now load a run that has `decoder.pt` + `codebook.json` (no learned encoder required).
- Run details show encoder mode + key/style in the hub UI.

## Are the “perfect” results real?

They are real for what the benchmark is measuring:

- Splits are enforced (`train/val/test` manifests) and the final `test_probe` evaluates on the held-out `test` split.
- The reported metrics come from `test_probe` for `outputs/dashboard_runs/1770012678_514ceb`.

However, note the scope:

- The benchmark is **carrier audio** from `medium_benchmark_data` with a synthetic watermark applied.
- It is not yet an evaluation on actual TTS outputs from IndexTTS2/Chatterbox pipelines.

Recommendation: validate on “real hub outputs” by generating a small corpus from each TTS model, watermarking it, and running detection on those files.

## Security / robustness / quality tradeoffs (high-level)

### Detection accuracy & usability

- **Deterministic codebook (multitone)**: tends to be excellent for clean/reverb conditions and stable across runs (good usability).
- **Learned encoder (S1-only)**: can be inconsistent across IDs/runs due to frozen random codewords (poor usability).

### Audio quality

- Learned encoder (if it trains well) can potentially hide energy more perceptually.
- Multitone deterministic codewords are **structured (a few stable tones)** and can therefore be **more audible** (often perceived as “ringing/whistling”), especially when we keep the band under ~4kHz for `resample_8k` robustness.

Mitigations implemented on this branch:

- `codebook_style=subband_noise`: uses many phase-randomized tones per ID (noise-like), which is typically less “painful” than a few stable tones at the same budget.
- `peak_limit` safety: the deterministic encoder scales the watermark down when the carrier is already near full-scale to avoid clipping harshness.

Practical tuning guidance:

- If it still sounds too loud, lower `codebook_budget_target_db` (try `-40` or `-45`) and retrain the decoder.
- If you need `resample_8k` robustness, keep `codebook_bandpass_high_hz <= 3800` (otherwise the attack deletes the watermark energy).

### Robustness to attacks

Deterministic multitone codewords can be vulnerable to targeted spectral attacks (notches, filtering, codec artifacts).
If robustness becomes a priority, consider:

- keyed frequency hopping (still deterministic, but spreads energy),
- multi-band signatures,
- training the decoder with attacked inputs (decoder-only augmentation),
- later: a learned encoder that is trained *successfully* under attack objectives.

### Forgery resistance

Deterministic designs require careful key management. If an attacker learns the `codebook_key` + scheme, forgery/removal becomes easier.

## ID scaling (2 IDs now, 8 later)

Current training and decoder head are still `N_MODELS=8` (8 attribution IDs, plus clean).

- The deterministic codebook inherently supports 8 IDs without rewiring.
- To use 8 IDs in TTS Hub, the hub needs mapping support for all 8 TTS models you care about.
- For best performance on 8 IDs in real-world hub audio, you should train/evaluate with all 8 represented (which the current manifest generator already does).

## Recommended next steps

1. **Short run validation** (2k clips) on medium data with multitone codebook:
   - Confirm `test_probe` behavior.
   - Confirm in TTS Hub that IndexTTS2 and Chatterbox outputs are detected reliably.
2. **Attack evaluation (no training changes yet)**:
   - Run with `--test_attacks resample_8k,noise_white_20db,mp3_64k` (or similar) to quantify robustness.
3. **Attack training (decoder-only augmentation)**:
   - Add a new `--train_attacks ...` pipeline that applies randomized attacks to watermarked inputs before decoding during Stage 1.
   - This is the next planned feature; once implemented, we can provide the canonical dashboard command.
