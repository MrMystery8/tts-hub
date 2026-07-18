# FYP Report — Depth-Gap Remediation Plan

Target file: `Ayaan Minhas-TP077859-APD3F2511CS(AI)-FYP Final Report (DRAFT).docx`
Method: edit via `python-docx`, insert after the anchor paragraph of each named heading; reuse existing styles (Normal, Heading 3/4, `Table Grid`, `Caption`).

Current measured word counts (verified): 4.6 = 62+152+69+144+84+199 ≈ 710w · 5.2.3 = 79w, no table · 6.1 = 488w · 4.4.2 = 3 specs (Tables 4.1–4.3).

---

## P1 — 4.6 Model Implementation (710w → ~2,000w)  [highest marks value]

Grounded in real repo facts, do NOT invent numbers.

**4.6.3 Reference preprocessing (84w → ~350w + table)**
Source of truth: `watermark/utils/io.py` (`load_audio`), `watermark/training/dataset.py`, `watermark/config.py`.
Add a numbered pipeline + a new table "Reference preprocessing stages":
resample to `SAMPLE_RATE = 16000` mono canonical `(1, T)` → crop/pad to `SEGMENT_SAMPLES = 48000` (3.0 s; random crop when training, centre crop at eval) → per-clip normalisation → manifest label assignment (`has_watermark`, `model_id` → `y_class`, class 0 = clean). Justify each choice against the assistive use case (short, imperfect user recordings).

**New 4.6.4 Watermark training regime** (promote current 4.6.4 to 4.6.5)
Add prose + a **hyperparameter table** from `watermark/training/stage1.py`, `train_full.py`, `config.py`:

| Item | Value | Source |
|---|---|---|
| Stage | Stage 1 only (frozen encoder, decoder pretrain) | stage1.py |
| Epochs | 100 | Run B |
| Optimiser / LR | AdamW, 3e-4 | stage1.py |
| detect_weight / id_weight | 12 / 2 | Run B |
| loc_consistency_weight | 0.1 (`LOC_CONSISTENCY_WEIGHT`) | config.py |
| Classes | K+1 = 4 used (K=3 backends); `N_MODELS` capacity 8 | config.py |
| Segment / window / hop | 3.0 s / 1.0 s / 50 % overlap | config.py |
| Aggregation | Top-K = 3 windows; loc Top-M = 8 | config.py |
| Encoder | hidden 32, GroupNorm groups 4, `x_wm = x + α·tanh(δ)` | encoder.py |
| Decoder | n_fft 512, n_mels 80, two heads (detect + ID) + loc head | decoder.py |
| Energy budget | −30 dB (δ power ≤ 0.1 % of carrier) | config.py |
| Data split | 5,120 clips → 4,096 / 512 / 512 | Run B |

Then ~250w on the **loss formulation**: window-level BCE + 0.5 × clip-level BCE for detection; K-way cross-entropy on positives only; localisation-consistency term; `EnergyBudgetLoss` (hard ReLU above target) and `CachedSTFTLoss` (multi-resolution, fft 512/1024/2048, spectral convergence + log-magnitude L1) as the imperceptibility objectives; note `UncertaintyLossWrapper` (Kendall et al.) is available. Explain *why* on-the-fly embedding of positives avoids a stale watermarked corpus.

**4.6.2 Worker adapter integration (144w → ~300w + table)**
Add a per-backend control table (params exposed per worker) from `workers/worker_index_tts2.py`, `worker_chatterbox_mtl.py`, `worker_qwen3_tts_mlx.py`: reference audio, text, language, seed, emotion/exaggeration controls, CFG/temperature, max tokens, device (MPS vs MLX). One row per backend × parameter; state the defaults actually shipped.

**4.6.1** add 2–3 sentences on dataset provenance: LibriSpeech `dev-clean` → `medium_benchmark_data`, manifest JSONs, speaker-disjoint intent, why LibriSpeech is a fair carrier corpus.

## P2 — 5.2.3 System Testing (79w → paragraph + table)
Add **Table 5.x: End-to-end system test scenarios**, 4 columns to match the unit-test table redesign (`Step / Action / Expected result / Status`), covering the full workflow as numbered steps: launch hub offline (airplane mode) → create voice from reference → save voice → generate on each of the 3 backends → watermark auto-embed → detect + attribute → history/persistence → mobile PWA delegation over Tailscale → restart and reload state. Add a one-line note on environment (MacBook Air M4, 24 GB, offline).

## P3 — Chapter 6 Conclusion (865w → ~1,400w)
Restructure 6.1 into:
1. Restatement (keep existing 153w opener).
2. **New table: objective → achieved / partially achieved → evidence** (one row per Ch1 objective, evidence = specific figures: AUC 0.988, TPR@1%FPR 0.795, attr 0.875, PESQ 4.64, STOI 1.00, SNR 21.5 dB, test suites, UAT).
3. Keep the honest "success is uneven" paragraph (good — markers reward it).
4. **New 6.1.1 Contribution to community and industry** (~200w): offline assistive AAC users, privacy-preserving alternative to cloud cloning, provenance/attribution as a deepfake-accountability contribution, SDG 3.
5. **New 6.1.2 Strengths of the system** (~150w): unified multi-backend hub, process isolation, transparent watermark, reproducible eval harness.

## P4 — 4.4.2 Use Case Specifications (3 → 6)
Add three more in the same table format/style as Tables 4.1–4.3, renumbering downstream captions and cross-references:
- UC-xx Manage/delete stored voice data (privacy — ties to interview findings)
- UC-xx Compare backends / benchmark run
- UC-xx Mobile companion delegation over private network

## P5 — 4.5 Interface Design (deferred)
Blocked on real screenshots (Figs 4.12–4.20). Once captured, expand each screen to element-by-element prose (layout regions, controls, states, error/empty states, accessibility notes) per the sample report's depth.

---

## Execution order & renumbering risk
P1 → P2 → P3 → P4 (P4 last: it renumbers Tables 4.4+ and their in-text references). After every insertion, re-run a caption-sequence check before moving on. Final step remains updating TOC / List of Figures / List of Tables fields in Word.

## Open data needed from you
- Run B batch size / scheduler (not persisted in any `outputs/*/config.json` found) — needed for the hyperparameter table.
- Confirmation of exact shipped defaults per backend worker.
