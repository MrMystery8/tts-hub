# Watermark Pipeline Diagnostics (Smoke + Voice Smoke)

**Date**: 2026-01-23 → 2026-01-24  
**Repo**: `tts-hub`  
**Scope**: Documents the end-to-end watermark pipeline smoke tests we ran, the observed issues (errors/warnings), what the results mean, and what remains to be fixed.

## What We Tested

### 1) Synthetic smoke test (sanity / hearing check)
- **Script**: `watermark/scripts/quick_smoke_train.py`
- **Purpose**: Verify Stage 1 → Stage 1B → Stage 2 runs end-to-end; save a clean vs watermarked WAV to do a quick listening test.
- **Run command**:
  - `.venv/bin/python -m watermark.scripts.quick_smoke_train`
- **Artifacts**:
  - `outputs/quick_smoke/audio/clean.wav`
  - `outputs/quick_smoke/audio/watermarked.wav`
  - `outputs/quick_smoke/audio/watermarked_reverb.wav`

### 2) Real voice smoke test (mini_benchmark_data)
- **Script**: `watermark/scripts/quick_voice_smoke_train.py`
- **Dataset**: `mini_benchmark_data/**/*.flac`
  - No manual conversion required: `watermark/utils/io.py::load_audio()` uses `soundfile` and supports FLAC; clips are resampled to `SAMPLE_RATE` and saved as WAV for listening.
- **Purpose**:
  - Run Stage 1 → Stage 1B → Stage 2 on real speech.
  - Save clean vs watermarked vs attacked (reverb) WAVs.
  - Produce a `decode_report.txt` with detection + payload diagnostics (AUC, preamble stats, payload accuracy).
- **Example run command**:
  - `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 8 --epochs_s1 1 --epochs_s1b 1 --epochs_s2 1`
- **Artifacts**:
  - `outputs/quick_voice_smoke/audio/clean.wav`
  - `outputs/quick_voice_smoke/audio/watermarked.wav`
  - `outputs/quick_voice_smoke/audio/watermarked_reverb.wav`
  - `outputs/quick_voice_smoke/audio/source.txt` (exact clip used for the listening pair)
  - `outputs/quick_voice_smoke/audio/decode_report.txt`

## Issues Observed During Runs

### A) CLI usage error (expected)
This occurred due to splitting the command across lines:
- Error:
  - `argument --source_dir: expected one argument`
  - `zsh: command not found: mini_benchmark_data`
- Fix: pass the directory on the same line:
  - `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data ...`

### B) PyTorch STFT resize warning on MPS (non-blocking)
We saw repeated warnings like:
- `UserWarning: An output with one or more elements was resized... (Triggered internally ... Resize.cpp)`

Interpretation:
- This is a PyTorch warning (commonly seen with `torch.stft` on some backends). It did not crash the run and can be treated as noise for now.

### C) RuntimeError: STFT input/window on different devices (blocking, fixed)
Earlier we hit:
- `RuntimeError: stft input and window must be on the same device but got self on cpu and window on mps:0`

Root cause:
- A “CPU attack” (reverb) produced a CPU tensor, but decoding was done with a decoder on MPS, and the cached STFT Hann window buffer lived on MPS.

Fix applied:
- `watermark/scripts/quick_voice_smoke_train.py` now forces decode inputs onto the decoder device in `summarize_decode()`.
- For robustness AUC under reverb, it now explicitly moves attacked audio back to model device before decoding.

## Results (From Saved `decode_report.txt`)

Notes on interpretation:
- `mini_auc`: AUC on a small set comparing **clean** vs **watermarked** detection scores. `1.0` is perfect.
- `mini_auc_reverb`: same, but after applying a **reverb attack**.
- `preamble_*`: fraction of correct preamble bits (16-bit match). Random guessing is ~`0.5`. Predicting all-zeros yields `(#zeros_in_preamble)/16` (≈`0.56` for the current keyed preamble), so don’t treat `~0.56` on negatives as “good”.
- Payload metrics are computed **only on positives**:
  - `model_id_acc` chance ≈ `1/8 = 0.125`
  - `version_acc` chance ≈ `1/16 = 0.0625`
  - `payload_exact_acc` chance ≈ `1/(8*16) = 0.0078`
- If present, `*_acc_cls` metrics come from the decoder’s classification heads (`avg_model_logits` / `avg_version_logits`) and are the preferred attribution signal.

### Run A — `outputs/quick_voice_smoke_dev` (32 clips)
- Source: `outputs/quick_voice_smoke_dev/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_dev/audio/decode_report.txt`:
  - `mini_auc=0.5508`, `mini_auc_reverb=0.5234`
  - `preamble_pos_avg=0.99`, `preamble_neg_avg=1.00` (decoder essentially output “preamble matches” for everything)
  - `payload_bit_acc=0.808` (later determined to be misleading when message fields are not varied / dominated by constants)

### Run B — `outputs/quick_voice_smoke_dev2` (32 clips)
- Source: `outputs/quick_voice_smoke_dev2/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_dev2/audio/decode_report.txt`:
  - `mini_auc=0.7852`, `mini_auc_reverb=0.6016`
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=1.00` (still broken separation)
  - `model_id_acc=0.250`, `version_acc=0.062` (payload mostly weak; version ~chance)

### Run C — `outputs/quick_voice_smoke_dev3` (32 clips)
- Source: `outputs/quick_voice_smoke_dev3/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_dev3/audio/decode_report.txt`:
  - `mini_auc=0.7344`, `mini_auc_reverb=0.6094`
  - `preamble_pos_avg=0.56`, `preamble_neg_avg=0.56` (decoder collapsing to “all zeros” for preamble bits)
  - `model_id_acc=0.125`, `version_acc=0.062` (≈chance)

### Run D — `outputs/quick_voice_smoke` (256 clips)
- Source: `outputs/quick_voice_smoke/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke/audio/decode_report.txt`:
  - `mini_auc=0.9971` (**detection is excellent**)
  - `mini_auc_reverb=0.9456` (**reverb robustness is strong on average**)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.58` (positives match preamble, negatives do not reliably match)
  - Payload attribution is still near chance:
    - `payload_exact_acc=0.062`
    - `model_id_acc=0.125` (≈chance)
    - `version_acc=0.078` (near chance)
    - `payload_bit_acc=0.524` (≈chance)

### Run E — `outputs/quick_voice_smoke_medium_cls` (96 clips, classification heads enabled)
- Source: `outputs/quick_voice_smoke_medium_cls/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_medium_cls/audio/decode_report.txt`:
  - `mini_auc=0.9952`, `mini_auc_reverb=0.9106` (detection still strong)
  - `preamble_pos_avg=0.56`, `preamble_neg_avg=0.56` (preamble collapsed → decision-rule gating will not work)
  - `model_id_acc_cls=0.125` (chance), `version_acc_cls=0.000` (worse than chance; often predicting the “unknown” class)

### Run F — `outputs/quick_voice_smoke_medium_ce3` (96 clips, stronger CE weights)
- Source: `outputs/quick_voice_smoke_medium_ce3/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_medium_ce3/audio/decode_report.txt`:
  - `mini_auc=0.9918`, `mini_auc_reverb=0.9310` (detection remains strong)
  - `preamble_pos_avg=0.59`, `preamble_neg_avg=0.56` (tiny separation, still mostly collapsed)
  - `model_id_acc_cls=0.167` (slightly above chance), `version_acc_cls=0.104` (above chance)

### Run G — `outputs/quick_voice_smoke_medium_ce3b` (96 clips, after “neg preamble → 0.5” change)
- Source: `outputs/quick_voice_smoke_medium_ce3b/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_medium_ce3b/audio/decode_report.txt`:
  - `mini_auc=0.9865`, `mini_auc_reverb=0.9375` (detection remains strong; reverb separation is very clean)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.49` (preamble collapse fixed; negatives pushed toward uncertainty)
  - `model_id_acc_cls=0.292`, `version_acc_cls=0.104` (identity improving but still not “production reliable”)

### Run H — `outputs/quick_voice_smoke_large_ce3b` (160 clips, larger sample)
- Source: `outputs/quick_voice_smoke_large_ce3b/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_large_ce3b/audio/decode_report.txt`:
  - `mini_auc=0.9711`, `mini_auc_reverb=0.9073` (still strong separation, but lower than the medium run)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.65` (negatives are drifting toward “preamble match” again)
  - `model_id_acc_cls=0.275`, `version_acc_cls=0.113` (above chance, but still far from reliable attribution)

### Run I — `outputs/quick_voice_smoke_large_ce3c` (160 clips, +post Stage1B fine-tune, higher neg_weight)
- Source: `outputs/quick_voice_smoke_large_ce3c/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_large_ce3c/audio/decode_report.txt`:
  - `mini_auc=0.9478`, `mini_auc_reverb=0.8639` (detection AUC dropped a bit; reverb robustness also dropped)
  - `preamble_pos_avg=0.99`, `preamble_neg_avg=0.36` (much better preamble separation; fewer false preamble hits on negatives)
  - `model_id_acc_cls=0.325`, `version_acc_cls=0.225` (identity improved vs Run H; still not fully reliable)

### Run J — `outputs/quick_voice_smoke_512_ce3` (512 clips, bigger run)
- Source: `outputs/quick_voice_smoke_512_ce3/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_512_ce3/audio/decode_report.txt`:
  - `mini_auc=0.9734`, `mini_auc_reverb=0.8758`
  - `preamble_pos_avg=0.99`, `preamble_neg_avg=0.89` (bad: many negatives look like “preamble match” → high false preamble)
  - `model_id_acc_cls=0.480`, `version_acc_cls=0.309` (identity substantially improved vs earlier runs)

### Run K — `outputs/quick_voice_smoke_512_ce3_neg5` (512 clips, higher negative preamble weight)
- Source: `outputs/quick_voice_smoke_512_ce3_neg5/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_512_ce3_neg5/audio/decode_report.txt`:
  - `mini_auc=0.9904`, `mini_auc_reverb=0.9014`
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.56` (preamble false positives reduced substantially vs Run J)
  - `model_id_acc_cls=0.418`, `version_acc_cls=0.242` (still above chance; slightly lower than Run J)

### Run L — `outputs/quick_voice_smoke_2k_ce3_neg5` (2048 clips, scaled up)
- Source: `outputs/quick_voice_smoke_2k_ce3_neg5/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_2k_ce3_neg5/audio/decode_report.txt`:
  - `mini_auc=0.9398`, `mini_auc_reverb=0.8220` (detection/reverb AUC dropped vs 512-clip runs)
  - `preamble_pos_avg=0.98`, `preamble_neg_avg=0.53` (preamble separation still OK; negatives near random)
  - `model_id_acc_cls=0.468`, `version_acc_cls=0.429`, `payload_exact_acc_cls=0.276` (best identity so far; far above chance)

### Run M — `outputs/quick_voice_smoke_4k_ce3_neg5_r05` (2703 clips, higher reverb)
- Source: `outputs/quick_voice_smoke_4k_ce3_neg5_r05/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_4k_ce3_neg5_r05/audio/decode_report.txt`:
  - `mini_auc=0.9998`, `mini_auc_reverb=0.9787` (excellent detection, strong robustness to reverb on average)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.25` (preamble separation improved substantially)
  - `model_id_acc_cls=0.127`, `version_acc_cls=0.072`, `payload_exact_acc_cls=0.002` (identity collapsed to chance)

### Run N — `outputs/quick_voice_smoke_512_dashboard` (512 clips, dashboard run)
- Source: `outputs/quick_voice_smoke_512_dashboard/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_512_dashboard/audio/decode_report.txt`:
  - `mini_auc=0.9944`, `mini_auc_reverb=0.8871` (strong detection/robustness)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.69` (false preamble too high vs target ≤0.60)
  - `model_id_acc_cls=0.160`, `version_acc_cls=0.129`, `payload_exact_acc_cls=0.043` (identity only slightly above chance)
- Additional insight from per-epoch probes in `outputs/quick_voice_smoke_512_dashboard/metrics.jsonl`:
  - During Stage 2, preamble separation was good (`preamble_neg_avg≈0.263`) and attribution peaked at `payload_exact_acc_cls≈0.117`.
  - The original post-Stage2 fine-tune (decoder unfreezing) degraded preamble separation and reduced final attribution.

### Run O — `outputs/quick_voice_smoke_2k_dashboard` (2048 clips, bigger dashboard run)
- Source: `outputs/quick_voice_smoke_2k_dashboard/audio/source.txt`
- Key metrics from `outputs/quick_voice_smoke_2k_dashboard/audio/decode_report.txt`:
  - `mini_auc=0.9935`, `mini_auc_reverb=0.9815` (very strong detection + reverb robustness by AUC)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.55` (better than Run N; still close to the failure band)
  - `model_id_acc_cls=0.144`, `version_acc_cls=0.071`, `payload_exact_acc_cls=0.043` (identity still near chance)
- Additional insight from per-epoch probes in `outputs/quick_voice_smoke_2k_dashboard/metrics.jsonl`:
  - **Strict operating point is still weak**: `tpr_at_fpr_1pct≈0.3125`, `tpr_at_fpr_1pct_reverb≈0.2109`.
  - Confusion matrices show a near-constant prediction collapse during Stage 2 (“presence-only watermark”).

### Run P — `outputs/quick_voice_smoke_1k_dashboard_v2` (1024 clips, full 128-pair grid)
- Source: `outputs/quick_voice_smoke_1k_dashboard_v2/audio/source.txt`
- This run uses **all 128 (model_id, version) pairs** uniformly (each pair repeated 4× across positives), so `payload_exact_acc*` is meaningful for the real attribution objective.
- Key metrics from `outputs/quick_voice_smoke_1k_dashboard_v2/audio/decode_report.txt`:
  - `mini_auc=0.9881`, `mini_auc_reverb=0.9172` (strong detection, decent robustness)
  - `preamble_pos_avg=0.99`, `preamble_neg_avg=0.47` (good preamble separation; low false preamble)
  - `model_id_acc_cls=0.238`, `version_acc_cls=0.299`, `payload_exact_acc_cls=0.059` (attribution improved vs Run O, still far from target)
- Additional insight from `outputs/quick_voice_smoke_1k_dashboard_v2/metrics.jsonl`:
  - **Strict detection confidence improved a lot**: final `tpr_at_fpr_1pct≈0.8047` (vs `≈0.3125` in Run O).
  - Model/version confusion matrices show less “single-class collapse” compared to Run O, but accuracy is still low (lots of off-diagonal mass).

### Run Q — `outputs/quick_voice_smoke_full2703_v2` (2703 clips, full 128-pair grid)
- Source: `outputs/quick_voice_smoke_full2703_v2/audio/source.txt`
- This run uses the full 2703-clip set (≈50/50 pos/neg), and positives cover **all 128 (model_id, version) pairs** (10–11 examples per pair).
- Config snapshot (from `outputs/quick_voice_smoke_full2703_v2/metrics.jsonl` + manifest):
  - Epochs: `s1=12`, `s1b=12`, `s2=30`, `s1b_post=20`
  - Augmentation: `reverb_prob=0.25`
  - Loss weights: `model_ce_weight=3`, `version_ce_weight=3`, `unknown_ce_weight=0`, `neg_weight=5`, `neg_preamble_target=0.5`
- Key metrics from `outputs/quick_voice_smoke_full2703_v2/audio/decode_report.txt`:
  - `mini_auc=0.9825`, `mini_auc_reverb=0.9055` (strong detection, decent robustness)
  - `preamble_pos_avg=0.97`, `preamble_neg_avg=0.16` (good preamble separation)
  - Attribution (classification heads):
    - `model_id_acc_cls=0.189` (only slightly above chance `0.125`)
    - `version_acc_cls=0.612` (strong)
    - `payload_exact_acc_cls=0.110` (meaningful but far from target)
- Additional insight from `outputs/quick_voice_smoke_full2703_v2/metrics.jsonl`:
  - Final strict operating point:
    - `tpr_at_fpr_1pct≈0.7930` (clean)
    - `tpr_at_fpr_1pct_reverb≈0.4141` (reverb)
  - Best `payload_exact_acc_cls` during the run peaked higher than the final value, so **checkpoint selection / early stopping** is likely important if we want the best attribution without regression.

### Run R — `outputs/quick_voice_smoke_512_attr_diag` (512 clips, attribution-focused diagnostic)
- Source: `outputs/quick_voice_smoke_512_attr_diag/audio/source.txt`
- This run is designed to be a **fast “are we learning attribution at all?” check**:
  - Total clips: 512 (256 pos / 256 neg)
  - Positives cover **all 128 (model_id, version) pairs** uniformly (2 examples per pair)
  - Training uses **no reverb augmentation** (`reverb_prob=0.0`) to isolate attribution learnability from robustness.
  - Loss weights were biased toward model attribution (`model_ce_weight=12`, `version_ce_weight=2`).
- Config snapshot (from `outputs/quick_voice_smoke_512_attr_diag/metrics.jsonl` + manifest):
  - Epochs: `s1=6`, `s1b=8`, `s2=12`, `s1b_post=12`
  - Augmentation: `reverb_prob=0.0`
  - Loss weights: `model_ce_weight=12`, `version_ce_weight=2`, `unknown_ce_weight=0`, `neg_weight=5`, `neg_preamble_target=0.5`
- Key metrics from `outputs/quick_voice_smoke_512_attr_diag/audio/decode_report.txt`:
  - `mini_auc=0.9975`, `mini_auc_reverb=0.9534` (excellent detection)
  - `preamble_pos_avg=1.00`, `preamble_neg_avg=0.70` (false preamble is too high in this run)
  - Attribution (classification heads):
    - `model_id_acc_cls=0.191` (still near chance)
    - `version_acc_cls=0.363`
    - `payload_exact_acc_cls=0.082`
- Additional insight from `outputs/quick_voice_smoke_512_attr_diag/metrics.jsonl`:
  - `model_id_acc_cls` briefly peaked early in Stage 2 (≈`0.3125`) and then dropped by the final probe (≈`0.1719`), suggesting the post-Stage2 fine-tune and/or loss-weight balance can **erase attribution gains** if not tuned carefully.

## Key Insight / Current Status

1. **The pipeline works end-to-end**: Stage 1 → Stage 1B → Stage 2 completes; audio artifacts are produced and are listenable.
2. **Detection can be made very strong**: We achieved near-perfect AUC and high reverb AUC (Run M).
3. **Attribution is still the bottleneck** (mainly `model_id`):
   - In the full 128-pair setting, `version_acc_cls` can become strong (Run Q: `0.612`) while `model_id_acc_cls` remains near chance (`~0.18–0.24` across Runs P/Q/R).
   - This suggests we’re learning “presence + some identity” but not robust, balanced identity across both fields.
4. **Final-epoch metrics can be misleading**: multiple runs show that attribution peaks mid-training and then regresses; selecting checkpoints by `payload_exact_acc_cls` (or conditional attribution metrics) is likely necessary.

### Gap vs acceptance criteria
From `docs/WATERMARK_PROJECT_PLAN.md` (“Acceptance Criteria”):
- Detection AUC target (`>0.95`) is already met in multiple voice runs.
- Strict detection target (`TPR @ 1% FPR > 85%`) is **close** on clean audio in Run Q (`~0.793`), but is still **well below target under reverb** (`~0.414`).
- Attribution target (`>85%` on true positives) is **not met yet** (current best full-grid exact attribution is `~0.11`).

### Root Cause Identified (Run M collapse)
Stage 2 was training on **mixed manifests** (pos + clean negatives), but it still embedded/optimized a message for *every* sample.
- Clean negatives have `model_id=None`, `version=None`, and `WatermarkDataset` maps that to a constant `(model_id=0, version=1)`.
- This makes Stage 2 supervision highly skewed (≈56% model_id=0, ≈53% version=1 in Run M), and the encoder converges to a “presence-only / constant-identity” watermark.

**Fix applied (2026-01-24)**:
- `watermark/training/stage2.py` now defaults to `message_mode="random_if_mixed"` to sample balanced random `(model_id, version)` during Stage 2 when a manifest contains clean negatives.
- `watermark/scripts/quick_voice_smoke_train.py` and `watermark/scripts/train_full.py` now run the optional post-Stage2 Stage 1B fine-tune as **heads-only** (warmup mode) to avoid damaging detection/preamble separation.
- `watermark/scripts/quick_voice_smoke_train.py` now assigns **independent** `(model_id, version)` pairs for positives (covers all `8*16=128` combinations) so `payload_exact_acc*` is meaningful.
- `watermark/training/stage2.py` now defaults to supervising payload/attribution losses on **clean watermarked audio** (`payload_on_clean=True`) while still supervising detection on augmented audio, to reduce “presence-only” collapse under heavy augmentations.

## Proposed Solutions (Next Iteration)

### 1) Train an explicit classification head for attribution
The decoder already has `head_model` (`watermark/models/decoder.py`) but Stage 1B/2 currently optimize bitwise message BCE only.
- Add a `CrossEntropy` term on `model_logits` (and optionally a separate version head) for positives.
- This usually converges faster than packing into a few bits, and avoids “bit-accuracy looks OK but ID accuracy is random”.

**Implemented follow-up (2026-01-23)**:
- Added `head_version` + top-k aggregated `avg_model_logits` / `avg_version_logits` in `watermark/models/decoder.py`.
- Stage 1B/Stage 2 now include optional `CrossEntropy` losses for model/version; `decode_report.txt` includes `*_acc_cls` metrics from these heads.
- Stage 1B negative preamble regularizer now targets uncertainty (`p≈0.5`) instead of all-zeros to reduce “preamble collapse”.

### 1.25) Specific failure mode: model_id lags version
Observed in Runs P/Q/R:
- `version_acc_cls` is consistently easier to learn than `model_id_acc_cls`.
- Confusion matrices show the model-id head still predicts a **small subset of classes** far too often (class-collapse), even when detection is strong.

Likely causes:
- **Multi-task trade-off**: the encoder can satisfy detection/quality objectives by embedding a “generic” watermark and only partially encoding identity.
- **Uneven supervision pressure**: payload BCE covers 3 model bits vs 4 version bits, so without careful weighting/scheduling it’s “cheaper” to encode version reliably than model_id.
- **Training-stage interference**: post-Stage2 Stage1B fine-tune can reduce identity accuracy (Run R), suggesting we need better checkpoint selection and/or a different post-Stage2 schedule.

Practical next steps to address it:
- Add a **joint 128-class head** (`pair_id = model_id + 8*version`) and optimize it directly; this removes the “encode only version” loophole.
- Rebalance losses: either (a) raise `model_ce_weight` while lowering `version_ce_weight`, or (b) weight payload bits so model_id bits matter more than version bits.
- Use **curriculum**: learn identity first with `reverb_prob=0.0`, then gradually increase augmentations once exact attribution clears a threshold.
- Add checkpoint selection based on `payload_exact_acc_cls` (not last epoch), and consider disabling or shortening `s1b_post` if it consistently causes regression.

### 1.5) Ensure Stage 2 supervision is balanced (critical)
If your manifest includes clean negatives (`has_watermark=0`), Stage 2 must not inherit `(model_id=None, version=None)` as a constant payload target.
- Use Stage 2 random message sampling (`message_mode="random"` or the default `"random_if_mixed"`).

### 2) Run a short Stage 1B fine-tune after Stage 2
Right now Stage 2 changes the encoder distribution while the decoder is frozen.
- After Stage 2 completes, run Stage 1B for a few epochs (decoder-only) to re-fit the message head to the new encoder embeddings.

### 3) Make message-conditioning harder to ignore in the encoder
If attribution remains random:
- Increase message influence (e.g., stronger conditioning path / bounded FiLM around 1.0 / add a message-conditioned additive carrier branch).
- Increase `msg_weight` (Stage 2) while monitoring perceptual quality.

### 4) Make attacks/diagnostics deterministic for apples-to-apples comparisons
Reverb is stochastic; a single “watermarked_reverb.wav” might fail even if average AUC is strong.
- Add a fixed seed for attacks inside smoke scripts when comparing runs.

## Recommended Next Diagnostics To Run

Use the built-in presets and a distinct `--out` to avoid overwriting:
- Medium run:
  - `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --profile medium --out outputs/quick_voice_smoke_medium`
- Mildly bigger run (more clips, still few epochs):
  - `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 256 --epochs_s1 3 --epochs_s1b 2 --epochs_s2 3 --out outputs/quick_voice_smoke_256q`

If attribution is still stuck at chance, try increasing CE weights (and keep quality listening checks):
- `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --profile medium --model_ce_weight 2.0 --version_ce_weight 2.0 --out outputs/quick_voice_smoke_medium_ce2`

If you want “robustness + identity” (repeat Run M intent, but with the Stage 2 fix in place):
- `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 2703 --epochs_s1 10 --epochs_s1b 10 --epochs_s2 20 --epochs_s1b_post 3 --model_ce_weight 3 --version_ce_weight 3 --neg_weight 5 --unknown_ce_weight 0 --reverb_prob 0.5 --out outputs/quick_voice_smoke_4k_r05_msgfix`

## Live Training Dashboard (Real-Time)

The training scripts can write a lightweight JSONL metrics log and you can view it live in a browser.

1) Run training (writes `metrics.jsonl` under `--out` by default):
- `.venv/bin/python -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --profile medium --out outputs/run_with_dashboard`

2) In a second terminal, start the dashboard:
- `.venv/bin/python -m watermark.scripts.live_dashboard --log outputs/run_with_dashboard/metrics.jsonl --port 8765`

3) Open:
- `http://127.0.0.1:8765`

The dashboard shows:
- Stage losses (per-epoch)
- Probe metrics (AUC / reverb AUC, preamble pos/neg, classification attribution accs)
- Default target thresholds (configurable by editing the script meta targets if needed)

Then review:
- `outputs/quick_voice_smoke_medium/audio/decode_report.txt`
- `outputs/quick_voice_smoke_256q/audio/decode_report.txt`
