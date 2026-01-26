# Watermark Runbook (MPS-friendly)

This runbook documents how to iterate quickly on the watermark system on macOS/MPS, and how to interpret the dashboard outputs.

## What’s implemented (current state)

- Watermark mode is **multiclass attribution**: `0=clean`, `1..K=model_id+1` (`watermark/config.py`).
- Decoder is **loc-gated** (`watermark/models/decoder.py`):
  - `loc` head: localized watermarkness score over the window time axis
  - `detect` head: auxiliary global detection (kept for compatibility/regularization)
  - `id` head: K-way attribution using **loc-weighted pooling** (Goal 2)
  - Combined class distribution uses the localization pooled watermark score:
    - `P(clean)=1-P(wm_loc)`
    - `P(class=i+1)=P(wm_loc)*P(id=i)`
- Training is intentionally staged:
  - Stage 1: decoder pretrain (`watermark/training/stage1.py`)
  - Stage 2: encoder train (`watermark/training/stage2.py`, decoder frozen)
  - Stage 3: finetune (`watermark/training/stage2.py`, encoder+decoder)

## Quick dashboard workflow (recommended)

### Start the dashboard

`./.venv/bin/python3 -m watermark.scripts.live_dashboard`

Runs controller mode and stores runs under `outputs/dashboard_runs/`.

### Goal 1: detection-first small run (fast)

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 512 --epochs_s1 6 --epochs_s2 10 --epochs_s1b_post 0 --reverb_prob 0.25 --detect_weight 6.0 --id_weight 0.1 --probe_clips 512 --probe_every 1 --probe_reverb_every 1`

What “good” looks like:

- `mini_auc` rises quickly (≥0.95 is typical when it’s working).
- `tpr_at_fpr_1pct` rises and stays stable.
- `thr_at_fpr_1pct` does not drift toward 1.0.
- Under reverb (`*_reverb`), you should still see separation (AUC not collapsing).

### Goal 2: ID finetune (reuse the detection run)

Use the previous run directory (under `outputs/dashboard_runs/<RUN_ID>/`) as the source of truth:

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --manifest outputs/dashboard_runs/<RUN_ID>/manifest.json --load_encoder outputs/dashboard_runs/<RUN_ID>/encoder.pt --load_decoder outputs/dashboard_runs/<RUN_ID>/decoder.pt --epochs_s1 0 --epochs_s2 0 --epochs_s1b_post 6 --reverb_prob 0.25 --detect_weight 3.0 --id_weight 3.0 --neg_weight 1.0 --freeze_detect_head_in_s3 --probe_clips 512 --probe_every 1 --probe_reverb_every 1`

What to look at during ID tuning:

- `id_acc_pos` (ID head accuracy on positives; ignores detection threshold).
- `confusion_attr` (K×K, watermarked-only confusion).
- Keep an eye on detection regression (`tpr_at_fpr_1pct` and `*_reverb`).

## How to interpret the confusion matrices

- `confusion` is **thresholded** at `FPR=1%` using `thr_at_fpr_1pct`.
  - If detection is weak, the threshold becomes extremely high → `pred_clean_rate` approaches 1.0 → `confusion` becomes mostly `0/0`.
  - This is expected and doesn’t necessarily mean the ID head has no signal.
- `confusion_attr` is **watermarked-only** (K×K) and does not depend on the detection threshold.
  - This is the primary tool for attribution debugging.

## Dataset / clips knobs

- `mini_benchmark_data` contains ~2.7k files; `--num_clips` samples from it (and repeats with replacement if you request more than the dataset size).
- `--probe_clips` controls how many clips are cached for probe metrics (larger is more stable, slower).

Recommended sizes on MPS (rough guidance):

- Small (fast iteration): `--num_clips 512`, `--probe_clips 256–512`
- Medium (more stable): `--num_clips 2048`, `--probe_clips 1024`
- Large (overnight-ish): `--num_clips 8192+`, `--probe_clips 2048`

## If detection improves in Stage 1 but regresses in Stage 2/3

This is currently the most important failure mode seen in practice. Typical mitigations:

- Increase Stage 2 epochs (encoder needs more time to learn a robust watermark under quality/budget constraints).
- Increase `detect_weight` during Stage 2.
- Keep `id_weight` low until detection is stable.
- In Stage 3, use `--freeze_detect_head_in_s3` and keep non-trivial `detect_weight` so detection doesn’t drift.

## Local CLI smoke tests (engineering contract)

- `./.venv/bin/python -m unittest tests.smoke_test`
- `./.venv/bin/python -m pytest -q watermark/tests`
