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

### A note on splits (train/val/test)

`quick_voice_smoke_train` now creates a proper split (defaults: `0.8/0.1/0.1`, split **by unique file path** to avoid leakage):

- `manifest_train.json` (used for training)
- `manifest_val.json` (used for per-epoch `probe` metrics)
- `manifest_test.json` (used for final `test_probe` once at the end)
- `manifest.json` is an alias of `manifest_train.json` for compatibility

The final `test_probe` now reports:
- Clean metrics (e.g. `tpr_at_fpr_1pct`, `id_acc_pos`)
- Reverb robustness metrics (e.g. `tpr_at_fpr_1pct_reverb`, `id_acc_pos_reverb`)
- Optional extra attack metrics (suffix `_<attack_name>`), controlled by `--test_attacks` (default: `resample_8k,noise_white_20db`)
- Imperceptibility diagnostics: `wm_snr_db_mean`, `wm_budget_ok_frac` (uses `BUDGET_TARGET_DB` from `watermark/config.py`)

### Hub mode (K=2 for the two supported TTS models)

TTS Hub currently maps only two TTS models to attribution IDs:

- `index-tts2` → `pred_model_id=0` (encoder `class_id=1`)
- `qwen3-tts-mlx` → `pred_model_id=1` (encoder `class_id=2`)

For clean-audio iteration you can train a smaller head with `K=2` via `--n_models 2` (so `num_classes=3` including clean).
The hub reads `config.json` from the selected run to instantiate the correct class count.

Example (fast K=2 run):

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --n_models 2 --source_dir mini_benchmark_data --num_clips 2048 --epochs_s1 40 --epochs_s2 0 --epochs_s1b_post 0 --reverb_prob 0.0 --detect_weight 8.0 --id_weight 5.0 --neg_weight 3.0 --probe_every 1 --probe_reverb_every 0 --probe_clips 512 --test_attacks ""`

### Goal 1: detection-first small run (fast)

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 512 --epochs_s1 6 --epochs_s2 10 --epochs_s1b_post 0 --reverb_prob 0.25 --detect_weight 6.0 --id_weight 0.1 --probe_clips 512 --probe_every 1 --probe_reverb_every 1`

What "good" looks like:

- `mini_auc` rises quickly (≥0.95 is typical when it's working).
- `tpr_at_fpr_1pct` rises and stays stable.
- `thr_at_fpr_1pct` does not drift toward 1.0.
- Under reverb (`*_reverb`), you should still see separation (AUC not collapsing).
- Under resampling/noise (`*_resample_8k`, `*_noise_white_20db`), `tpr_at_fpr_1pct_*` should not collapse compared to clean.

If you suspect S2 “isn’t helping”, evaluate the same run under attacks + budget metrics:

`./.venv/bin/python3 watermark/scripts/eval_run_suite.py --run outputs/dashboard_runs/<RUN_ID> --n 256`

Or compare a batch of recent dashboard runs:

`./.venv/bin/python3 watermark/scripts/compare_dashboard_runs.py --limit 12 --n 256`

Important: this goal is intentionally *detection-first*. With `id_weight=0.1`, you should expect weak attribution (`id_acc_pos` barely above chance) unless you later do explicit ID training.

### Goal 1b: balanced run (detection + ID, single run)

If you want a *deployable* encoder+decoder that can both detect and attribute, don’t start with `id_weight=0.1`.

Good starting point:

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --source_dir mini_benchmark_data --num_clips 2048 --epochs_s1 6 --epochs_s2 20 --epochs_s1b_post 0 --reverb_prob 0.25 --detect_weight 6.0 --id_weight 1.0 --probe_clips 1024 --probe_every 1 --probe_reverb_every 1`

Notes:
- This typically preserves strong detection while encouraging the encoder+decoder to actually learn attribution signal early.
- If `id_acc_pos` still lags, only then consider Stage 3 finetune.

### Goal 2: ID finetune (reuse the detection run)

Use the previous run directory (under `outputs/dashboard_runs/<RUN_ID>/`) as the source of truth:

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --manifest outputs/dashboard_runs/<RUN_ID>/manifest.json --load_encoder outputs/dashboard_runs/<RUN_ID>/encoder.pt --load_decoder outputs/dashboard_runs/<RUN_ID>/decoder.pt --epochs_s1 0 --epochs_s2 0 --epochs_s1b_post 6 --reverb_prob 0.25 --detect_weight 3.0 --id_weight 3.0 --neg_weight 1.0 --freeze_detect_head_in_s3 --probe_clips 512 --probe_every 1 --probe_reverb_every 1`

Note: If the source run folder contains `manifest_train.json`/`manifest_val.json`/`manifest_test.json`, the script will **reuse those exact splits** (it won’t resplit).

What to look at during ID tuning:

- `id_acc_pos` (ID head accuracy on positives; ignores detection threshold).
- `confusion_attr` (K×K, watermarked-only confusion).
- Keep an eye on detection regression (`tpr_at_fpr_1pct` and `*_reverb`).

#### When the “runbook pipeline” underperforms for ID

Recent split-based dashboard runs suggest the common failure mode is:

- Detection-first run achieves strong detection on held-out test (`tpr_at_fpr_1pct` high),
- but the subsequent Stage 3 ID finetune only modestly improves `id_acc_pos`, sometimes while reducing detection.

Why this happens:

- Stage 1 explicitly trains the decoder against a **frozen** encoder. If you set `id_weight` very low in S1, the decoder’s ID head won’t learn much signal.
- Stage 2 freezes the decoder, so it cannot “catch up” on ID if it was undertrained in S1.
- Then Stage 3 has to learn ID while also juggling quality/budget constraints and (often) reverb augmentation — which can be slow and can move the model away from the strong-detection basin.

Practical recommendation (attribution-focused training):

1) Don’t set `id_weight` extremely low if you intend to ship attribution.
   - A good starting point is `detect_weight≈6`, `id_weight≈1` for S1+S2.
2) Treat Stage 3 as optional.
   - For detection-only watermarking: S1+S2 is usually enough.
   - For attribution: use Stage 3 only if val/test `id_acc_pos` is still weak, and keep it short + monitored.

If you already ran a detection-first job with `id_weight=0.1` and want to “recover” attribution:

- Do a **decoder-only ID warmup** first (same encoder, just teach the decoder’s ID head):

`./.venv/bin/python3 -m watermark.scripts.quick_voice_smoke_train --manifest outputs/dashboard_runs/<RUN_ID>/manifest.json --load_encoder outputs/dashboard_runs/<RUN_ID>/encoder.pt --load_decoder outputs/dashboard_runs/<RUN_ID>/decoder.pt --epochs_s1 10 --epochs_s2 0 --epochs_s1b_post 0 --reverb_prob 0.25 --detect_weight 6.0 --id_weight 2.0 --probe_clips 512 --probe_every 1 --probe_reverb_every 1`

- Then, if needed, continue with encoder training (Stage 2) and only then finetune (Stage 3).

### Checkpointing and Model Reuse

### Best Checkpoint Logic
The system automatically selects the "best" checkpoint based on the schedule:
- **Detection Phase** (Stage 1/2): Defaults to `tpr_at_fpr_1pct`.
- **Finetune Phase** (Stage 3): Defaults to `id_acc_pos` (ID accuracy).

**Important Nuances:**
1. **Reverb Priority**: If `reverb` probe metrics are available (e.g., `tpr_at_fpr_1pct_reverb`), the checkpoint manager **automatically prefers them** over clean metrics. This ensures "best" actually implies "robust".
2. **Collapse Guardrail**: When optimizing for `id_acc_pos`, the system enforces a **hard guardrail**: it will REJECT a new "best" ID checkpoint if `tpr_at_fpr_1pct < 0.30`. This prevents saving a model that has excellent attribution but cannot detect the watermark at all.

The quick smoke train now supports robust checkpointing with the following artifacts saved in each run directory:

- `encoder.pt` / `decoder.pt` - Always saved (best model weights for easy reuse)
- `config.json` - Full CLI arguments and derived settings
- `manifest_train.json` / `manifest_val.json` / `manifest_test.json` - Split manifests (train/val/test)
- `manifest.json` - Alias of `manifest_train.json` for compatibility
- `checkpoints/` directory containing:
  - `last.pt` - Crash-safe checkpoint (overwritten each save interval)
  - `best.pt` - Best model based on chosen metric
  - `best_meta.json` - Best checkpoint metadata (metric name/value/epoch)
  - `last_meta.json` - Last checkpoint metadata (epoch/stage info)

#### Checkpointing Options

Additional CLI arguments for checkpointing control:

- `--ckpt_dir <path>` - Directory to save checkpoints (default: `<out>/checkpoints`)
- `--save_last` / `--no_save_last` - Enable/disable saving last checkpoint (default: True)
- `--save_best` / `--no_save_best` - Enable/disable saving best checkpoint (default: True)
- `--best_metric <name>` - Metric to use for best checkpoint (default: auto-select)
- `--best_mode {max,min}` - Maximize/minimize best metric (default: max)
- `--save_every <N>` - Save last checkpoint every N epochs (0 to save every epoch, default: 1)
- `--save_every <N>` - Save last checkpoint every N epochs (0 to save every epoch, default: 1)
- `--resume <path>` - Resume from checkpoint path (automatically detects stage and skips to correct epoch)
- `--extend_to_epochs_s1 <N>` - When resuming Stage 1, extend training to this many total epochs (useful for "add more epochs" workflow)

#### Default Best Metric Selection

- If doing detection-first (S1/S2): prefers `tpr_at_fpr_1pct_reverb` if available, else `tpr_at_fpr_1pct`
- If doing ID finetune (S3): prefers `id_acc_pos` (with detection guardrail to prevent threshold collapse)

The `encoder.pt` and `decoder.pt` files in the run directory always contain the best performing model weights, making it easy to continue training or deploy models.

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

## Overnight Auto-Tuning (S1 Weights)

For finding the best balance of `detect_weight`, `id_weight`, and `neg_weight` without manual babysitting, use the overnight tuner.

### How it works
- Generates a **shared manifest** so all trials use the exact same data split.
- Runs **multiple phases** (A -> B -> C), promoting only the best performing trials.
- Optimizes for a **Composite Score** (`TPR@1%FPR * ID_Accuracy`).
- Respects a **Time Budget** (e.g., 7.5 hours).

### Usage
```bash
# Run from root
./.venv/bin/python -m watermark.scripts.overnight_tune_s1 \
  --source_dir mini_benchmark_data \
  --out_root outputs/dashboard_runs/overnight_tune_001 \
  --max_hours 7.5 \
  --num_initial 6
```

### Monitoring the Tuner
The tuner creates a standard run directory for each trial (e.g., `trial_001_...`). You can point the dashboard at any specific trial's `metrics.jsonl` to see its progress, or simply tail the tuner logs.


## If detection improves in Stage 1 but regresses in Stage 2/3

This is currently the most important failure mode seen in practice. Typical mitigations:

- Increase Stage 2 epochs (encoder needs more time to learn a robust watermark under quality/budget constraints).
- Increase `detect_weight` during Stage 2.
- Keep `id_weight` low until detection is stable.
- In Stage 3, use `--freeze_detect_head_in_s3` and keep non-trivial `detect_weight` so detection doesn’t drift.


## Benchmark Workflows

### 1. Mini Benchmark (Default)
The repo comes with a small committed dataset in `mini_benchmark_data/`. This is used by default if no other source is specified.
- **Size**: ~500 clips
- **Use case**: Quick smoke tests, development iteration.

### 2. Medium Benchmark (LibriSpeech 100h Subset)
For more robust evaluation, use the "Medium" benchmark derived from LibriSpeech `train-clean-100`.

**Setup:**
1. Download and extract the data (~20k clips):
   ```bash
   python tools/create_medium_benchmark.py
   ```
2. Generate the training manifest (balanced 50/50 watermark/clean):
   ```bash
   python tools/create_medium_manifest.py
   ```
   This creates `medium_benchmark_train.json` and `medium_benchmark_test.json`.

**Running Training:**
```bash
./.venv/bin/python -m watermark.scripts.quick_voice_smoke_train \
    --manifest medium_benchmark_train.json \
    --epochs_s1 6 --epochs_s2 6
```

## Local CLI smoke tests (engineering contract)

- `./.venv/bin/python -m unittest tests.smoke_test`
- `./.venv/bin/python -m pytest -q watermark/tests`
