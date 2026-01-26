# Trial and Error: Model ID Detection & Attribution

**Objective**: Achieve robust watermark attribution (>85% accuracy on 128 (Model, Version) pairs) without destroying detection performance (AUC > 0.95).

---

## 1. The Problem (Class Collapse)
Initial runs showed high Detection AUC (>0.99) but attribution accuracy at random chance (~0.7% for pairs).
**Diagnosis**: The model found a "shortcut"—optimizing purely for presence detection (binary classification) and ignoring the identity bits because they are harder to learn and have less gradient signal.

---

## 2. Experiment Log

### Experiment A: The Baseline (0.5k)
*   **Date**: 2026-01-25
*   **Scale**: 500 clips
*   **Config**: `model_ce_weight=1.0`, `epochs_s1=6`
*   **Result**:
    *   **Detection AUC**: **0.997** (High)
    *   **Attribution Acc**: **1.6%** (Chance level)
*   **Conclusion**: Confirmed "Class Collapse". The model memorized "presence" and ignored "identity".

### Experiment B: Identity-First (0.5k "The Snap")
*   **Hypothesis**: We can force the encoder to embed bits by suppressing the detection objective and amplifying identity penalties.
*   **Config**:
    *   `epochs_s1=2` (Minimal detection warmup)
    *   `epochs_s1b=10` (Deep decoder warmup)
    *   `model_ce_weight=15.0` (Massive penalty)
    *   `msg_weight=10.0`
*   **Result**:
    *   **Detection AUC**: **0.67** (Collapsed)
    *   **Attribution Acc**: **11.6%** (~14x Chance)
*   **Conclusion**: **"The Snap" confirmed.** The shortcut was broken. The model sacrificed detection to learn identity. Learnability is proven.

### Experiment C: Scaled Identity-First (2k)
*   **Hypothesis**: Scale will heal the "Tug-of-War" between Detection and Identity.
*   **Config**: Same as Exp B, but 2048 clips.
*   **Result (Epoch 9)**:
    *   **Detection AUC**: **0.91** (Recovered!)
    *   **Attribution Acc**: **7.4%** (Maintained ~9x Chance)
*   **Conclusion**: Scale helps capacity. The model optimized both objectives simultaneously for a brief window before oscillating.

---

## 3. Strategic Pivot: The "Robust Route" (Current Phase)

**Critique**: Expert review pointed out that 7.4% accuracy is still insufficient for production, and the training setup had severe flaws:
1.  **Dataset Leak**: `dataset.py` was returning `(0,0)` messages for clean/unlabeled audio, teaching the model to output class 0 on clean inputs.
2.  **Destructive Weights**: `model_ce_weight=15` destroyed the detection features instead of reusing them.
3.  **No Freezing**: We tried to "weight" our way out of a feature learning problem instead of freezing the backbone.

### Planned Code Changes
*   **Dataset**: Modify `dataset.py` to return `model_id=-1` for unlabeled/clean audio.
*   **Trainer**: Modify `train.py` / `stage1b.py` to support freezing `decoder.backbone` and `head_detect`.
*   **Checkpointing**: Save `best_attr.pt` based on `pair_acc_cls_cond_1pct` (conditional accuracy) to capture the "sweet spot" automatically.

### Experiment D: "Safe Snap" (Smoke Test, 0.5k)
*   **Date**: 2026-01-25
*   **Hypothesis**: With `freeze_backbone` (BN fixed) and `dataset` fixed (no leak), we can learn identity without destroying detection.
*   **Config**: `num_clips=512`, `freeze_backbone=True`, `model_ce_weight=5.0`, `reverb_prob=0.0`.
*   **Results & Time Series**:
    *   **Phase 1: Encoder Training (S2)**
        *   `accept_rate` yielded **100% -> 94.9%**.
        *   `mini_auc` held steady > 0.99.
        *   `pair_acc` stayed flat at ~2% (Random chance ~0.8%).
        *   *Insight*: Encoder learned to embed *something* without breaking detection.
    *   **Phase 2: Attribution Training (S1B Post)**
        *   `pair_acc` jumped: 2.0% -> **8.23%**.
        *   `accept_rate` remained stable at **94.9%**.
        *   *Insight*: Once the encoder was fixed, the frozen attribution heads successfully decoded the signal.
*   **Conclusion**: **SUCCESS (Stability verified)**. The "BN Freeze" and "Dataset Patch" worked. We decoupled "embedding capacity" (Stage 2) from "semantic attribution" (Stage 1B Post).

### Experiment E: "Identity Acquisition" (Scale Up, 2k)
*   **Goal**: Scale to 2048 clips to let attribution heads generalize. Keep `freeze_backbone` to maintain safety.
*   **Config**:
    *   `num_clips=2048`
    *   `epochs_s1b=20`, `epochs_s2=30`, `epochs_s1b_post=20` (Longer training)
    *   `freeze_backbone=True`
    *   `pair_ce_weight=6.0` (Increased to force ID learning)
