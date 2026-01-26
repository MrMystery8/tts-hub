# Experiment H2 Success Report (Jan 26)

## Summary
The critical blockers have been resolved. The system is now learning both Detection and Payload simultaneously without collapsing.

## Results (Run `1769362225_9deebf`)

| Metric | Value | Status | Meaning |
|:---|:---|:---|:---|
| **`mini_auc`** | **0.940** | ✅ **Pass** | Detection logic is fixed. Negative mixing in Stage 3 worked perfectly. |
| **`detect_pos_mean`** | 0.78 | — | True Positives generally detected. |
| **`detect_neg_mean`** | 0.14 | — | True Negatives generally rejected (was 0.99 in H1). |
| **`model_id_bit_acc`** | **0.822** | ✅ **Pass** | Beating Majority Baseline (0.64). We are genuinely learning bits. |
| **`baseline_majority_bit_acc`** | 0.641 | — | The "All Zeros/Mask" trap baseline. |
| **`idver_correct`** | 0.016 | ⚠️ Low | Exact payload match is still rare (1.6%), but non-zero. |

## Why this is a Breakthrough
1.  **Escaped All-Zeros Trap**: In Phase 1/2, the model predicted zeros (or constant). With XOR Scrambling (`0xA55A`), the model achieves 82% physical bit accuracy, significantly higher than the best possible constant predictor (64%).
2.  **Fixed Detection Collapse**: In H1, the detector collapsed (AUC 0.43) because Stage 3 lacked negative samples. In H2, adding `neg_weight` logic restored strong separation (0.78 vs 0.14).
3.  **Stability**: The training is stable. Budget loss is actively being minimized (5.8 -> lower).

## Next Steps
Now that the "learning machinery" is proven working (Metrics > Baselines):
1.  **Scale Up**: The current accuracy (82% per bit) is not enough for 32-bit exact matches (`0.82^32` is low). We need longer training or more capacity.
2.  **Curriculum**: As per recommendations, start with a relaxed budget (-20dB) and tighten to -30dB to help the model learn the payload faster.
3.  **Run Experiment I**:
    *   `num_clips`: **2000** (4x data).
    *   `msg_weight`: **5.0** (Boost payload signal).
    *   `neg_weight`: **1.0** (Balance detection).
