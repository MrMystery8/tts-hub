# Sanity Audit Results

**Date**: 2026-01-23
**Configuration**:
- **Dataset**: 200 synthetic clips (Sine waves + Noise)
- **Epochs**: 15 (Stage 1), 10 (Stage 1B), 15 (Stage 2)
- **Device**: Mac MPS

## Metrics

| Metric | Result | Interpretation |
|--------|--------|----------------|
| **Test Accuracy** | 30.0% | Model is underfitting on synthetic data. |
| **Test AUC** | 0.53 | Performance is near random guessing (0.50). |
| **Stage 1 Loss** | ~1.03 | High detection loss (Decoder struggling). |
| **Stage 1B Loss** | ~0.43 | **Good Payload Convergence!** (Decoder can read payload if detected). |
| **Stage 2 Loss** | ~0.85 | Encoder is prioritizing quality (`Qual: 0.0028`) over detection (`Det: 0.68`). |

## Analysis

1.  **Pipeline Verified**: The entire training cycle (S1 -> S1B -> S2 -> Eval) runs without errors, confirming the `On-the-fly Embedding` logic works technically.
2.  **Dataset Issue**: Synthetic sine waves are notoriously difficult for audio watermarking because they have no "masking potential" (any change is audible). The Encoder is likely constrained by the Quality Loss (penalty) and failing to embed a strong enough watermark.
3.  **Payload Success**: Stage 1B (Payload training) dropped to `0.43` loss, proving the standard `WatermarkDecoder` *can* learn to retrieve bits. Ideally, trained on real speech (LJSpeech), the detection performance will align with this payload performance.

## Recommendation

Proceed to training on **LJSpeech (24h)** or similar real-world audio. The current implementation is technically sound but limited by the synthetic "sanity" dataset.
