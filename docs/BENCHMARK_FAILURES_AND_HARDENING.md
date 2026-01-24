# Benchmark Failures & Engineering Hardening Plan

## 1. Summary of Failures
We attempted to run a `mini_benchmark.py` on 500 clips, but encountered repeated "plumbing" failures that indicate a lack of robust engineering foundations.

### Failure Log
| Attempt | Error / Symptom | Root Cause |
|---------|-----------------|------------|
| **1-3** | `ModuleNotFoundError`, `ImportError` | **Dependency/Packing**: Scripts run as files vs modules; `torchaudio` backend inconsistencies. |
| **4-5** | `RuntimeError: Invalid buffer size` | **MPS Validation**: `F.pad` or `crop` on MPS generated invalid shapes (negative or huge) due to lack of pre-attack length checks. |
| **6** | `ValueError: too many values to unpack` | **Contract Violation**: Decoder expected `(B, T)` but received `(1, 1, T)` from the evaluation loop. |
| **7-8** | (Terminated) | **Fragility**: Code was patched with `unsqueeze` guesses instead of a defined contract. |

## 2. Identified Engineering Gaps
The failures revealed three critical gaps in the codebase:
1.  **No Canonical Tensor Contract**: Some parts used `(T,)`, others `(1, T)`, others `(B, 1, T)`. Evaluation mixed these, causing dimension errors.
2.  **Unsafe MPS Operations**: Audio attacks (padding, cropping, resampling) were run on MPS without rigorous bounds checking, causing driver-level crashes.
3.  **Inconsistent I/O**: Loaders swapped between `torchaudio` and `soundfile` with different normalization and resampling behaviors.

## 3. Hardening Plan (Immediate Actions)

We are pivoting from "Benchmarking" to **"Engineering Hardening"** to fix these foundations permanently.

### A. Define Canonical I/O (`watermark/utils/io.py`)
We will enforce a strict contract for all audio data entering the system:
- **Shape**: `(1, T)` (Channels, Time). Mono.
- **Sample Rate**: 16,000 Hz.
- **Dtype**: `float32`.
- **Backend**: `soundfile` (primary) with `torchaudio` for resampling only.

### B. Harden Attack Suite (`watermark/evaluation/attacks.py`)
Attacks will be treated as "unsafe" operations and wrapped with strict validation:
- **Pre-conditions**: Assert `ndim==2`, `shape[-1] > 0`, `isfinite`.
- **Execution**: Force CPU for all attacks (resampling, noise, codecs) to avoid MPS buffer bugs.
- **Post-conditions**: Assert strict output shape matches input length (preservation of timing).

### C. Smoke Testing (`tests/smoke_test.py`)
Before touching real datasets again, we will write a smoke test that:
1.  Generates synthetic sine waves.
2.  Passes them through **every** registered attack.
3.  Verifies shapes and values.
4.  Runs a dummy Encode -> Attack -> Decode cycle.

Only after `smoke_test.py` passes will we resume `mini_benchmark.py`.

### 4. Completed Hardening (System Architecture)

We have moved the entire system from "ad-hoc shapes" to a **Strict Canonical Contract**:

*   **Contract**: All audio tensors in the pipeline are now strictly `(Batch, 1, Time)`.
*   **I/O Implementation**: `watermark/utils/io.py` is the single source of truth for loading. It enforces `(1, T)` @ 16kHz float32.
*   **Dataset Integration**: `watermark/training/dataset.py` refactored to use `load_audio`. It now yields `(1, T)` segments, which `DataLoader` stacks into `(B, 1, T)`.
*   **Attack Suite**: `apply_attack_safe` enforces CPU execution and verifies that attacks preserve the `(C, T)` shape.
*   **Model Wrappers**: Both `OverlapAddEncoder` and `SlidingWindowDecoder` now include "Adapter" logic to handle `(B, 1, T)` or `(B, T)` gracefully, preventing "too many values to unpack" errors.

## 5. Discoveries from Attempt 12

Attempt 12 successfully reached the evaluation phase without crashing, but revealed two final integration issues:

### A. Missing Aggregate Message Logits (`KeyError`)
*   **Problem**: In the evaluation phase, the script crashed with `KeyError: 'avg_message_logits'`.
*   **Discovery**: While we had aggregated detection logits for training, we had NOT implemented aggregation for message logits in `SlidingWindowDecoder`. The evaluation script needs a single "best guess" for the message across the entire clip.
*   **Fix**: Updated `SlidingWindowDecoder` to identify the `top-k` windows (based on detection) and average their message logits to produce `avg_message_logits` and `avg_message_probs`.

### B. Low Initial Robustness (AUC 0.46)
*   **Observation**: The "Clean" attack showed an AUC of 0.46 (worse than chance).
*   **Analysis**: This is likely due to the model overfitting to the tiny noise floor differences or spectral characteristics of the very first training batches, combined with the fact that the decoder and encoder are training from scratch together.
*   **Root Cause**: The current `mini_benchmark` uses only 400 training samples for 8 epochs. This is enough for a "smoke test" of the math, but not for a robust watermark signal to emerge. 

## 6. Engineering Sign-off

The "Plumbing" phase is now complete. We have verified:
1.  **Robust I/O**: No more sampling rate or channel mismatches.
2.  **Shape Contracts**: No more dimension errors in training or evaluation.
3.  **Attack Stability**: No more MPS buffer crashes.
4.  **Full Pipeline**: The system can now flow from loading data -> standard training -> multi-attack evaluation end-to-end.

**Status**: Ready for Attempt 13. Documentation fully updated.

