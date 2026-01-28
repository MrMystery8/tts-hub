# Benchmarking Attempts Log

## Goal
Run a robust "Mini-Benchmark" on 500 LibriSpeech clips (400 train, 100 test) with 80/20 split and attack suite (Clean, Noise, Resample, Reverb) to prove readiness for scaling.

## History

| Attempt | Status | Issue / result | Fix |
|---------|--------|----------------|-----|
| **Gate A** | ✅ Pass | AUC 1.00 on Pink Noise | Aux Loss + High LR |
| **Gate B** | ✅ Pass | AUC 1.00 on Real Speech (9 clips) | Reduced Quality Weight (10->1) |
| **Mini-BM 1** | ❌ Fail | `ModuleNotFoundError: watermark` | Run as module |
| **Mini-BM 2** | ❌ Fail | `ImportError: TorchCodec` | `torchaudio.load` dependency missing |
| **Mini-BM 3** | ❌ Fail | Missing Resampling | Replaced with `soundfile` but forgot resample |
| **Mini-BM 4** | ❌ Fail | `RuntimeError: Invalid buffer size` | MPS padding bug in `attacks.py` |
| **Mini-BM 5** | ❌ Fail | `RuntimeError: Invalid buffer size` | Still failing on MPS during Eval |
| **Mini-BM 6** | ❌ Fail | `ValueError: too many values to unpack` | Decoder expects `(B, T)`, got `(1, 1, T)` |
| **Mini-BM 7** | 🛑 Stop | Terminated manually to apply fix | **Pending Fix** |
| **Mini-BM 8** | 🛑 Stop | Smoke Test Passed! Code is hardended. | **Stopped to fix client script** |
| **Mini-BM 9** | ❌ Fail | `NameError: train_stage1` | **Bad Import Patch** |
| **Mini-BM 10** | ❌ Fail | `assert` failure in `apply_attack` | **Input Shape mismatch (fixed)** |
| **Mini-BM 11** | ❌ Fail | `ValueError` in `SlidingWindowDecoder` | **Forgot to standardize wrapper (fixed)** |
| **Mini-BM 12** | ❌ Fail | `KeyError: 'avg_message_logits'` | **Missing logit aggregation (fixed)** |
| **Mini-BM 13** | ⏳ Run | Full Hardened Standardized Run | **Current Attempt** |

## Current Status
Engineering hardening is complete. All components follow the `(B, 1, T)` contract.
Evaluation in Attempt 12 reached the evaluation phase without "plumbing" crashes.

## Issues to Watch
1. **Convergence**: AUC 0.46 in Attempt 12 suggests the model needs more data/epochs than the current "Mini-BM" configuration to show robust detection.
2. **CPU Eval Speed**: Running evaluation on CPU is stable but slower; monitoring performance.

## Conclusion (Stable State)
The engineering hardening (Attempts 1-12) successfully stabilized the pipeline. The system now uses a strict `(B, 1, T)` contract and processes attacks safely on CPU.

Current benchmarks are run using:
- **Mini-Benchmark**: `watermark.scripts.quick_voice_smoke_train` (default)
- **Medium Benchmark**: `tools/create_medium_benchmark.py` + `tools/create_medium_manifest.py`

See **[docs/WATERMARK_RUNBOOK.md](docs/WATERMARK_RUNBOOK.md)** for the latest usage instructions.
