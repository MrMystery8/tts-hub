# Watermark Final Findings

This note preserves the current watermarking conclusions for the final report writeup.

## Selected run

Use `outputs/dashboard_runs/sweep3_B_static_12_2` as the selected watermark model.

Rationale:
- It is the best of the three 3-model sweep runs by tiered robustness.
- It uses the final target taxonomy: `n_models=3`, so classes are `clean + 3 model IDs`.
- It has the same perceptual quality as the other sweep runs because the encoder is the same.

Training setup:
- Source: `medium_benchmark_data`
- Clips: `5120` total, split into about `4096` train, `512` validation, `512` test
- Schedule: Stage 1 only (`epochs_s1=100`, `epochs_s2=0`, `epochs_s1b_post=0`)
- Weights: `detect_weight=12`, `id_weight=2`
- Best metric: `composite_score = tpr_at_fpr_1pct * id_acc_pos`

## Best validation checkpoint

Best validation probe for Run B was epoch 98:

| Metric | Value |
|---|---:|
| AUC | 0.989 |
| TPR at 1% FPR | 0.847 |
| ID accuracy on positives | 0.897 |
| Watermark accuracy | 0.828 |
| Composite score | 0.760 |
| Clean probability on positives | 0.064 |
| Clean probability on negatives | 0.891 |

Interpretation: the model separates clean vs watermarked audio well and attributes the positive samples to the correct 3-model class at about 0.90 validation accuracy.

## Held-out test and tiered robustness

Run B tiered evaluation used `512` held-out test items (`264` positive, `248` clean).

Clean held-out test:

| Metric | Value |
|---|---:|
| AUC | 0.988 |
| TPR at 1% FPR | 0.795 |
| ID accuracy on positives | 0.875 |
| Watermark accuracy | 0.769 |
| Composite score | 0.696 |

Tier means:

| Tier | Description | AUC | TPR at 1% FPR | ID acc | Composite |
|---|---|---:|---:|---:|---:|
| T1 | Codec/resample/noise/trim | 0.986 | 0.787 | 0.860 | 0.676 |
| T2 | Denoise/loudnorm/EQ/background mix | 0.974 | 0.527 | 0.482 | 0.275 |
| T3 | Reverb/recapture simulation | 0.882 | 0.356 | 0.366 | 0.128 |

Main robustness finding:
- T1 robustness is strong and close to clean performance.
- T2 keeps reasonable detection AUC, but attribution degrades sharply.
- T3 degrades substantially; reverb and recapture-like conditions are not solved.

Report wording should avoid saying "robustness is solved." A defensible statement is:

> The watermarking system remains effective under common T1 transformations, degrades under T2 signal-processing transformations, and is weak under T3 reverb/recapture-like conditions.

## Perceptual quality

Quality evaluation for Run B used `200` clips:

| Metric | Mean | P10 | P50 |
|---|---:|---:|---:|
| PESQ | 4.638 | 4.630 | 4.643 |
| STOI | 0.99996 | 0.99990 | 1.00000 |
| SNR | 21.52 dB | 16.42 dB | 21.81 dB |

Interpretation:
- PESQ is essentially at the wideband ceiling.
- STOI is essentially perfect.
- SNR looks worse than PESQ/STOI because the watermark perturbation is speech-shaped/masked, so SNR should not be used as the primary imperceptibility claim.
- Informal listening suggested that the watermark residual sounded similar to the source voice rather than like independent broadband noise. This matters because speech-like residual energy can sit in already-active speech regions and be psychoacoustically masked by the carrier. SNR still counts that residual energy as distortion, but PESQ/STOI and listening are better aligned with whether the watermark is actually perceptible.
- The high clean-vs-watermarked perceptual similarity is therefore expected: the watermark appears to be embedded in a voice-correlated way instead of as a separate audible noise layer.

Report wording:

> Although the waveform SNR does not meet the earlier -30 dB budget proxy, perceptual metrics and listening checks indicate that the watermark is effectively transparent for the evaluated samples.

## S2/S3 decision

Current evidence does not justify more Stage 2 or Stage 3 training as the main path.

Observed pattern:
- Stage 1 already gives the best clean/T1 tradeoff for the current goal.
- Stage 2 can improve strict detection in some runs but often hurts attribution.
- Stage 3 did not recover the tradeoff in the tested configurations.

Report framing:

> The final selected configuration is Stage 1 only because it achieved the strongest practical detection-attribution tradeoff for the bounded evaluation suite. Stage 2 and Stage 3 remain possible future work for robustness-specific training, but were not selected for the final model.

## What to do next

Priority now is report writing, not more training.

Optional only if time remains:
- Run one larger B-style model, e.g. `--num_clips 10240`, to check whether more data improves confidence.
- Run quality-after-attack checks for T1 transformations.
- Try robustness-aware Stage 1 as a future-work ablation, not as a blocker for the current report.
