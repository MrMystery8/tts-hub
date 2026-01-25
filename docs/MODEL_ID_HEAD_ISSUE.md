# Model ID / Version Attribution (“Model-ID Head”) Issue — Root Cause, Evidence, Fixes

This document captures the long-running debugging thread where **watermark detection is strong** but **attribution (model_id, version) is near-chance**, plus the code-level changes made to address it.

Terminology used throughout:
- **Detection**: deciding whether a clip is watermarked (`clip_detect_prob`, AUC, TPR@1%FPR).
- **Attribution**: recovering `(model_id, version)` (or `pair_id = version*8 + model_id`) on true positives.
- **Preamble**: the fixed 16-bit prefix used to align message decode.

---

## 1) Symptom Summary

The repeated pattern across many runs:
1) **AUC / detection quality** quickly becomes very strong.
2) **Model/version accuracy** stays near chance or collapses to a few classes.
3) **Clean negatives sometimes “look like” the preamble** (high `preamble_neg_avg`), causing hallucinated payloads unless you gate by detection.

This is a known multitask failure mode: detection is easier than identity bits, so a model can learn a robust “presence” feature without learning “who.”

---

## 2) Concrete Evidence (Artifacts + Metrics)

Below are numbers pulled directly from the repo’s run artifacts.

### 2.1 Early “detection solved, attribution broken, negatives match preamble” run

Run: `outputs/quick_voice_smoke_512_attr_diag/`
- Decode report: `outputs/quick_voice_smoke_512_attr_diag/audio/decode_report.txt`

Key metrics:
- `mini_auc=0.9975`
- `preamble_neg_avg=0.70` (too high; negatives often look like preamble matches)
- `payload_exact_acc_cls=0.082`
- `model_id_acc_cls=0.191` (close to chance for 8 classes = 0.125)
- `version_acc_cls=0.363` (above chance for 16 classes = 0.0625, still weak)

This is the “classic” failure signature: detection is essentially done, attribution is not.

### 2.2 “Attribution improved but detection regressed” run (trade-off)

Run: `outputs/quick_voice_smoke_512_attr_diag_v2/`
- Decode report: `outputs/quick_voice_smoke_512_attr_diag_v2/audio/decode_report.txt`

Metrics show attribution can rise while detection falls:
- `mini_auc=0.9383` (worse detection)
- `payload_exact_acc_cls=0.246` (better attribution in that decode report)

Takeaway: **checkpoint selection matters** and “last epoch” can hide that attribution briefly worked.

### 2.3 2k run before Stage-2 attribution fix (baseline)

Run: `outputs/dashboard_runs/1769272369_a2d2ae/`
- Probe metrics: `outputs/dashboard_runs/1769272369_a2d2ae/metrics.jsonl`

Final probe (stage `final`) summary:
- `mini_auc=0.9992`, `tpr_at_fpr_1pct=0.9668`
- `preamble_neg_avg=0.2404` (fixed vs the earlier ~0.65–0.70)
- Attribution still near chance:
  - `model_id_acc_cls=0.1328`
  - `version_acc_cls=0.0938`
  - `pair_acc_cls=0.0156`
  - `payload_exact_acc_cls_cond_1pct=0.0162`

### 2.4 2k run with **Stage-2 payload on all carriers** (improvement but not solved)

Run: `outputs/dashboard_runs/1769281044_9074a4/`
- Report: `outputs/dashboard_runs/1769281044_9074a4/report.md`
- Probe metrics: `outputs/dashboard_runs/1769281044_9074a4/metrics.jsonl`

Final probe (stage `final`) summary:
- Detection remains strong:
  - `mini_auc=0.9968`, `tpr_at_fpr_1pct=0.9355`
  - `preamble_neg_avg=0.1335`
- Attribution improved materially vs baseline, but still far from target:
  - `model_id_acc_cls=0.2246`
  - `version_acc_cls=0.1523`
  - `pair_acc_cls=0.0527`
  - `payload_exact_acc_cls_cond_1pct=0.0501`

This validates that Stage-2 attribution supervision was a real bottleneck, but it is not the only remaining one.

---

## 3) Root Cause (Code + Why It Breaks)

### RC-A — **Mixed manifests + placeholder IDs create “constant identity” supervision**

`watermark/training/dataset.py` generates a fixed-shape `message` for all samples, including negatives and unlabeled items:

```py
# watermark/training/dataset.py
if has_labels:
    message = self.codec.encode(model_id, version).float()
else:
    # Historical footgun: unlabeled positives were mapped to a constant (0,0).
    # This has since been hardened to avoid a constant payload for unlabeled
    # watermarked items.
```

If Stage 2 (encoder training) implicitly “trains message for every sample,” then **unlabeled/negative items are treated as `(0,0)`**. The encoder/decoder can converge to “presence-only + constant identity.”

### RC-B — **Multi-task shortcut: detection is easier than identity bits**

Even with correct labels, a shared backbone can learn a strong watermark-presence feature and never allocate sufficient capacity to reliably encode/decode identity bits unless:
- the losses are balanced carefully, and/or
- you decouple detection from message decoding, and/or
- you use coding redundancy / better aggregation for message bits.

### RC-C — **Evaluation without conditioning on detector acceptance makes attribution look worse (and less realistic)**

Real usage is: detect → if accepted, decode/attribute. The repo now reports “conditional attribution” using the 1% FPR detection threshold:

```py
# watermark/evaluation/probe.py
thr_1pct = float(np.quantile(neg_scores, 0.99))
accepted = [r for r in pos_pred_records if float(r["score"]) >= thr_1pct]
metrics["payload_exact_acc_cls_cond_1pct"] = float(exact_ok / len(accepted))
```

This matters when clean audio hallucinates IDs (especially when `preamble_neg_avg` is high).

---

## 4) Fixes Implemented in This Repo (Code Snippets)

### 4.1 Stage-2 message sampling: avoid constant default IDs

Stage 2 now samples a balanced random `(model_id, version)` when labels are missing/invalid (and supports mixed batches safely):

```py
# watermark/training/stage2.py
batch_model = batch.get("model_id", torch.full((B,), -1, dtype=torch.long)).to(device)
batch_ver = batch.get("version", torch.full((B,), -1, dtype=torch.long)).to(device)
ids_valid = (batch_model >= 0) & (batch_ver >= 0)

rand_model, rand_ver, _ = sample_pair_ids(B, n_models=n_models, n_versions=n_versions, device=device)
target_model = torch.where(ids_valid, batch_model, rand_model)
target_ver = torch.where(ids_valid, batch_ver, rand_ver)
```

### 4.2 Stage-2 payload-loss gating toggle (`--stage2_payload_on_all`)

Stage 2 watermarks *all* carriers on-the-fly, so you often want payload losses to apply to the whole batch even if the manifest contains negatives.

```py
# watermark/training/stage2.py
if payload_pos_only and ("has_watermark" in batch):
    pos_mask = (batch["has_watermark"].to(device) > 0.5)
else:
    pos_mask = torch.ones((B,), device=device, dtype=torch.bool)
```

This is exposed as:
- `watermark/scripts/quick_voice_smoke_train.py`: `--stage2_payload_on_all`
- `watermark/scripts/train_full.py`: `--stage2_payload_on_all`

And the newest run `outputs/dashboard_runs/1769281044_9074a4/` shows this improves attribution vs baseline.

### 4.3 Joint `(model_id, version)` “pair head” (128-way)

The decoder contains:
- `head_model`: `n_models + 1` (includes unknown class)
- `head_version`: `n_versions + 1` (includes unknown class)
- `head_pair`: `n_models * n_versions` (128-way joint head)

```py
# watermark/models/decoder.py
self.head_model = nn.Linear(feat_dim, n_models + 1)
self.head_version = nn.Linear(feat_dim, n_versions + 1)
self.head_pair = nn.Linear(feat_dim, (n_models * n_versions))
```

The pair head prevents the “loophole” where the network partially encodes version but ignores model ID (or vice versa).

### 4.4 “Unknown” supervision on clean negatives (Stage 1B)

Stage 1B explicitly trains clean audio to map to the unknown class (for model/version heads), reducing confident hallucinations on negatives:

```py
# watermark/training/stage1b.py
unk = torch.full((top_model_logits_neg.shape[0],), n_classes - 1, device=device, dtype=torch.long)
loss_unk_model_ce = F.cross_entropy(top_model_logits_neg, unk)
loss = loss + unknown_ce_weight * loss_unk_model_ce
```

### 4.5 Probe metrics now include “conditional attribution”

To reflect real usage (detect → then attribute), probe metrics now log conditional accuracies at the strict `thr_at_fpr_1pct` operating point:
- `model_id_acc_cls_cond_1pct`
- `version_acc_cls_cond_1pct`
- `payload_exact_acc_cls_cond_1pct`
- `pair_acc_cls_cond_1pct`

See: `watermark/evaluation/probe.py`.

---

## 5) Why “Model ID Head” Still Isn’t Reliable Yet (What Data Says)

After the Stage-2 fix, the newest run improved attribution but it is still far below usable:
- Baseline 2k (`1769272369_a2d2ae`): `pair_acc_cls=0.0156`
- Fixed 2k (`1769281044_9074a4`): `pair_acc_cls=0.0527`

That’s a real jump, but still close to chance for 128 classes (`1/128 ≈ 0.0078`) and not reliable for real attribution.

Additionally, the confusion matrices for the latest run show **class imbalance/collapse** behavior (predictions concentrate on a few columns; unknown isn’t being used by positives).

This indicates the remaining bottlenecks are likely:
- loss balancing / curriculum (identity first, robustness later),
- message redundancy / ECC or a more message-friendly readout,
- better separation of detection and attribution gradients (avoid detection dominating).

---

## 6) Recommended Next Experiments (Evidence-Driven)

Run these in order, using the dashboard and keeping `probe_clips >= 1024`:

1) **Identity acquisition (no attacks)**
   - `--reverb_prob 0.0`
   - keep `--stage2_payload_on_all`
   - increase payload learning time: e.g. `--epochs_s2 24` and `--epochs_s1b_post 24`
   - Pass signal: `pair_acc_cls` should climb noticeably above ~0.05 on clean probe.

2) **Robustness ramp (only after identity works clean)**
   - gradually set `--reverb_prob 0.1` → `0.2` → `0.25`
   - keep payload supervised on clean watermarked audio (current Stage 2 does `payload_on_clean=True` by default).

3) **If attribution stays near-chance on clean**
   - treat it as an architecture/objective problem:
     - stronger per-bit aggregation (time evidence accumulation),
     - coding redundancy / ECC for the identity payload,
     - or a stronger decoupling: detection head independent of message decode.

---

## 7) How to Reproduce the Current “Best Known” Settings Quickly

Dashboard run (recommended) corresponds to:

```bash
./.venv/bin/python -m watermark.scripts.quick_voice_smoke_train \
  --source_dir mini_benchmark_data \
  --num_clips 2048 \
  --epochs_s1 6 --epochs_s1b 1 --epochs_s2 12 --epochs_s1b_post 12 \
  --msg_weight 3 \
  --model_ce_weight 4 --version_ce_weight 2 --pair_ce_weight 4 \
  --unknown_ce_weight 1 --neg_weight 5 --neg_preamble_target 0.5 \
  --reverb_prob 0.0 \
  --probe_clips 1024 --probe_every 1 --probe_reverb_every 999999 \
  --log_steps_every 25 \
  --stage2_payload_on_all
```

Artifacts and where to look:
- `.../metrics.jsonl`: charts + conditional metrics
- `.../audio/decode_report.txt`: single-clip qualitative summary
- `.../report.md`: summarized “best vs latest” probe stats (controller runs)
