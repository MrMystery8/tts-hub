from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from watermark.config import CLASS_CLEAN, N_CLASSES
from watermark.evaluation.attacks import ATTACKS, apply_attack_safe


@dataclass(frozen=True)
class ProbeItem:
    audio: torch.Tensor  # (1, T)
    y_class: int  # 0=clean, 1..K=model classes


def _to_device_audio_bt(audio_ct: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert (1, T) -> (1, 1, T) on device."""
    return audio_ct.unsqueeze(0).to(device=device, dtype=torch.float32)


def _wm_score_from_out(out: dict[str, Any]) -> float:
    score = out.get("clip_wm_prob", out.get("clip_detect_prob"))
    if score is None:
        raise KeyError("decoder output missing clip_wm_prob/clip_detect_prob")
    return float(score.item()) if hasattr(score, "item") else float(score)


def compute_probe_metrics(
    items: Iterable[ProbeItem],
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    compute_reverb: bool = True,
    max_items: Optional[int] = None,
    num_classes: int = N_CLASSES,
    # Compatibility: older call sites passed a codec object; ignored in multiclass mode.
    codec: Any = None,
) -> dict[str, Any]:
    """
    Lightweight mid-training probe for multiclass attribution.

    - Watermark score: `1 - P(class0)` (aka `clip_wm_prob`)
    - Attribution: argmax class (0..K)
    """
    _ = codec
    enc_was_training = encoder.training
    dec_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    items_list = list(items)

    from sklearn.metrics import roc_auc_score
    import numpy as np

    y_true_wm: list[float] = []
    y_score: list[float] = []
    y_true_class: list[int] = []
    y_pred_class: list[int] = []
    p_clean: list[float] = []

    n = 0
    for it in items_list:
        n += 1
        if max_items is not None and n > int(max_items):
            break

        y = int(it.y_class)
        clip = _to_device_audio_bt(it.audio, device)
        y_t = torch.tensor([y], device=device, dtype=torch.long)

        with torch.no_grad():
            if y != int(CLASS_CLEAN):
                inp = encoder(clip, y_t)
            else:
                inp = clip
            out = decoder(inp)

        score = _wm_score_from_out(out)
        logits = out.get("clip_class_logits")
        if logits is None:
            raise KeyError("decoder output missing clip_class_logits")
        pred = int(torch.argmax(logits, dim=-1).item())
        probs = out.get("clip_class_probs")
        if probs is None:
            probs = torch.softmax(logits, dim=-1)
        p0 = float(probs[0, int(CLASS_CLEAN)].item())

        y_true_wm.append(1.0 if y != int(CLASS_CLEAN) else 0.0)
        y_score.append(score)
        y_true_class.append(y)
        y_pred_class.append(pred)
        p_clean.append(p0)

    # Binary detection metrics
    metrics: dict[str, Any] = {"n_items": int(len(y_true_wm)), "n_classes": int(num_classes)}
    if len(set(y_true_wm)) > 1:
        metrics["mini_auc"] = float(roc_auc_score(y_true_wm, y_score))
    else:
        metrics["mini_auc"] = 0.5

    pos_scores = [s for s, y in zip(y_score, y_true_wm) if y == 1.0]
    neg_scores = [s for s, y in zip(y_score, y_true_wm) if y == 0.0]
    if pos_scores:
        metrics["detect_pos_mean"] = float(np.mean(pos_scores))
    if neg_scores:
        metrics["detect_neg_mean"] = float(np.mean(neg_scores))

    if pos_scores and neg_scores:
        thr_1pct = float(np.quantile(neg_scores, 0.99))
        tpr_1pct = float(np.mean([1.0 if s >= thr_1pct else 0.0 for s in pos_scores]))
        metrics["thr_at_fpr_1pct"] = thr_1pct
        metrics["tpr_at_fpr_1pct"] = tpr_1pct

    # Multiclass attribution metrics
    if y_true_class:
        correct = [1.0 if a == b else 0.0 for a, b in zip(y_true_class, y_pred_class)]
        metrics["attr_acc"] = float(np.mean(correct))

        wm_idx = [i for i, y in enumerate(y_true_class) if y != int(CLASS_CLEAN)]
        if wm_idx:
            metrics["wm_acc"] = float(np.mean([correct[i] for i in wm_idx]))
        metrics["pred_clean_rate"] = float(np.mean([1.0 if p == int(CLASS_CLEAN) else 0.0 for p in y_pred_class]))

        cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long)
        for t, p in zip(y_true_class, y_pred_class):
            if 0 <= t < int(num_classes) and 0 <= p < int(num_classes):
                cm[t, p] += 1
        metrics["confusion"] = cm.tolist()

    # P(clean) diagnostics
    if p_clean:
        metrics["p_clean_mean"] = float(np.mean(p_clean))
        pos_p0 = [p for p, y in zip(p_clean, y_true_wm) if y == 1.0]
        neg_p0 = [p for p, y in zip(p_clean, y_true_wm) if y == 0.0]
        if pos_p0:
            metrics["p_clean_pos_mean"] = float(np.mean(pos_p0))
        if neg_p0:
            metrics["p_clean_neg_mean"] = float(np.mean(neg_p0))

    # Reverb probe (attack only, no re-encoding; attack is applied to the carrier used above)
    if compute_reverb and "reverb" in ATTACKS and y_true_class:
        y_score_r: list[float] = []
        y_pred_r: list[int] = []
        for it in items_list[: (int(max_items) if max_items is not None else len(items_list))]:
            y = int(it.y_class)
            clip = _to_device_audio_bt(it.audio, device)
            y_t = torch.tensor([y], device=device, dtype=torch.long)

            with torch.no_grad():
                inp = encoder(clip, y_t) if y != int(CLASS_CLEAN) else clip

            attacked = apply_attack_safe(inp.squeeze(0).cpu(), ATTACKS["reverb"]).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                out_r = decoder(attacked)

            y_score_r.append(_wm_score_from_out(out_r))
            logits_r = out_r["clip_class_logits"]
            y_pred_r.append(int(torch.argmax(logits_r, dim=-1).item()))

        if len(set(y_true_wm)) > 1 and y_score_r:
            metrics["mini_auc_reverb"] = float(roc_auc_score(y_true_wm[: len(y_score_r)], y_score_r))

        correct_r = [1.0 if t == p else 0.0 for t, p in zip(y_true_class[: len(y_pred_r)], y_pred_r)]
        if correct_r:
            metrics["attr_acc_reverb"] = float(np.mean(correct_r))
            wm_idx_r = [i for i, y in enumerate(y_true_class[: len(y_pred_r)]) if y != int(CLASS_CLEAN)]
            if wm_idx_r:
                metrics["wm_acc_reverb"] = float(np.mean([correct_r[i] for i in wm_idx_r]))

        pos_scores_r = [s for s, y in zip(y_score_r, y_true_wm[: len(y_score_r)]) if y == 1.0]
        neg_scores_r = [s for s, y in zip(y_score_r, y_true_wm[: len(y_score_r)]) if y == 0.0]
        if pos_scores_r and neg_scores_r:
            thr_1pct_r = float(np.quantile(neg_scores_r, 0.99))
            tpr_1pct_r = float(np.mean([1.0 if s >= thr_1pct_r else 0.0 for s in pos_scores_r]))
            metrics["thr_at_fpr_1pct_reverb"] = thr_1pct_r
            metrics["tpr_at_fpr_1pct_reverb"] = tpr_1pct_r

    encoder.train(enc_was_training)
    decoder.train(dec_was_training)
    return metrics

