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
    y_pred_id: list[int] = []  # argmax id head (0..K-1), meaningless for clean but still recorded
    p_clean: list[float] = []
    win_wm_scores: list[float] = []

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
        win_score = out.get("clip_wm_prob_loc", out.get("clip_wm_prob", out.get("clip_detect_prob")))
        if win_score is not None:
            win_wm_scores.append(float(win_score.item()) if hasattr(win_score, "item") else float(win_score))
        id_logits = out.get("clip_id_logits")
        if id_logits is None:
            # Back-compat: fall back to multiclass logits and map to id space.
            logits = out.get("clip_class_logits")
            if logits is None:
                raise KeyError("decoder output missing clip_id_logits or clip_class_logits")
            pred_class = int(torch.argmax(logits, dim=-1).item())
            pred_id = max(0, pred_class - 1)
            probs = out.get("clip_class_probs")
            if probs is None:
                probs = torch.softmax(logits, dim=-1)
            p0 = float(probs[0, int(CLASS_CLEAN)].item())
        else:
            pred_id = int(torch.argmax(id_logits, dim=-1).item())
            class_probs = out.get("clip_class_probs")
            if class_probs is None:
                # derive via detect + id if present; otherwise approximate from class logits
                logits = out.get("clip_class_logits")
                if logits is None:
                    raise KeyError("decoder output missing clip_class_probs/clip_class_logits")
                class_probs = torch.softmax(logits, dim=-1)
            p0 = float(class_probs[0, int(CLASS_CLEAN)].item())

        y_true_wm.append(1.0 if y != int(CLASS_CLEAN) else 0.0)
        y_score.append(score)
        y_true_class.append(y)
        y_pred_id.append(pred_id)
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

    thr_1pct = None
    if pos_scores and neg_scores:
        thr_1pct = float(np.quantile(neg_scores, 0.99))
        tpr_1pct = float(np.mean([1.0 if s >= thr_1pct else 0.0 for s in pos_scores]))
        metrics["thr_at_fpr_1pct"] = thr_1pct
        metrics["tpr_at_fpr_1pct"] = tpr_1pct

    # Attribution metrics
    if y_true_class:
        # Thresholded (product-style) prediction: if not detected -> clean else id+1
        thr = float(thr_1pct) if thr_1pct is not None else 0.5
        y_pred_class_thr: list[int] = []
        for s, pid in zip(y_score, y_pred_id):
            y_pred_class_thr.append(int(pid + 1) if float(s) >= thr else int(CLASS_CLEAN))

        correct_thr = [1.0 if t == p else 0.0 for t, p in zip(y_true_class, y_pred_class_thr)]
        metrics["attr_acc"] = float(np.mean(correct_thr))

        wm_idx = [i for i, y in enumerate(y_true_class) if y != int(CLASS_CLEAN)]
        if wm_idx:
            metrics["wm_acc"] = float(np.mean([correct_thr[i] for i in wm_idx]))
            # ID-only accuracy on positives, independent of detection threshold
            id_correct = [1.0 if (y_pred_id[i] == (int(y_true_class[i]) - 1)) else 0.0 for i in wm_idx]
            metrics["id_acc_pos"] = float(np.mean(id_correct))

        metrics["pred_clean_rate"] = float(np.mean([1.0 if p == int(CLASS_CLEAN) else 0.0 for p in y_pred_class_thr]))
        metrics["pred_pos_rate"] = 1.0 - float(metrics["pred_clean_rate"])
        metrics["thr_used_for_confusion"] = float(thr)

        # Full (K+1)x(K+1) confusion (thresholded prediction)
        cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long)
        for t, p in zip(y_true_class, y_pred_class_thr):
            if 0 <= t < int(num_classes) and 0 <= p < int(num_classes):
                cm[t, p] += 1
        metrics["confusion"] = cm.tolist()

        # Attribution-only confusion (KxK) on watermarked subset (no clean row/col).
        if wm_idx:
            k = int(num_classes) - 1
            cm_id = torch.zeros((k, k), dtype=torch.long)
            for i in wm_idx:
                true_id = int(y_true_class[i]) - 1
                pred_id = int(y_pred_id[i])
                if 0 <= true_id < k and 0 <= pred_id < k:
                    cm_id[true_id, pred_id] += 1
            metrics["confusion_attr"] = cm_id.tolist()

    # P(clean) diagnostics
    if p_clean:
        metrics["p_clean_mean"] = float(np.mean(p_clean))
        pos_p0 = [p for p, y in zip(p_clean, y_true_wm) if y == 1.0]
        neg_p0 = [p for p, y in zip(p_clean, y_true_wm) if y == 0.0]
        if pos_p0:
            metrics["p_clean_pos_mean"] = float(np.mean(pos_p0))
        if neg_p0:
            metrics["p_clean_neg_mean"] = float(np.mean(neg_p0))

    # Localization pooled score diagnostics (if available)
    if win_wm_scores:
        metrics["wm_prob_loc_mean"] = float(np.mean(win_wm_scores))
        pos_loc = [s for s, y in zip(win_wm_scores, y_true_wm) if y == 1.0]
        neg_loc = [s for s, y in zip(win_wm_scores, y_true_wm) if y == 0.0]
        if pos_loc:
            metrics["wm_prob_loc_pos_mean"] = float(np.mean(pos_loc))
        if neg_loc:
            metrics["wm_prob_loc_neg_mean"] = float(np.mean(neg_loc))

    # Reverb probe (attack only, no re-encoding; attack is applied to the carrier used above)
    if compute_reverb and "reverb" in ATTACKS and y_true_class:
        y_score_r: list[float] = []
        y_pred_id_r: list[int] = []
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
            id_logits_r = out_r.get("clip_id_logits")
            if id_logits_r is None:
                logits_r = out_r["clip_class_logits"]
                pred_class_r = int(torch.argmax(logits_r, dim=-1).item())
                y_pred_id_r.append(max(0, pred_class_r - 1))
            else:
                y_pred_id_r.append(int(torch.argmax(id_logits_r, dim=-1).item()))

        if len(set(y_true_wm)) > 1 and y_score_r:
            metrics["mini_auc_reverb"] = float(roc_auc_score(y_true_wm[: len(y_score_r)], y_score_r))

        # Thresholded predictions under reverb (use reverb-neg threshold if available)
        pos_scores_r = [s for s, y in zip(y_score_r, y_true_wm[: len(y_score_r)]) if y == 1.0]
        neg_scores_r = [s for s, y in zip(y_score_r, y_true_wm[: len(y_score_r)]) if y == 0.0]
        thr_r = None
        if pos_scores_r and neg_scores_r:
            thr_r = float(np.quantile(neg_scores_r, 0.99))
            tpr_1pct_r = float(np.mean([1.0 if s >= thr_r else 0.0 for s in pos_scores_r]))
            metrics["thr_at_fpr_1pct_reverb"] = thr_r
            metrics["tpr_at_fpr_1pct_reverb"] = tpr_1pct_r

        thr_use_r = float(thr_r) if thr_r is not None else 0.5
        y_true_r = y_true_class[: len(y_pred_id_r)]
        y_pred_class_thr_r: list[int] = []
        for s, pid in zip(y_score_r, y_pred_id_r):
            y_pred_class_thr_r.append(int(pid + 1) if float(s) >= thr_use_r else int(CLASS_CLEAN))

        correct_r = [1.0 if t == p else 0.0 for t, p in zip(y_true_r, y_pred_class_thr_r)]
        if correct_r:
            metrics["attr_acc_reverb"] = float(np.mean(correct_r))
            wm_idx_r = [i for i, y in enumerate(y_true_r) if y != int(CLASS_CLEAN)]
            if wm_idx_r:
                metrics["wm_acc_reverb"] = float(np.mean([correct_r[i] for i in wm_idx_r]))
                id_correct_r = [1.0 if (y_pred_id_r[i] == (int(y_true_r[i]) - 1)) else 0.0 for i in wm_idx_r]
                metrics["id_acc_pos_reverb"] = float(np.mean(id_correct_r))

    encoder.train(enc_was_training)
    decoder.train(dec_was_training)
    return metrics
