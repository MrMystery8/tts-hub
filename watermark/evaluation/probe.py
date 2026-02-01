from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from watermark.config import BUDGET_TARGET_DB, CLASS_CLEAN, N_CLASSES
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


_ATTACK_METRIC_KEYS: tuple[str, ...] = (
    "mini_auc",
    "detect_pos_mean",
    "detect_neg_mean",
    "thr_at_fpr_1pct",
    "tpr_at_fpr_1pct",
    "attr_acc",
    "wm_acc",
    "id_acc_pos",
    "pred_clean_rate",
    "pred_pos_rate",
    "p_clean_mean",
    "p_clean_pos_mean",
    "p_clean_neg_mean",
    "wm_prob_loc_mean",
    "wm_prob_loc_pos_mean",
    "wm_prob_loc_neg_mean",
)


def _probe_once(
    items_list: list[ProbeItem],
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    attack_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_items: Optional[int] = None,
    num_classes: int = N_CLASSES,
    include_confusion: bool = True,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    import numpy as np

    y_true_wm: list[float] = []
    y_score: list[float] = []
    y_true_class: list[int] = []
    y_pred_id: list[int] = []  # argmax id head (0..K-1), meaningless for clean but still recorded
    p_clean: list[float] = []
    win_wm_scores: list[float] = []

    wm_snr_db: list[float] = []
    wm_delta_peak_abs: list[float] = []

    n = 0
    for it in items_list:
        n += 1
        if max_items is not None and n > int(max_items):
            break

        y = int(it.y_class)
        clip = _to_device_audio_bt(it.audio, device)
        y_t = torch.tensor([y], device=device, dtype=torch.long)

        with torch.no_grad():
            inp = encoder(clip, y_t) if y != int(CLASS_CLEAN) else clip

        if y != int(CLASS_CLEAN):
            orig_ct = clip.squeeze(0).detach().cpu().to(dtype=torch.float32)
            wm_ct = inp.squeeze(0).detach().cpu().to(dtype=torch.float32)
            delta = wm_ct - orig_ct
            power_orig = float(orig_ct.pow(2).mean().item())
            power_delta = float(delta.pow(2).mean().item())
            eps = 1e-12
            snr = 10.0 * float(np.log10((power_orig + eps) / (power_delta + eps)))
            wm_snr_db.append(snr)
            wm_delta_peak_abs.append(float(delta.abs().max().item()))

        if attack_fn is not None:
            attacked = apply_attack_safe(inp.squeeze(0).detach().cpu(), attack_fn).unsqueeze(0).to(device=device, dtype=torch.float32)
            inp_for_dec = attacked
        else:
            inp_for_dec = inp

        with torch.no_grad():
            out = decoder(inp_for_dec)

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

    if y_true_class:
        thr_use = float(thr_1pct) if thr_1pct is not None else 0.5
        y_pred_class_thr: list[int] = []
        for s, pid in zip(y_score, y_pred_id):
            y_pred_class_thr.append(int(pid + 1) if float(s) >= thr_use else int(CLASS_CLEAN))

        correct = [1.0 if t == p else 0.0 for t, p in zip(y_true_class, y_pred_class_thr)]
        if correct:
            metrics["attr_acc"] = float(np.mean(correct))
            wm_idx = [i for i, y in enumerate(y_true_class) if y != int(CLASS_CLEAN)]
            if wm_idx:
                metrics["wm_acc"] = float(np.mean([correct[i] for i in wm_idx]))
                id_correct = [1.0 if (y_pred_id[i] == (int(y_true_class[i]) - 1)) else 0.0 for i in wm_idx]
                metrics["id_acc_pos"] = float(np.mean(id_correct))

        metrics["pred_clean_rate"] = float(np.mean([1.0 if p == int(CLASS_CLEAN) else 0.0 for p in y_pred_class_thr]))
        metrics["pred_pos_rate"] = 1.0 - float(metrics["pred_clean_rate"])
        metrics["thr_used_for_confusion"] = float(thr_use)

        if include_confusion:
            cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long)
            for t, p in zip(y_true_class, y_pred_class_thr):
                if 0 <= t < int(num_classes) and 0 <= p < int(num_classes):
                    cm[t, p] += 1
            metrics["confusion"] = cm.tolist()

            wm_idx = [i for i, y in enumerate(y_true_class) if y != int(CLASS_CLEAN)]
            if wm_idx:
                k = int(num_classes) - 1
                cm_id = torch.zeros((k, k), dtype=torch.long)
                for i in wm_idx:
                    true_id = int(y_true_class[i]) - 1
                    pred_id = int(y_pred_id[i])
                    if 0 <= true_id < k and 0 <= pred_id < k:
                        cm_id[true_id, pred_id] += 1
                metrics["confusion_attr"] = cm_id.tolist()

    if p_clean:
        metrics["p_clean_mean"] = float(np.mean(p_clean))
        pos_p0 = [p for p, y in zip(p_clean, y_true_wm) if y == 1.0]
        neg_p0 = [p for p, y in zip(p_clean, y_true_wm) if y == 0.0]
        if pos_p0:
            metrics["p_clean_pos_mean"] = float(np.mean(pos_p0))
        if neg_p0:
            metrics["p_clean_neg_mean"] = float(np.mean(neg_p0))

    if win_wm_scores:
        metrics["wm_prob_loc_mean"] = float(np.mean(win_wm_scores))
        pos_loc = [s for s, y in zip(win_wm_scores, y_true_wm) if y == 1.0]
        neg_loc = [s for s, y in zip(win_wm_scores, y_true_wm) if y == 0.0]
        if pos_loc:
            metrics["wm_prob_loc_pos_mean"] = float(np.mean(pos_loc))
        if neg_loc:
            metrics["wm_prob_loc_neg_mean"] = float(np.mean(neg_loc))

    if wm_snr_db:
        target_snr_db = float(-float(BUDGET_TARGET_DB))
        snrs = np.array(wm_snr_db, dtype=np.float32)
        metrics["wm_budget_target_db"] = float(BUDGET_TARGET_DB)
        metrics["wm_snr_db_mean"] = float(np.mean(snrs))
        metrics["wm_snr_db_p10"] = float(np.quantile(snrs, 0.10))
        metrics["wm_snr_db_p50"] = float(np.quantile(snrs, 0.50))
        metrics["wm_snr_db_p90"] = float(np.quantile(snrs, 0.90))
        metrics["wm_budget_ok_frac"] = float(np.mean(snrs >= target_snr_db))
        metrics["wm_delta_power_db_mean"] = float(-metrics["wm_snr_db_mean"])

    if wm_delta_peak_abs:
        peaks = np.array(wm_delta_peak_abs, dtype=np.float32)
        metrics["wm_delta_peak_abs_mean"] = float(np.mean(peaks))
        metrics["wm_delta_peak_abs_p90"] = float(np.quantile(peaks, 0.90))

    return metrics


def compute_probe_metrics(
    items: Iterable[ProbeItem],
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    compute_reverb: bool = True,
    max_items: Optional[int] = None,
    num_classes: int = N_CLASSES,
    extra_attacks: Optional[Iterable[str]] = None,
    include_confusion: bool = True,
    # Compatibility: older call sites passed a codec object; ignored in multiclass mode.
    codec: Any = None,
) -> dict[str, Any]:
    """
    Lightweight mid-training probe for multiclass attribution.

    Base (clean) metrics are always computed.
    Optionally computes attack metrics (reverb + any names in `extra_attacks`) by applying the attack
    AFTER embedding (for positives) and BEFORE decoding.
    """
    _ = codec
    enc_was_training = encoder.training
    dec_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    items_list = list(items)

    metrics = _probe_once(
        items_list,
        encoder=encoder,
        decoder=decoder,
        device=device,
        attack_fn=None,
        max_items=max_items,
        num_classes=num_classes,
        include_confusion=bool(include_confusion),
    )

    def merge_attack_metrics(suffix: str, attacked: dict[str, Any]) -> None:
        for k in _ATTACK_METRIC_KEYS:
            if k in attacked:
                metrics[f"{k}{suffix}"] = attacked[k]

    if compute_reverb and "reverb" in ATTACKS:
        m_r = _probe_once(
            items_list,
            encoder=encoder,
            decoder=decoder,
            device=device,
            attack_fn=ATTACKS["reverb"],
            max_items=max_items,
            num_classes=num_classes,
            include_confusion=False,
        )
        merge_attack_metrics("_reverb", m_r)

    if extra_attacks:
        for name in extra_attacks:
            a = str(name).strip()
            if not a or a in {"clean", "reverb"}:
                continue
            fn = ATTACKS.get(a)
            if fn is None:
                continue
            m_a = _probe_once(
                items_list,
                encoder=encoder,
                decoder=decoder,
                device=device,
                attack_fn=fn,
                max_items=max_items,
                num_classes=num_classes,
                include_confusion=False,
            )
            merge_attack_metrics(f"_{a}", m_a)

    encoder.train(enc_was_training)
    decoder.train(dec_was_training)
    return metrics

