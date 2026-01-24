from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from watermark.evaluation.attacks import ATTACKS, apply_attack_safe
from watermark.config import N_MODELS, N_VERSIONS


@dataclass(frozen=True)
class ProbeItem:
    audio: torch.Tensor  # (1, T)
    has_watermark: float
    message: torch.Tensor  # (32,)
    model_id: int
    version: int


def _to_device_audio_bt(audio_ct: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert (1, T) -> (1, 1, T) on device.
    """
    return audio_ct.unsqueeze(0).to(device=device, dtype=torch.float32)


def compute_probe_metrics(
    items: Iterable[ProbeItem],
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    codec: Any,
    device: torch.device,
    compute_reverb: bool = True,
    max_items: Optional[int] = None,
) -> dict[str, Any]:
    """
    Lightweight probe for mid-training monitoring.

    Computes (on a small fixed probe set):
    - Detection AUC (clean vs watermarked)
    - Preamble pos/neg averages
    - Attribution accuracy from classification heads (if present)
    - Optional AUC under reverb (CPU attack -> decode on device)
    """
    enc_was_training = encoder.training
    dec_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    items_list = list(items)

    # Lazy import (keeps import cost low if unused)
    from sklearn.metrics import roc_auc_score

    # Optional numpy for percentiles/quantiles (already a dependency elsewhere in repo).
    import numpy as np

    y_true: list[float] = []
    y_score: list[float] = []
    preamble_scores: list[float] = []
    preamble_scores_pos: list[float] = []
    preamble_scores_neg: list[float] = []

    pos_total = 0
    pos_total_labeled = 0
    pos_model_cls_correct = 0
    pos_version_cls_correct = 0
    pos_exact_cls = 0
    pos_pair_cls_correct = 0
    saw_cls_heads = False
    saw_pair_head = False
    model_confusion: Optional[torch.Tensor] = None  # (N_MODELS, n_model_classes)
    version_confusion: Optional[torch.Tensor] = None  # (N_VERSIONS, n_version_classes)
    model_unknown = 0
    version_unknown = 0

    # Bit decode (legacy)
    pos_model_bits_correct = 0.0
    pos_model_bits_total = 0.0
    pos_version_bits_correct = 0.0
    pos_version_bits_total = 0.0

    # Store labeled-positive per-item predictions so we can report attribution conditioned
    # on detector acceptance (e.g. at the 1% FPR threshold).
    pos_pred_records: list[dict[str, int | float | None]] = []

    n = 0
    for it in items_list:
        n += 1
        if max_items is not None and n > max_items:
            break

        label = float(it.has_watermark)
        clip = _to_device_audio_bt(it.audio, device)
        msg = it.message.unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            wm = encoder(clip, msg)
            inp = wm if label == 1.0 else clip
            out = decoder(inp)

        score = float(out["clip_detect_prob"].item())
        y_true.append(label)
        y_score.append(score)

        msg_probs = out.get("avg_message_probs")
        if msg_probs is not None:
            decoded = codec.decode(msg_probs.squeeze(0))
            preamble_scores.append(decoded["preamble_score"])
            if label == 1.0:
                preamble_scores_pos.append(decoded["preamble_score"])
            else:
                preamble_scores_neg.append(decoded["preamble_score"])

        if label == 1.0:
            pos_total += 1
            labeled = (it.model_id >= 0) and (it.version >= 0)
            if labeled:
                pos_total_labeled += 1

            # Classification heads (preferred)
            if "avg_pair_logits" in out:
                saw_cls_heads = True
                saw_pair_head = True
                p = out["avg_pair_logits"].squeeze(0)
                pred_pair = int(torch.argmax(p).item())
                if labeled and pred_pair == (it.version * N_MODELS + it.model_id):
                    pos_pair_cls_correct += 1
                    pos_exact_cls += 1
                if labeled:
                    pred_model = int(pred_pair % N_MODELS)
                    pred_ver = int(pred_pair // N_MODELS)
                    if model_confusion is None:
                        model_confusion = torch.zeros((N_MODELS, N_MODELS + 1), dtype=torch.long)
                    if version_confusion is None:
                        version_confusion = torch.zeros((N_VERSIONS, N_VERSIONS + 1), dtype=torch.long)
                    model_confusion[it.model_id, pred_model] += 1
                    version_confusion[it.version, pred_ver] += 1
                    if pred_model == it.model_id:
                        pos_model_cls_correct += 1
                    if pred_ver == it.version:
                        pos_version_cls_correct += 1
                    pos_pred_records.append(
                        {
                            "score": score,
                            "pred_pair": pred_pair,
                            "pred_model": pred_model,
                            "pred_ver": pred_ver,
                            "true_pair": int(it.version * N_MODELS + it.model_id),
                            "true_model": int(it.model_id),
                            "true_ver": int(it.version),
                        }
                    )
            elif "avg_model_logits" in out and "avg_version_logits" in out:
                saw_cls_heads = True
                m = out["avg_model_logits"].squeeze(0)
                v = out["avg_version_logits"].squeeze(0)
                pred_model = int(torch.argmax(m).item())
                pred_ver = int(torch.argmax(v).item())
                if model_confusion is None:
                    model_confusion = torch.zeros((N_MODELS, int(m.numel())), dtype=torch.long)
                if version_confusion is None:
                    version_confusion = torch.zeros((N_VERSIONS, int(v.numel())), dtype=torch.long)
                if labeled and 0 <= it.model_id < model_confusion.shape[0] and 0 <= pred_model < model_confusion.shape[1]:
                    model_confusion[it.model_id, pred_model] += 1
                if labeled and 0 <= it.version < version_confusion.shape[0] and 0 <= pred_ver < version_confusion.shape[1]:
                    version_confusion[it.version, pred_ver] += 1
                if pred_model == (model_confusion.shape[1] - 1):
                    model_unknown += 1
                if pred_ver == (version_confusion.shape[1] - 1):
                    version_unknown += 1
                if labeled and pred_model == it.model_id:
                    pos_model_cls_correct += 1
                if labeled and pred_ver == it.version:
                    pos_version_cls_correct += 1
                if labeled and pred_model == it.model_id and pred_ver == it.version:
                    pos_exact_cls += 1
                if labeled:
                    pos_pred_records.append(
                        {
                            "score": score,
                            "pred_pair": None,
                            "pred_model": pred_model,
                            "pred_ver": pred_ver,
                            "true_pair": int(it.version * N_MODELS + it.model_id),
                            "true_model": int(it.model_id),
                            "true_ver": int(it.version),
                        }
                    )

            # Bit-level accuracy split by fields (to avoid constant-field illusions)
            if msg_probs is not None and labeled:
                probs = msg_probs.squeeze(0)
                model_idx = torch.tensor([16, 17, 18, 23, 24, 25], device=probs.device)
                ver_idx = torch.tensor([19, 20, 21, 22, 26, 27, 28, 29], device=probs.device)
                target_bits = it.message.to(probs.device)

                pred_model_bits = (probs.index_select(0, model_idx) > 0.5).float()
                tgt_model_bits = target_bits.index_select(0, model_idx)
                pos_model_bits_correct += (pred_model_bits == tgt_model_bits).float().sum().item()
                pos_model_bits_total += float(tgt_model_bits.numel())

                pred_ver_bits = (probs.index_select(0, ver_idx) > 0.5).float()
                tgt_ver_bits = target_bits.index_select(0, ver_idx)
                pos_version_bits_correct += (pred_ver_bits == tgt_ver_bits).float().sum().item()
                pos_version_bits_total += float(tgt_ver_bits.numel())

    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
        metrics: dict[str, Any] = {
            "mini_auc": float(auc),
            "n_items": len(y_true),
        }

        # Detection score distribution stats (helps interpret AUC when raw probs are low).
        pos_scores = [s for s, y in zip(y_score, y_true) if y == 1.0]
        neg_scores = [s for s, y in zip(y_score, y_true) if y == 0.0]
        if pos_scores:
            metrics["detect_pos_mean"] = float(np.mean(pos_scores))
            metrics["detect_pos_p10"] = float(np.percentile(pos_scores, 10))
            metrics["detect_pos_p50"] = float(np.percentile(pos_scores, 50))
            metrics["detect_pos_p90"] = float(np.percentile(pos_scores, 90))
        if neg_scores:
            metrics["detect_neg_mean"] = float(np.mean(neg_scores))
            metrics["detect_neg_p10"] = float(np.percentile(neg_scores, 10))
            metrics["detect_neg_p50"] = float(np.percentile(neg_scores, 50))
            metrics["detect_neg_p90"] = float(np.percentile(neg_scores, 90))

        # Operating-point metric: TPR at 1% FPR (useful mid-training stop/go signal).
        if pos_scores and neg_scores:
            thr_1pct = float(np.quantile(neg_scores, 0.99))
            tpr_1pct = float(np.mean([1.0 if s >= thr_1pct else 0.0 for s in pos_scores]))
            metrics["thr_at_fpr_1pct"] = thr_1pct
            metrics["tpr_at_fpr_1pct"] = tpr_1pct

            # Attribution conditioned on detector acceptance at the strict threshold.
            # This is closer to real usage: first decide "watermarked?", then attribute.
            if pos_total_labeled > 0 and pos_pred_records:
                accepted = [r for r in pos_pred_records if float(r["score"]) >= thr_1pct]
                metrics["n_pos_labeled_accept_1pct"] = int(len(accepted))
                metrics["pos_accept_rate_1pct"] = float(len(accepted) / max(1, pos_total_labeled))
                if accepted:
                    model_ok = sum(1 for r in accepted if int(r["pred_model"]) == int(r["true_model"]))
                    ver_ok = sum(1 for r in accepted if int(r["pred_ver"]) == int(r["true_ver"]))
                    exact_ok = sum(
                        1
                        for r in accepted
                        if (int(r["pred_model"]) == int(r["true_model"]) and int(r["pred_ver"]) == int(r["true_ver"]))
                    )
                    metrics["model_id_acc_cls_cond_1pct"] = float(model_ok / len(accepted))
                    metrics["version_acc_cls_cond_1pct"] = float(ver_ok / len(accepted))
                    metrics["payload_exact_acc_cls_cond_1pct"] = float(exact_ok / len(accepted))
                    if saw_pair_head:
                        pair_ok = sum(1 for r in accepted if r["pred_pair"] is not None and int(r["pred_pair"]) == int(r["true_pair"]))
                        metrics["pair_acc_cls_cond_1pct"] = float(pair_ok / len(accepted))

        if preamble_scores:
            metrics["preamble_avg"] = float(sum(preamble_scores) / len(preamble_scores))
        if preamble_scores_pos:
            metrics["preamble_pos_avg"] = float(sum(preamble_scores_pos) / len(preamble_scores_pos))
        if preamble_scores_neg:
            metrics["preamble_neg_avg"] = float(sum(preamble_scores_neg) / len(preamble_scores_neg))

        if pos_total_labeled > 0 and saw_cls_heads:
            metrics["model_id_acc_cls"] = float(pos_model_cls_correct / pos_total_labeled)
            metrics["version_acc_cls"] = float(pos_version_cls_correct / pos_total_labeled)
            metrics["payload_exact_acc_cls"] = float(pos_exact_cls / pos_total_labeled)
            if saw_pair_head:
                metrics["pair_acc_cls"] = float(pos_pair_cls_correct / pos_total_labeled)
            if model_confusion is not None:
                metrics["model_confusion"] = model_confusion.cpu().tolist()
                metrics["model_unknown_rate"] = float(model_unknown / max(1, pos_total_labeled))
            if version_confusion is not None:
                metrics["version_confusion"] = version_confusion.cpu().tolist()
                metrics["version_unknown_rate"] = float(version_unknown / max(1, pos_total_labeled))

        if pos_model_bits_total > 0:
            metrics["model_id_bit_acc"] = float(pos_model_bits_correct / pos_model_bits_total)
        if pos_version_bits_total > 0:
            metrics["version_bit_acc"] = float(pos_version_bits_correct / pos_version_bits_total)

        if compute_reverb and "reverb" in ATTACKS and len(y_true) > 1:
            y_score_reverb: list[float] = []
            for it in items_list:
                if max_items is not None and len(y_score_reverb) >= max_items:
                    break
                label = float(it.has_watermark)
                clip = _to_device_audio_bt(it.audio, device)
                msg = it.message.unsqueeze(0).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    wm = encoder(clip, msg)
                    inp = wm if label == 1.0 else clip

                attacked_ct = apply_attack_safe(inp.squeeze(0).cpu(), ATTACKS["reverb"])
                attacked_bt = attacked_ct.unsqueeze(0).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    out = decoder(attacked_bt)
                y_score_reverb.append(float(out["clip_detect_prob"].item()))

            auc_reverb = roc_auc_score(y_true[: len(y_score_reverb)], y_score_reverb) if len(set(y_true)) > 1 else 0.5
            metrics["mini_auc_reverb"] = float(auc_reverb)
            pos_scores = [s for s, y in zip(y_score_reverb, y_true) if y == 1.0]
            neg_scores = [s for s, y in zip(y_score_reverb, y_true) if y == 0.0]
            if pos_scores:
                metrics["reverb_pos_mean"] = float(sum(pos_scores) / len(pos_scores))
            if neg_scores:
                metrics["reverb_neg_mean"] = float(sum(neg_scores) / len(neg_scores))
            if pos_scores:
                metrics["reverb_pos_p10"] = float(np.percentile(pos_scores, 10))
                metrics["reverb_pos_p50"] = float(np.percentile(pos_scores, 50))
                metrics["reverb_pos_p90"] = float(np.percentile(pos_scores, 90))
            if neg_scores:
                metrics["reverb_neg_p10"] = float(np.percentile(neg_scores, 10))
                metrics["reverb_neg_p50"] = float(np.percentile(neg_scores, 50))
                metrics["reverb_neg_p90"] = float(np.percentile(neg_scores, 90))
            if pos_scores and neg_scores:
                thr_1pct = float(np.quantile(neg_scores, 0.99))
                tpr_1pct = float(np.mean([1.0 if s >= thr_1pct else 0.0 for s in pos_scores]))
                metrics["thr_at_fpr_1pct_reverb"] = thr_1pct
                metrics["tpr_at_fpr_1pct_reverb"] = tpr_1pct

        return metrics
    finally:
        encoder.train(enc_was_training)
        decoder.train(dec_was_training)
