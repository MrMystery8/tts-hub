"""
Stage 2 & 3: Encoder Training & Finetuning (Multiclass Attribution)

Stage 2 (default): train encoder only (decoder frozen).
Stage 3 (finetune_mode=True): finetune encoder + decoder together.

Loss terms:
- Detect BCE (window + clip)
- ID CE (window + clip) on positives only
- (optional, finetune only) Clean detect regularization to avoid collapse
- Quality (multi-resolution STFT)
- Energy budget
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from watermark.config import BUDGET_TARGET_DB, SAMPLE_RATE
from watermark.training.losses import CachedSTFTLoss, EnergyBudgetLoss, UncertaintyLossWrapper


class DifferentiableAugmenter:
    """Only transforms that preserve gradient flow."""

    def __init__(self, device: torch.device, sample_rate: int = SAMPLE_RATE, reverb_prob: float = 0.25):
        self.device = device
        self.sample_rate = int(sample_rate)
        self.reverb_prob = float(reverb_prob)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.reverb_prob > 0 and random.random() < self.reverb_prob:
            return self.apply_reverb(audio)
        transform = random.choice([self.identity, self.add_noise, self.apply_eq, self.volume_change])
        return transform(audio)

    def identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def add_noise(self, x: torch.Tensor, snr: float = 25.0) -> torch.Tensor:
        power = x.pow(2).mean()
        noise = torch.randn_like(x) * (power / 10 ** (float(snr) / 10.0)).sqrt()
        return x + noise

    def apply_eq(self, x: torch.Tensor) -> torch.Tensor:
        k = random.choice([3, 5, 7])
        kernel = torch.ones(1, 1, k, device=self.device) / float(k)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = F.conv1d(x, kernel, padding=k // 2)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., : x.shape[-1]]
        return out.squeeze(1)

    def volume_change(self, x: torch.Tensor) -> torch.Tensor:
        db = random.uniform(-6.0, 6.0)
        return x * 10 ** (db / 20.0)

    def apply_reverb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable reverb approximation.
        Expects x as (B, T). Preserves length.
        """
        if x.dim() != 2:
            if x.dim() == 3 and x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                return x

        _b, t_len = x.shape
        rir_seconds = random.uniform(0.05, 0.25)
        rir_len = max(64, int(self.sample_rate * rir_seconds))

        t = torch.linspace(0, 1, rir_len, device=self.device)
        decay_rate = random.uniform(4.0, 10.0)
        decay = torch.exp(-decay_rate * t)

        rir = torch.randn(rir_len, device=self.device) * decay
        rir[0] = rir[0] + 1.0  # direct path
        rir = rir / (rir.abs().sum() + 1e-6)

        y = F.conv1d(x.unsqueeze(1), rir.view(1, 1, -1), padding=rir_len - 1)
        y = y[..., :t_len].squeeze(1)

        wet = random.uniform(0.2, 0.6)
        return (1 - wet) * x + wet * y


def train_stage2(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    reverb_prob: float = 0.25,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
    finetune_mode: bool = False,
    class_weights: Optional[torch.Tensor] = None,  # optional ID class weights (K,)
    detect_weight: float = 1.0,
    id_weight: float = 2.0,
    freeze_detect_head: bool = False,
    # Legacy args (kept for call-site compatibility; unused in multiclass)
    msg_weight: float = 1.0,
    model_ce_weight: float = 1.0,
    version_ce_weight: float = 1.0,
    pair_ce_weight: float = 1.0,
    unknown_ce_weight: float = 1.0,
    neg_weight: float = 1.0,
    stage2_payload_on_all: bool = False,
    start_epoch: int = 0,
    should_stop: Optional[Callable[[], bool]] = None,
):
    """
    Stage 2: Encoder Training (or Stage 3: Finetuning).

    Notes:
    - Multiclass labels come from `batch["y_class"]` where 0 is clean.
    - Encoder is only updated from watermarked (y_class != 0) samples.
    - In `finetune_mode`, a clean detect loss is added to keep the decoder calibrated.
    """
    _ = (msg_weight, model_ce_weight, version_ce_weight, pair_ce_weight, unknown_ce_weight, stage2_payload_on_all)

    stage_name = "s3_finetune" if finetune_mode else "s2_encoder"
    print(f"Starting {stage_name} for {epochs} epochs")
    w_id = None
    if class_weights is not None:
        w_id = class_weights.to(device)
        print(f"Using class_weights: shape={tuple(w_id.shape)}")

    aug = DifferentiableAugmenter(device, reverb_prob=reverb_prob)
    stft_loss = CachedSTFTLoss().to(device)
    budget_loss = EnergyBudgetLoss(target_db=float(BUDGET_TARGET_DB), limit_type="hard").to(device)

    loss_wrapper = UncertaintyLossWrapper(num_losses=(5 if finetune_mode else 4)).to(device)

    if finetune_mode:
        decoder.train()
        for p in decoder.parameters():
            p.requires_grad = True
        if freeze_detect_head:
            # Best-effort: freeze detect head to prevent the detector from drifting
            # while improving attribution. Works for SlidingWindowDecoder wrappers.
            base = getattr(decoder, "decoder", None)
            if base is not None and hasattr(base, "head_detect"):
                for p in base.head_detect.parameters():
                    p.requires_grad = False
    else:
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False

    encoder.train()

    params = list(encoder.parameters()) + list(loss_wrapper.parameters())
    if finetune_mode:
        params += list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    for epoch in range(start_epoch, start_epoch + int(epochs)):
        if should_stop is not None and bool(should_stop()):
            print(f"[{stage_name}] Stop requested before epoch {epoch+1}; exiting.")
            break
        epoch_loss = 0.0
        epoch_stats: dict[str, float] = {"loss_detect": 0.0, "loss_id": 0.0, "loss_qual": 0.0, "loss_budget": 0.0}
        if finetune_mode:
            epoch_stats["loss_clean_detect"] = 0.0

        batch_count = 0
        stop_requested_this_epoch = False
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None

        for i, batch in enumerate(loader):
            audio = batch["audio"].to(device)  # (B, 1, T)
            y_class = batch["y_class"].to(device)  # (B,)

            pos = (y_class != 0)
            if not pos.any():
                continue

            audio_pos = audio[pos]
            y_pos = y_class[pos]
            y_id_pos = (y_pos - 1).to(dtype=torch.long)

            wm_audio_pos = encoder(audio_pos, y_pos)
            aug_wm = aug(wm_audio_pos.squeeze(1)).unsqueeze(1)

            out_pos = decoder(aug_wm)
            # Prefer localization pooled watermarkness for detect training.
            win_wm_prob = out_pos.get("all_window_wm_prob_loc")
            if win_wm_prob is None:
                raise KeyError("decoder output missing all_window_wm_prob_loc (loc-gated decoder expected)")
            win_id_logits = out_pos["all_window_id_logits"]  # (N, n_win, K)
            n_pos, n_win = win_wm_prob.shape

            # Detect loss on positives (target=1)
            y_det_win_pos = torch.ones((n_pos, n_win), device=device, dtype=torch.float32)
            loss_det_win = F.binary_cross_entropy(win_wm_prob.clamp(1e-6, 1 - 1e-6), y_det_win_pos)
            y_det_clip_pos = torch.ones((n_pos,), device=device, dtype=torch.float32)
            clip_wm_prob = out_pos.get("clip_wm_prob")
            if clip_wm_prob is None:
                raise KeyError("decoder output missing clip_wm_prob")
            loss_det_clip = F.binary_cross_entropy(clip_wm_prob.clamp(1e-6, 1 - 1e-6), y_det_clip_pos)
            loss_detect = loss_det_win + 0.5 * loss_det_clip

            # ID loss on positives
            w_use = None
            if w_id is not None:
                if int(w_id.numel()) == int(win_id_logits.shape[-1]):
                    w_use = w_id
                elif int(w_id.numel()) == int(win_id_logits.shape[-1] + 1):
                    w_use = w_id[1:]
            y_id_win = y_id_pos.view(-1, 1).expand(-1, n_win).reshape(-1)
            loss_id_win = F.cross_entropy(win_id_logits.reshape(n_pos * n_win, -1), y_id_win, weight=w_use)
            loss_id_clip = F.cross_entropy(out_pos["clip_id_logits"], y_id_pos, weight=w_use)
            loss_id = loss_id_win + 0.5 * loss_id_clip

            loss_clean_detect = None
            if finetune_mode and float(neg_weight) > 0:
                neg = (y_class == 0)
                if neg.any():
                    audio_neg = audio[neg]
                    aug_clean = aug(audio_neg.squeeze(1)).unsqueeze(1)
                    out_neg = decoder(aug_clean)
                    win_wm_prob_n = out_neg.get("all_window_wm_prob_loc")
                    if win_wm_prob_n is None:
                        raise KeyError("decoder output missing all_window_wm_prob_loc on clean batch")
                    n_neg, n_win_n = win_wm_prob_n.shape
                    y_det_win_neg = torch.zeros((n_neg, n_win_n), device=device, dtype=torch.float32)
                    loss_det_win_n = F.binary_cross_entropy(win_wm_prob_n.clamp(1e-6, 1 - 1e-6), y_det_win_neg)
                    y_det_clip_neg = torch.zeros((n_neg,), device=device, dtype=torch.float32)
                    clip_wm_prob_n = out_neg.get("clip_wm_prob")
                    if clip_wm_prob_n is None:
                        raise KeyError("decoder output missing clip_wm_prob on clean batch")
                    loss_det_clip_n = F.binary_cross_entropy(clip_wm_prob_n.clamp(1e-6, 1 - 1e-6), y_det_clip_neg)
                    loss_clean_detect = (loss_det_win_n + 0.5 * loss_det_clip_n) * float(neg_weight)
                else:
                    loss_clean_detect = torch.tensor(0.0, device=device)

            loss_qual = stft_loss(audio_pos.squeeze(1), wm_audio_pos.squeeze(1))
            loss_budget = budget_loss(audio_pos, wm_audio_pos)

            if finetune_mode:
                assert loss_clean_detect is not None
                loss = loss_wrapper(
                    [
                        float(detect_weight) * loss_detect,
                        float(id_weight) * loss_id,
                        loss_clean_detect,
                        loss_qual,
                        loss_budget,
                    ]
                )
            else:
                loss = loss_wrapper(
                    [
                        float(detect_weight) * loss_detect,
                        float(id_weight) * loss_id,
                        loss_qual,
                        loss_budget,
                    ]
                )

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            epoch_stats["loss_detect"] += float(loss_detect.item())
            epoch_stats["loss_id"] += float(loss_id.item())
            if finetune_mode and loss_clean_detect is not None:
                epoch_stats["loss_clean_detect"] += float(loss_clean_detect.item())
            epoch_stats["loss_qual"] += float(loss_qual.item())
            epoch_stats["loss_budget"] += float(loss_budget.item())
            batch_count += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                on_step(
                    {
                        "type": "step",
                        "stage": stage_name,
                        "epoch": int(epoch + 1),
                        "batch": int(i),
                        "n_batches": n_batches,
                        "loss": float(loss.item()),
                        "loss_detect": float(loss_detect.item()),
                        "loss_id": float(loss_id.item()),
                        "loss_clean_detect": float(loss_clean_detect.item()) if (finetune_mode and loss_clean_detect is not None) else None,
                        "loss_qual": float(loss_qual.item()),
                        "loss_budget": float(loss_budget.item()),
                        "sigmas": loss_wrapper.log_vars.detach().exp().cpu().numpy().tolist(),
                    }
                )

            if i % int(log_interval) == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

            if should_stop is not None and bool(should_stop()):
                stop_requested_this_epoch = True
                print(f"[{stage_name}] Stop requested during epoch {epoch+1}; finishing epoch early.")
                break

        avg_loss = float(epoch_loss / max(1, batch_count))
        for k in list(epoch_stats.keys()):
            epoch_stats[k] = float(epoch_stats[k] / max(1, batch_count))

        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": stage_name,
                    "epoch": int(epoch + 1),
                    "loss": avg_loss,
                    **epoch_stats,
                    "lr": float(opt.param_groups[0].get("lr", lr)),
                    "log_vars": loss_wrapper.log_vars.detach().cpu().numpy().tolist(),
                    "stop_requested": bool(stop_requested_this_epoch),
                }
            )

        if stop_requested_this_epoch:
            break
