"""
Stage 1: Decoder Pretraining (Multiclass Attribution)

Trains the decoder to:
1. Distinguish clean vs watermarked.
2. Attribute watermarked audio to one of K model classes.

Uses a FROZEN encoder (acting as a fixed synthetic watermark generator).
Implementation uses a two-head decoder (detect + ID) plus a localization head for detection.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional

from watermark.config import LOC_CONSISTENCY_WEIGHT


def train_stage1(
    decoder: torch.nn.Module,
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    class_weights: Optional[torch.Tensor] = None,  # optional ID class weights (K,)
    detect_weight: float = 1.0,
    id_weight: float = 2.0,
    loc_consistency_weight: float = float(LOC_CONSISTENCY_WEIGHT),
    # legacy args (kept for call-site compatibility; unused in multiclass)
    neg_weight: float = 0.0,
    neg_preamble_target: float = 0.5,
    unknown_ce_weight: float = 0.0,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
    start_epoch: int = 0,
    should_stop: Optional[Callable[[], bool]] = None,
):
    """
    Stage 1: Decoder Pretraining.
    - Dataset yields CLEAN audio segments.
    - We apply watermark on-the-fly to samples where `y_class != 0` using the frozen encoder.
    - We train decoder using:
        - detect loss (binary watermark presence) via localization pooling
        - ID loss (K-way model ID) on watermarked samples only
    """
    decoder.train()
    encoder.eval() # Encoder is frozen in Stage 1
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    loss_hist = []
    
    print(f"Starting Stage 1: Decoder Pretraining for {epochs} epochs")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        if should_stop is not None and bool(should_stop()):
            print(f"[Stage1] Stop requested before epoch {epoch+1}; exiting Stage 1.")
            break
        epoch_loss = 0.0
        epoch_loss_detect = 0.0
        epoch_loss_id = 0.0
        batch_count = 0
        stop_requested_this_epoch = False
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None

        for i, batch in enumerate(loader):
            audio = batch["audio"].to(device)
            y_class = batch["y_class"].to(device)  # (B,)
            y_detect = (y_class != 0).to(dtype=torch.float32)  # (B,)
            y_id = (y_class - 1).to(dtype=torch.long)  # (B,) (masked for clean)
            
            # === On-the-fly Embedding (positives only) ===
            with torch.no_grad():
                input_audio = audio
                pos = (y_class != 0)
                if pos.any():
                    wm = encoder(audio[pos], y_class[pos])
                    input_audio = input_audio.clone()
                    input_audio[pos] = wm
                input_audio = input_audio.detach()
            
            # Forward pass
            outputs = decoder(input_audio)
            
            # --- Detect loss (all samples) ---
            win_wm_prob = outputs.get("all_window_wm_prob_loc")
            if win_wm_prob is None:
                # Fallback if caller provided a non-sliding decoder.
                win_wm_prob = outputs.get("wm_prob_loc", outputs.get("wm_prob"))
                if win_wm_prob is None:
                    raise KeyError("decoder output missing all_window_wm_prob_loc/wm_prob_loc/wm_prob")
                win_wm_prob = win_wm_prob.view(-1, 1)

            B = int(y_detect.shape[0])
            if win_wm_prob.dim() != 2 or int(win_wm_prob.shape[0]) != B:
                raise ValueError(f"expected win_wm_prob (B,n_win), got {tuple(win_wm_prob.shape)}")
            _, n_win = win_wm_prob.shape

            y_det_win = y_detect.view(-1, 1).expand(-1, n_win)
            loss_det_win = F.binary_cross_entropy(win_wm_prob.clamp(1e-6, 1 - 1e-6), y_det_win)

            clip_wm_prob = outputs.get("clip_wm_prob", outputs.get("clip_detect_prob"))
            if clip_wm_prob is None:
                clip_wm_prob = outputs.get("wm_prob_loc", outputs.get("wm_prob"))
            loss_det_clip = F.binary_cross_entropy(clip_wm_prob.clamp(1e-6, 1 - 1e-6), y_detect)
            loss_detect = loss_det_win + 0.5 * loss_det_clip

            loss_cons = torch.tensor(0.0, device=device)
            if float(loc_consistency_weight) > 0 and "clip_detect_logit" in outputs:
                loss_cons = F.mse_loss(torch.sigmoid(outputs["clip_detect_logit"]), clip_wm_prob.detach())

            # --- ID loss (positives only) ---
            pos = (y_class != 0)
            loss_id = torch.tensor(0.0, device=device)
            if pos.any():
                win_id_logits = outputs["all_window_id_logits"][pos]  # (N, n_win, K)
                clip_id_logits = outputs["clip_id_logits"][pos]  # (N, K)
                y_id_pos = y_id[pos]  # (N,)
                N = int(win_id_logits.shape[0])
                y_id_win = y_id_pos.view(-1, 1).expand(-1, n_win).reshape(-1)
                w = class_weights.to(device) if class_weights is not None else None
                loss_id_win = F.cross_entropy(win_id_logits.reshape(N * n_win, -1), y_id_win, weight=w)
                loss_id_clip = F.cross_entropy(clip_id_logits, y_id_pos, weight=w)
                loss_id = loss_id_win + 0.5 * loss_id_clip

            loss = (
                float(detect_weight) * loss_detect
                + float(id_weight) * loss_id
                + float(loc_consistency_weight) * loss_cons
            )
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_loss_detect += loss_detect.item()
            epoch_loss_id += loss_id.item()
            batch_count += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                on_step(
                    {
                        "type": "step",
                        "stage": "s1",
                        "epoch": epoch + 1,
                        "batch": int(i),
                        "n_batches": n_batches,
                        "loss": float(loss.item()),
                        "loss_detect": float(loss_detect.item()),
                        "loss_id": float(loss_id.item()),
                        "loss_cons": float(loss_cons.item()) if loss_cons is not None else None,
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")

            if should_stop is not None and bool(should_stop()):
                stop_requested_this_epoch = True
                print(f"[Stage1] Stop requested during epoch {epoch+1}; finishing epoch early.")
                break
        
        avg_loss = epoch_loss / max(1, batch_count)
        avg_loss_detect = epoch_loss_detect / max(1, batch_count)
        avg_loss_id = epoch_loss_id / max(1, batch_count)
        loss_hist.append(avg_loss)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} (Detect: {avg_loss_detect:.4f}, ID: {avg_loss_id:.4f})")

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": "s1",
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "loss_detect": avg_loss_detect,
                    "loss_id": avg_loss_id,
                    "lr": opt.param_groups[0].get("lr"),
                    "stop_requested": bool(stop_requested_this_epoch),
                }
            )

        if stop_requested_this_epoch:
            break
        
    return loss_hist
