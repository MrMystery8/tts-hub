"""
Stage 1: Decoder Pretraining (Adaptive Balancing Variant)

IDENTICAL to stage1.py but uses a LossBalancer for total_loss calculation.
Keeps behavior isolated from the static path.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional

from watermark.config import LOC_CONSISTENCY_WEIGHT
from watermark.utils.loss_balancing import LossBalancer


def train_stage1_adaptive(
    decoder: torch.nn.Module,
    encoder: torch.nn.Module,
    balancer: LossBalancer,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    class_weights: Optional[torch.Tensor] = None,
    loc_consistency_weight: float = float(LOC_CONSISTENCY_WEIGHT),
    # Legacy args compat
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
    start_epoch: int = 0,
):
    """
    Stage 1 Adaptive Training.
    
    Arg `balancer` expects a LossBalancer instance (e.g. UncertaintyBalancer).
    The balancer parameters MUST be included in the optimizer.
    """
    decoder.train()
    balancer.train() # Enable param updates for balancer
    encoder.eval()   # Frozen
    
    # Optimizer must include both decoder AND balancer params
    # We use a single group for simplicity as requested (no separate LR)
    params = list(decoder.parameters()) + list(balancer.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    
    loss_hist = []
    
    print(f"Starting Stage 1: Adaptive Pretraining for {epochs} epochs")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0.0
        epoch_loss_detect = 0.0
        epoch_loss_id = 0.0
        batch_count = 0
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
            
            # --- Detect loss ---
            win_wm_prob = outputs.get("all_window_wm_prob_loc")
            if win_wm_prob is None:
                win_wm_prob = outputs.get("wm_prob_loc", outputs.get("wm_prob"))
                if win_wm_prob is None:
                    raise KeyError("decoder output missing all_window_wm_prob_loc")
                win_wm_prob = win_wm_prob.view(-1, 1)

            B = int(y_detect.shape[0])
            _, n_win = win_wm_prob.shape
            y_det_win = y_detect.view(-1, 1).expand(-1, n_win)
            loss_det_win = F.binary_cross_entropy(win_wm_prob.clamp(1e-6, 1 - 1e-6), y_det_win)

            clip_wm_prob = outputs.get("clip_wm_prob", outputs.get("clip_detect_prob"))
            if clip_wm_prob is None:
                clip_wm_prob = outputs.get("wm_prob_loc", outputs.get("wm_prob"))
            loss_det_clip = F.binary_cross_entropy(clip_wm_prob.clamp(1e-6, 1 - 1e-6), y_detect)
            loss_detect = loss_det_win + 0.5 * loss_det_clip

            # --- Consistency loss ---
            loss_cons = torch.tensor(0.0, device=device)
            # (Keeping consistency logic identical to static, though typically not balanced by UncertaintyBalancer 
            # unless we add a 3rd head. For now we just add it to total raw.)
            if float(loc_consistency_weight) > 0 and "clip_detect_logit" in outputs:
                loss_cons = F.mse_loss(torch.sigmoid(outputs["clip_detect_logit"]), clip_wm_prob.detach())

            # --- ID loss ---
            pos = (y_class != 0)
            loss_id = torch.tensor(0.0, device=device)
            if pos.any():
                win_id_logits = outputs["all_window_id_logits"][pos]
                clip_id_logits = outputs["clip_id_logits"][pos]
                y_id_pos = y_id[pos]
                N = int(win_id_logits.shape[0])
                y_id_win = y_id_pos.view(-1, 1).expand(-1, n_win).reshape(-1)
                w = class_weights.to(device) if class_weights is not None else None
                loss_id_win = F.cross_entropy(win_id_logits.reshape(N * n_win, -1), y_id_win, weight=w)
                loss_id_clip = F.cross_entropy(clip_id_logits, y_id_pos, weight=w)
                loss_id = loss_id_win + 0.5 * loss_id_clip

            # === ADAPTIVE BALANCING ===
            # Combine losses using the balancer
            # Note: We add consistency loss separately (unbalanced/static weight) 
            # because the spec only mentioned balancing Rec/ID.
            weighted_total, bal_info = balancer.combine(loss_detect, loss_id, step=batch_count, epoch=epoch)
            
            loss = weighted_total + float(loc_consistency_weight) * loss_cons
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_loss_detect += loss_detect.item()
            epoch_loss_id += loss_id.item()
            batch_count += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                step_metrics = {
                    "type": "step",
                    "stage": "s1_adaptive",
                    "epoch": epoch + 1,
                    "batch": int(i),
                    "loss": float(loss.item()),
                    "loss_detect": float(loss_detect.item()),
                    "loss_id": float(loss_id.item()),
                    # Add balancer info
                    "w_det": bal_info["w_det"],
                    "w_id": bal_info["w_id"],
                    "s_det": bal_info["s_det"],
                    "s_id": bal_info["s_id"],
                }
                on_step(step_metrics)
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f} | w_det: {bal_info['w_det']:.3f} w_id: {bal_info['w_id']:.3f}")
        
        avg_loss = epoch_loss / max(1, batch_count)
        avg_loss_detect = epoch_loss_detect / max(1, batch_count)
        avg_loss_id = epoch_loss_id / max(1, batch_count)
        loss_hist.append(avg_loss)
        
        # Get final balancer state for epoch log
        _, final_info = balancer.combine(
            torch.tensor(0.0), torch.tensor(0.0), step=0, epoch=epoch
        )
        
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} (Weights: det={final_info['w_det']:.3f}, id={final_info['w_id']:.3f})")

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": "s1_adaptive",
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "loss_detect": avg_loss_detect,
                    "loss_id": avg_loss_id,
                    "lr": opt.param_groups[0].get("lr"),
                    # Add balancer info to epoch logs too
                    "w_det": final_info["w_det"],
                    "w_id": final_info["w_id"],
                    "s_det": final_info["s_det"],
                    "s_id": final_info["s_id"],
                }
            )
        
    return loss_hist
