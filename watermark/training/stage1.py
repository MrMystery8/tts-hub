"""
Stage 1: Detection Training

Trains the decoder to distinguishing watermarked audio from clean audio.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.2.

CORRECTION (On-the-fly Embedding):
Dataset yields CLEAN audio. We use a frozen encoder to watermark samples
where `has_watermark=1` dynamically during training.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional


def train_stage1(
    decoder: torch.nn.Module,
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
):
    """
    Stage 1: Detection Training.
    - Dataset yields CLEAN audio.
    - We apply watermark on-the-fly to samples where 'has_watermark'==1 using the FROZEN encoder.
    - We train decoder to distinguish watermarked vs clean.
    """
    decoder.train()
    encoder.eval() # Encoder is frozen in Stage 1
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    loss_hist = []
    
    print(f"Starting Stage 1: Detection Training for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_loss_window = 0.0
        epoch_loss_clip = 0.0
        batch_count = 0
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None

        for i, batch in enumerate(loader):
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)  # (B,)
            message = batch["message"].to(device)
            
            # === On-the-fly Embedding ===
            with torch.no_grad():
                # Fix dimensions: Encoder expects (B, 1, T). Audio is already (B, 1, T).
                watermarked_audio = encoder(audio, message)
                
                # Mix: use watermarked where has_wm=1, else clean
                mask = has_wm.view(-1, 1, 1)
                input_audio = mask * watermarked_audio + (1 - mask) * audio
                input_audio = input_audio.detach()
            
            # Forward pass
            # Decoder adapts to (B, 1, T) or (B, T)
            outputs = decoder(input_audio)
            
            # 1. Per-window loss (stable gradients) - LOGITS
            # (B, n_win)
            window_logits = outputs["all_window_logits"]
            n_win = window_logits.shape[1]
            
            # Expand clip label to all windows
            has_wm_exp = has_wm.unsqueeze(1).expand(-1, n_win)
            
            loss_window = F.binary_cross_entropy_with_logits(
                window_logits, has_wm_exp
            )
            
            # 2. Clip loss - FIX: Use LOGIT, not prob!
            clip_logit = outputs["clip_detect_logit"]  # (B,)
            
            loss_clip = F.binary_cross_entropy_with_logits(
                clip_logit, has_wm
            )
            
            # Combined loss
            loss = loss_window + 0.5 * loss_clip
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_loss_window += loss_window.item()
            epoch_loss_clip += loss_clip.item()
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
                        "loss_window": float(loss_window.item()),
                        "loss_clip": float(loss_clip.item()),
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / max(1, batch_count)
        avg_loss_window = epoch_loss_window / max(1, batch_count)
        avg_loss_clip = epoch_loss_clip / max(1, batch_count)
        loss_hist.append(avg_loss)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": "s1",
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "loss_window": avg_loss_window,
                    "loss_clip": avg_loss_clip,
                    "lr": opt.param_groups[0].get("lr"),
                }
            )
        
    return loss_hist
