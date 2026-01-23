"""
Stage 1: Detection Training

Trains the decoder to distinguishing watermarked audio from clean audio.
Implementation follows WATERMARK_PROJECT_PLAN.md v16, section 5.2.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional


def train_stage1(
    decoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    log_interval: int = 10
):
    """
    Train detection on BOTH positives and negatives.
    Uses per-window + clip-level loss for stable gradients.
    
    BUG FIX from project plan: Use clip_detect_LOGIT (not prob) with BCEWithLogitsLoss!
    """
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    
    print(f"Starting Stage 1: Detection Training for {epochs} epochs")
    
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        
        for i, batch in enumerate(loader):
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)  # (B,)
            
            # Forward pass
            outputs = decoder(audio)
            
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
            
            total_loss += loss.item()
            batches += 1
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
