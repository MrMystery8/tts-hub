"""
Stage 1: Decoder Pretraining (Multiclass Attribution)

Trains the decoder to:
1. Distinguish clean vs watermarked.
2. Attribute watermarked audio to one of K model classes.

Uses a FROZEN encoder (acting as a fixed synthetic watermark generator).
Implementation follows multiclass attribution architecture.
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
    class_weights: Optional[torch.Tensor] = None,
    # legacy args (kept for call-site compatibility; unused in multiclass)
    neg_weight: float = 0.0,
    neg_preamble_target: float = 0.5,
    unknown_ce_weight: float = 0.0,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
):
    """
    Stage 1: Decoder Pretraining.
    - Dataset yields CLEAN audio segments.
    - We apply watermark on-the-fly to samples where `y_class != 0` using the frozen encoder.
    - We train decoder using cross-entropy on both window-level and clip-level logits.
    """
    decoder.train()
    encoder.eval() # Encoder is frozen in Stage 1
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    loss_hist = []
    
    print(f"Starting Stage 1: Decoder Pretraining for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_loss_win = 0.0
        epoch_loss_clip = 0.0
        batch_count = 0
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None

        for i, batch in enumerate(loader):
            audio = batch["audio"].to(device)
            y_class = batch["y_class"].to(device)  # (B,)
            has_wm = (y_class != 0).to(dtype=torch.float32)  # (B,)
            
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
            
            # 1) Window-level CE
            window_logits = outputs["all_window_class_logits"]  # (B, n_win, C)
            B, n_win, _ = window_logits.shape
            y_win = y_class.view(-1, 1).expand(-1, n_win).reshape(-1)
            loss_win = F.cross_entropy(
                window_logits.reshape(B * n_win, -1),
                y_win,
                weight=class_weights.to(device) if class_weights is not None else None,
            )

            # 2) Clip-level CE (top-k aggregated logits)
            loss_clip = F.cross_entropy(
                outputs["clip_class_logits"],
                y_class,
                weight=class_weights.to(device) if class_weights is not None else None,
            )

            loss = loss_win + 0.5 * loss_clip
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_loss_win += loss_win.item()
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
                        "loss_win_ce": float(loss_win.item()),
                        "loss_clip_ce": float(loss_clip.item()),
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / max(1, batch_count)
        avg_loss_win = epoch_loss_win / max(1, batch_count)
        avg_loss_clip = epoch_loss_clip / max(1, batch_count)
        loss_hist.append(avg_loss)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} (WinCE: {avg_loss_win:.4f}, ClipCE: {avg_loss_clip:.4f})")

        if on_epoch_end is not None:
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": "s1",
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "loss_win_ce": avg_loss_win,
                    "loss_clip_ce": avg_loss_clip,
                    "lr": opt.param_groups[0].get("lr"),
                }
            )
        
    return loss_hist
