"""
Stage 2: Encoder Training

Trains the encoder to embed watermarks that obey the decoder,
while minimizing audio degradation.
Implementation follows WATERMARK_PROJECT_PLAN.md v16, section 5.4.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from watermark.training.losses import CachedSTFTLoss


class DifferentiableAugmenter:
    """Only transforms that preserve gradient flow."""
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        transform = random.choice([
            self.identity,
            self.add_noise,
            self.apply_eq,
            self.volume_change,
        ])
        return transform(audio)
    
    def identity(self, x): return x
    
    def add_noise(self, x, snr=25):
        power = x.pow(2).mean()
        noise = torch.randn_like(x) * (power / 10**(snr/10)).sqrt()
        return x + noise
    
    def apply_eq(self, x):
        k = random.choice([3, 5, 7])
        kernel = torch.ones(1, 1, k, device=self.device) / k
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Apply conv and same pad
        out = F.conv1d(x, kernel, padding=k//2)
        if out.shape[-1] != x.shape[-1]:
             out = out[..., :x.shape[-1]]
        return out.squeeze(1)
    
    def volume_change(self, x):
        db = random.uniform(-6, 6)
        return x * 10**(db/20)


def train_stage2(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    top_k: int = 3,
    lr: float = 3e-4,
    log_interval: int = 10
):
    """
    Train encoder with differentiable augments only.
    
    FIX from project plan: Use TOP-K windows for loss (matches inference objective!)
    """
    print(f"Starting Stage 2: Encoder Training for {epochs} epochs")
    
    aug = DifferentiableAugmenter(device)
    
    # Quality loss
    stft_loss = CachedSTFTLoss().to(device)
    
    # Freeze decoder
    for p in decoder.parameters():
        p.requires_grad = False
    
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        
        for i, batch in enumerate(loader):
            # Train only on samples meant to be watermarked according to manifest?
            # Or force all samples to be watermarked for encoder training?
            # Usually we take any audio and force watermark it for training.
            
            audio = batch["audio"].unsqueeze(1).to(device)  # (B, 1, T)
            message = batch["message"].to(device)  # (B, 32)
            
            # Embed
            wm = encoder(audio, message)
            
            # Augment (gradient-safe)
            augmented = aug(wm.squeeze(1))
            
            # Decode
            outputs = decoder(augmented)
            
            # 1. Quality Loss
            loss_qual = stft_loss(audio.squeeze(1), wm.squeeze(1))
            
            # 2. Detection & Message Loss (Top-K)
            detect = outputs["all_window_probs"]  # (B, n_win)
            detect_logits = outputs["all_window_logits"]
            msg_logits = outputs["all_message_logits"]
            
            B, n_win = detect.shape
            k = min(top_k, n_win)
            
            # Top-k by detection probability
            _, top_idx = torch.topk(detect, k, dim=1)
            
            # Gather top-k detect logits
            top_det_logits = torch.gather(detect_logits, 1, top_idx)
            
            # Detection loss (Target = 1)
            loss_det = F.binary_cross_entropy_with_logits(
                top_det_logits.mean(dim=1),
                torch.ones(B, device=device)
            )
            
            # Gather top-k message logits
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_logits.shape[-1])
            top_msg_logits = torch.gather(msg_logits, 1, top_idx_exp)
            
            # Message loss
            loss_msg = F.binary_cross_entropy_with_logits(
                top_msg_logits.mean(dim=1),
                message
            )
            
            # Combined Loss
            loss = loss_det + 0.5 * loss_msg + 10.0 * loss_qual
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            batches += 1
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f} (Qual: {loss_qual.item():.4f}, Det: {loss_det.item():.4f})")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
