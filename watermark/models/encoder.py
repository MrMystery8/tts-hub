"""
Watermark Encoder Module

Contains:
- WatermarkEncoder: Use bounded perturbation: x_wm = x + alpha * tanh(delta)
- OverlapAddEncoder: Wrapper for full-length processing

Matches WATERMARK_PROJECT_PLAN.md v17, sections 4.2-4.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from watermark.config import (
    ENCODER_HIDDEN,
    ENCODER_GROUPS,
    N_CLASSES,
    WINDOW_SAMPLES,
    HOP_RATIO,
)


class WatermarkEncoder(nn.Module):
    """
    Encoder with FiLM conditioning.
    
    Changes:
    - Output is `alpha * tanh(delta)`
    - Alpha is FIXED (or manually scheduled external to model), not learnable per-sample.
      (Here we default to a conservative fixed value, but it can be overridden).
    """
    
    def __init__(
        self,
        *,
        num_classes: int = N_CLASSES,
        embed_dim: int = 32,
        hidden: int = ENCODER_HIDDEN,
        groups: int = ENCODER_GROUPS,
    ):
        super().__init__()
        
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)

        # Class embedding for attribution watermarking (class 0 is clean/no watermark).
        self.class_embed = nn.Embedding(self.num_classes, self.embed_dim)

        # FiLM layers: class embedding → (gamma, beta) modulation parameters
        self.film1 = nn.Linear(self.embed_dim, hidden * 2)
        self.film2 = nn.Linear(self.embed_dim, hidden * 2)
        self.film3 = nn.Linear(self.embed_dim, hidden * 2)
        
        # Conv layers with GroupNorm (stable on small batches, MPS-safe)
        self.conv1 = nn.Conv1d(1, hidden, 7, padding=3)
        self.gn1 = nn.GroupNorm(groups, hidden)
        
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.gn2 = nn.GroupNorm(groups, hidden)
        
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.gn3 = nn.GroupNorm(groups, hidden)
        
        self.out = nn.Conv1d(hidden, 1, 3, padding=1)
        
        # FIXED Alpha (Target ~ -30dB relative to signal roughly)
        # Users can override this attribute during training schedule.
        # We store it as a buffer so it's part of state_dict but not optimized by optimizer.
        self.register_buffer("alpha", torch.tensor(0.01)) 
    
    def _film(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: gamma * x + beta."""
        g, b = params.chunk(2, dim=1)
        return g.unsqueeze(-1) * x + b.unsqueeze(-1)
    
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Embed class-conditional watermark into audio.
        
        Args:
            audio: (B, 1, T) input audio
            message: (B,) class index (0=clean, 1..K=model classes)
        
        Returns:
            (B, 1, T) watermarked audio
        """
        class_id = message
        if class_id.dim() == 2 and class_id.shape[1] == 1:
            class_id = class_id.squeeze(1)
        class_id = class_id.to(dtype=torch.long)
        if class_id.dim() != 1:
            raise ValueError(f"class_id must be (B,) or (B,1), got {tuple(class_id.shape)}")

        emb = self.class_embed(class_id)  # (B, D)

        x = self.conv1(audio)
        x = self._film(self.gn1(x), self.film1(emb))
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self._film(self.gn2(x), self.film2(emb))
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self._film(self.gn3(x), self.film3(emb))
        x = F.relu(x)
        
        # Delta generation
        delta = self.out(x)
        
        # Bounded perturbation
        watermark = torch.tanh(delta)
        
        return audio + self.alpha * watermark


class OverlapAddEncoder(nn.Module):
    """
    Embeds watermark repeatedly using overlap-add for arbitrary-length audio.
    """
    
    def __init__(
        self, 
        base_encoder: WatermarkEncoder, 
        window: int = WINDOW_SAMPLES, 
        hop_ratio: float = HOP_RATIO
    ):
        super().__init__()
        self.encoder = base_encoder
        self.window = window
        self.hop = int(window * hop_ratio)
        self.register_buffer('hann', torch.hann_window(window))
    
    def forward(self, audio: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        Embed watermark with overlap-add.
        """
        B, C, T = audio.shape
        T_orig = T
        
        import math
        n_win = math.ceil((T - self.window) / self.hop) + 1
        if n_win < 1: n_win = 1
        
        out_len = (n_win - 1) * self.hop + self.window
        pad = out_len - T
        
        if pad > 0:
            audio_padded = F.pad(audio, (0, pad))
        else:
            audio_padded = audio
            
        windows = audio_padded.unfold(2, self.window, self.hop)
        B, C, N, W = windows.shape
        
        flat = windows.reshape(B * N, 1, W)
        cls = class_id
        if cls.dim() == 2 and cls.shape[1] == 1:
            cls = cls.squeeze(1)
        if cls.dim() != 1:
            raise ValueError(f"class_id must be (B,) or (B,1), got {tuple(class_id.shape)}")
        cls_exp = cls.unsqueeze(1).expand(-1, N).reshape(B * N)
        
        # Encode
        wm_flat = self.encoder(flat, cls_exp) * self.hann
        
        # Residual
        audio_flat = flat * self.hann.view(1, 1, -1)
        residual_flat = wm_flat - audio_flat
        residual = residual_flat.squeeze(1).reshape(B, N, W).permute(0, 2, 1)
        
        # Fold Residual
        watermark_residual = F.fold(
            residual,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2)
        
        # Normalizer
        norm_in = torch.ones_like(residual) * self.hann.view(1, 1, -1).reshape(1, 1, W).permute(0, 2, 1) # Fix shape match
        # Actually simplest:
        # norm_in = torch.ones(1, 1, W).to(audio.device) * self.hann.view(1, 1, -1)
        # But constructing via residual shape is safer for dimensions.
        norm_in = torch.ones(B, N, W, device=audio.device) * self.hann.view(1, 1, -1)
        norm_in = norm_in.permute(0, 2, 1) # B, W, N -> Wait.
        # F.fold expects (B, C*KERNEL, Blocks) -> (B, C, L)
        # Here we did: residual is (B, W, N)
        # So kernel_size=(1, W).
        
        normalizer = F.fold(
            norm_in,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2).clamp(min=1e-8)
        
        watermark_residual = watermark_residual / normalizer
        
        watermark_residual = watermark_residual[:, :, :T_orig]
        
        return audio + watermark_residual
