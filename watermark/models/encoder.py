"""
Watermark Encoder Module

Contains:
- WatermarkEncoder: FiLM-conditioned CNN that embeds watermarks
- OverlapAddEncoder: Tensorized wrapper for full-length audio

Implementation follows WATERMARK_PROJECT_PLAN.md v17, sections 4.2-4.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from watermark.config import (
    MSG_BITS,
    ENCODER_HIDDEN,
    ENCODER_GROUPS,
    WINDOW_SAMPLES,
    HOP_RATIO,
)


class WatermarkEncoder(nn.Module):
    """
    Encoder with FiLM conditioning.
    Message creates (gamma, beta) pairs that modulate conv features.
    Different bit patterns → different watermarks (not just amplitude scaling).
    
    Architecture: ~50K params with GroupNorm for stable small-batch training.
    """
    
    def __init__(self, msg_bits: int = MSG_BITS, hidden: int = ENCODER_HIDDEN, groups: int = ENCODER_GROUPS):
        super().__init__()
        
        # FiLM layers: message → (gamma, beta) modulation parameters
        self.film1 = nn.Linear(msg_bits, hidden * 2)
        self.film2 = nn.Linear(msg_bits, hidden * 2)
        self.film3 = nn.Linear(msg_bits, hidden * 2)
        
        # Conv layers with GroupNorm (stable on small batches, MPS-safe)
        self.conv1 = nn.Conv1d(1, hidden, 7, padding=3)
        self.gn1 = nn.GroupNorm(groups, hidden)
        
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.gn2 = nn.GroupNorm(groups, hidden)
        
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.gn3 = nn.GroupNorm(groups, hidden)
        
        self.out = nn.Conv1d(hidden, 1, 3, padding=1)
        
        # Learnable watermark strength, clamped to [0.01, 0.1]
        self.alpha = nn.Parameter(torch.tensor(0.02))
    
    def _film(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: gamma * x + beta."""
        g, b = params.chunk(2, dim=1)
        return g.unsqueeze(-1) * x + b.unsqueeze(-1)
    
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Embed watermark into audio.
        
        Args:
            audio: (B, 1, T) input audio
            message: (B, 32) message tensor
        
        Returns:
            (B, 1, T) watermarked audio
        """
        x = self.conv1(audio)
        x = self._film(self.gn1(x), self.film1(message))
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self._film(self.gn2(x), self.film2(message))
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self._film(self.gn3(x), self.film3(message))
        x = F.relu(x)
        
        watermark = torch.tanh(self.out(x))
        alpha = torch.clamp(self.alpha, 0.01, 0.1)
        
        return audio + alpha * watermark


class OverlapAddEncoder(nn.Module):
    """
    Embeds watermark repeatedly using overlap-add.
    Tensorized with F.fold (no Python loops in reconstruction).
    
    Key bug fix from project plan: Computes pure watermark residual
    before folding to avoid double-counting audio.
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
    
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Embed watermark with overlap-add for arbitrary-length audio.
        
        Args:
            audio: (B, 1, T) input audio (any length)
            message: (B, 32) message tensor
        
        Returns:
            (B, 1, T) watermarked audio (same length as input)
        """
        B, C, T = audio.shape
        T_orig = T
        
        # Calculate windows needed to cover FULL length
        # We need n_win such that (n_win-1)*hop + window >= T
        # (n_win-1)*hop >= T - window
        # n_win-1 >= (T - window)/hop
        # n_win >= (T - window)/hop + 1
        # Use ceil to cover end
        import math
        n_win = math.ceil((T - self.window) / self.hop) + 1
        
        # FIX: Ensure at least 1 window if T < window
        if n_win < 1: n_win = 1
        
        out_len = (n_win - 1) * self.hop + self.window
        
        # Pad to fit windows AND ensure minimum size of window
        pad = out_len - T
        
        # Always pad to fit windows
        if pad > 0:
            audio_padded = F.pad(audio, (0, pad))
        else:
            audio_padded = audio
            
        # Unfold from PADDED audio
        windows = audio_padded.unfold(2, self.window, self.hop)
        B, C, N, W = windows.shape
        
        # Batch encode all windows
        flat = windows.reshape(B * N, 1, W)
        msg_exp = message.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        wm_flat = self.encoder(flat, msg_exp) * self.hann
        
        # FIX: Compute residual explicitly BEFORE folding to avoid double-counting audio!
        audio_flat = flat * self.hann.view(1, 1, -1)
        residual_flat = wm_flat - audio_flat
        residual = residual_flat.squeeze(1).reshape(B, N, W).permute(0, 2, 1)
        
        # F.fold: reconstruct WATERMARK RESIDUAL only
        watermark_residual = F.fold(
            residual,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2)
        
        # Normalizer (sum of overlapping Hann windows)
        wm = residual_flat.squeeze(1).reshape(B, N, W).permute(0, 2, 1)
        norm_in = torch.ones_like(wm) * self.hann.view(1, -1, 1)
        normalizer = F.fold(
            norm_in,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2).clamp(min=1e-8)
        
        watermark_residual = watermark_residual / normalizer
        
        # Crop back to original T
        watermark_residual = watermark_residual[:, :, :T_orig]
        
        # Add residual to original audio (not padded)
        return audio + watermark_residual
