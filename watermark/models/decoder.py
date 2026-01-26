"""
Watermark Decoder Module

Contains:
- WatermarkDecoder: Pure-torch mel spectrogram (MPS-safe) multiclass classifier
- SlidingWindowDecoder: Sliding-window + top-k aggregation wrapper
- AttributionDecisionRule: Simple clip-level decision helper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from watermark.config import (
    CLASS_CLEAN,
    DECODER_N_FFT,
    DECODER_N_MELS,
    N_CLASSES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_RATIO,
    TOP_K,
)


class WatermarkDecoder(nn.Module):
    """
    Decoder with pure-torch mel frontend (MPS-safe) and a single multiclass head.

    Classes:
      - 0: clean (no watermark)
      - 1..K: attribution classes (e.g., model IDs)
    """
    
    def __init__(
        self, 
        n_fft: int = DECODER_N_FFT,
        sample_rate: int = SAMPLE_RATE,
        num_classes: int = N_CLASSES,
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.n_mels = DECODER_N_MELS
        self.sample_rate = sample_rate
        self.num_classes = int(num_classes)
        
        # Register BOTH as buffers
        self.register_buffer('mel_fb', self._create_mel_filterbank())
        self.register_buffer('stft_window', torch.hann_window(n_fft))
        
        # Backbone (Standard CNN)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        feat_dim = 128 * 4 * 4
        
        # Single multiclass attribution head
        self.head_class = nn.Linear(feat_dim, self.num_classes)
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix (pure numpy, MPS-safe)."""
        n_freqs = self.n_fft // 2 + 1
        mel_low = 0
        mel_high = 2595 * np.log10(1 + self.sample_rate / 2 / 700)
        mel_pts = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((self.n_fft + 1) * hz_pts / self.sample_rate).astype(int)
        
        fb = np.zeros((self.n_mels, n_freqs))
        for i in range(self.n_mels):
            left, center, right = bins[i], bins[i+1], bins[i+2]
            for j in range(left, center):
                if center != left:
                    fb[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    fb[i, j] = (right - j) / (right - center)
        
        return torch.from_numpy(fb).float()
    
    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram using pure torch (MPS-safe)."""
        spec = torch.stft(
            audio, 
            self.n_fft, 
            self.hop, 
            window=self.stft_window, 
            return_complex=True
        )
        mag = spec.abs()
        mel = torch.matmul(self.mel_fb, mag)
        return torch.log(mel + 1e-8).unsqueeze(1)
    
    def forward(self, audio: torch.Tensor) -> dict:
        """
        Predict class logits for audio.
        """
        # Adapter for Canonical (B, 1, T) -> (B, T)
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
            
        mel = self._compute_mel(audio)
        
        # Pad time dimension to be divisible by 16 (for pooling)
        T = mel.shape[3]
        pad_amt = (16 - (T % 16)) % 16
        if pad_amt > 0:
            mel = F.pad(mel, (0, pad_amt))
            
        features = self.backbone(mel).view(mel.size(0), -1)

        class_logits = self.head_class(features)
        class_probs = torch.softmax(class_logits, dim=-1)
        wm_prob = 1.0 - class_probs[:, int(CLASS_CLEAN)]

        return {
            "class_logits": class_logits,      # (B, C)
            "class_probs": class_probs,        # (B, C)
            "wm_prob": wm_prob,                # (B,)
        }


class SlidingWindowDecoder(nn.Module):
    """
    Wraps base decoder with sliding window + top-k aggregation.
    """
    
    def __init__(
        self, 
        base_decoder: WatermarkDecoder, 
        window: int = WINDOW_SAMPLES, 
        hop_ratio: float = HOP_RATIO, 
        top_k: int = TOP_K
    ):
        super().__init__()
        self.decoder = base_decoder
        self.window = window
        self.hop = int(window * hop_ratio)
        self.top_k = top_k
    
    def forward(self, audio: torch.Tensor) -> dict:
        """
        Sliding-window inference with top-k aggregation by watermark probability.

        Returns:
          - `clip_class_logits`: (B, C)
          - `clip_class_probs`: (B, C)
          - `clip_wm_prob`: (B,) where wm_prob = 1 - P(clean)
          - `clip_detect_prob`: alias for `clip_wm_prob` (compat)
          - per-window logits/probs and indices
        """
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
            
        B, T = audio.shape
        
        if T < self.window:
            audio = F.pad(audio, (0, self.window - T))
            T = self.window
        
        n_win = (T - self.window) // self.hop + 1
        windows = audio.unfold(1, self.window, self.hop)
        flat = windows.reshape(-1, self.window)
        
        outputs = self.decoder(flat)
        
        # Reshape to (B, n_win, ...)
        class_logits = outputs["class_logits"].reshape(B, n_win, -1)  # (B, n_win, C)
        class_probs = torch.softmax(class_logits, dim=-1)
        wm_probs = 1.0 - class_probs[:, :, int(CLASS_CLEAN)]  # (B, n_win)

        # Top-k aggregation (by watermark probability)
        k = min(self.top_k, n_win)
        top_vals, top_idx = torch.topk(wm_probs, k, dim=1)
        clip_wm_prob = top_vals.mean(dim=1)

        # Gather top-k class logits and average
        n_classes = class_logits.shape[-1]
        top_cls_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
        top_cls_logits = torch.gather(class_logits, 1, top_cls_idx)
        clip_class_logits = top_cls_logits.mean(dim=1)
        clip_class_probs = torch.softmax(clip_class_logits, dim=-1)
        
        return {
            "clip_class_logits": clip_class_logits,
            "clip_class_probs": clip_class_probs,
            "clip_wm_prob": clip_wm_prob,
            "clip_detect_prob": clip_wm_prob,  # compat alias
            "all_window_class_logits": class_logits,
            "all_window_class_probs": class_probs,
            "all_window_wm_probs": wm_probs,
            "n_windows": n_win,
            "top_k_idx": top_idx,
        }


class AttributionDecisionRule:
    """
    Simple decision rule for multiclass attribution.

    - If `clip_wm_prob < wm_threshold`, predict clean (class 0).
    - Else predict the argmax class (1..K).
    """
    
    def __init__(self, wm_threshold: float = 0.8):
        self.wm_threshold = float(wm_threshold)
    
    def decide(self, outputs: dict) -> dict:
        """
        Make clip-level decision.
        """
        clip_wm_prob = outputs.get("clip_wm_prob", outputs.get("clip_detect_prob"))
        if hasattr(clip_wm_prob, "item"):
            clip_wm_prob = float(clip_wm_prob.item())
        else:
            clip_wm_prob = float(clip_wm_prob)

        probs = outputs.get("clip_class_probs")
        logits = outputs.get("clip_class_logits")
        if probs is None and logits is not None:
            probs = torch.softmax(logits, dim=-1)
        if probs is None:
            raise ValueError("outputs must include clip_class_probs or clip_class_logits")

        if probs.dim() == 2:
            # (B, C) -> single clip expected; take first item
            probs = probs[0]

        pred_class = int(torch.argmax(probs).item())
        positive = clip_wm_prob >= self.wm_threshold and pred_class != int(CLASS_CLEAN)
        reason = "wm_low" if not positive else "wm_high"

        return {
            "positive": bool(positive),
            "reason": reason,
            "clip_wm_prob": float(clip_wm_prob),
            "pred_class": int(pred_class),
            "pred_model_id": int(pred_class - 1) if pred_class != int(CLASS_CLEAN) else None,
        }
