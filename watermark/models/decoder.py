"""
Watermark Decoder Module

Contains:
- WatermarkDecoder: Pure-torch mel spectrogram (MPS-safe) watermark detector
- SlidingWindowDecoder: Top-k aggregation wrapper
- ClipDecisionRule: Clip-level decision with majority voting
- decide_batch: Helper for batched inference

Implementation follows WATERMARK_PROJECT_PLAN.md v16, sections 4.4-4.6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import TYPE_CHECKING

from watermark.config import (
    MSG_BITS,
    N_MODELS,
    DECODER_N_FFT,
    DECODER_N_MELS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_RATIO,
    TOP_K,
)

if TYPE_CHECKING:
    from watermark.models.codec import MessageCodec


class WatermarkDecoder(nn.Module):
    """
    Decoder with:
    - Pure-torch mel (MPS-safe, no torchaudio MelSpectrogram)
    - Outputs LOGITS (use BCEWithLogitsLoss)
    
    PERFORMANCE FIXES from project plan:
    - Hann window cached as buffer (not recreated every forward)
    - mel_fb already on device via register_buffer (no .to() in forward)
    """
    
    def __init__(
        self, 
        msg_bits: int = MSG_BITS, 
        n_models: int = N_MODELS, 
        n_fft: int = DECODER_N_FFT,
        sample_rate: int = SAMPLE_RATE
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.n_mels = DECODER_N_MELS
        self.sample_rate = sample_rate
        
        # Register BOTH as buffers (avoids device copies in forward!)
        self.register_buffer('mel_fb', self._create_mel_filterbank())
        self.register_buffer('stft_window', torch.hann_window(n_fft))
        
        # CNN backbone with GroupNorm (stable on small batches)
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
        
        # Output LOGITS (not probabilities!)
        self.head_detect = nn.Linear(feat_dim, 1)
        self.head_message = nn.Linear(feat_dim, msg_bits)
        self.head_model = nn.Linear(feat_dim, n_models + 1)  # +1 for unknown
    
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
        # Use cached window (no creation in forward!)
        spec = torch.stft(
            audio, 
            self.n_fft, 
            self.hop, 
            window=self.stft_window, 
            return_complex=True
        )
        mag = spec.abs()
        # mel_fb already on correct device via register_buffer
        mel = torch.matmul(self.mel_fb, mag)
        return torch.log(mel + 1e-8).unsqueeze(1)
    
    def forward(self, audio: torch.Tensor) -> dict:
        """
        Detect watermark in audio.
        
        Args:
            audio: (B, T) input audio
        
        Returns:
            dict with LOGITS and probs:
                - detect_logit: (B, 1)
                - message_logits: (B, 32)
                - model_logits: (B, n_models+1)
                - detect_prob: (B, 1) sigmoid of detect_logit
                - message_probs: (B, 32) sigmoid of message_logits
        """
        mel = self._compute_mel(audio)
        
        # MPS FIX: AdaptiveAvgPool2d((4, 4)) requires input dims to be divisible by 4.
        # Backbone does 2 MaxPool2d(2), so factor is 4. 
        # Feature map time dim must be divisible by 4.
        # So original mel time dim must be divisible by 16.
        # Pad time dimension (dim 3)
        T = mel.shape[3]
        pad_amt = (16 - (T % 16)) % 16
        if pad_amt > 0:
            mel = F.pad(mel, (0, pad_amt))
            
        features = self.backbone(mel).view(mel.size(0), -1)
        
        detect_logit = self.head_detect(features)
        message_logits = self.head_message(features)
        
        return {
            "detect_logit": detect_logit,
            "message_logits": message_logits,
            "model_logits": self.head_model(features),
            
            # Probs for inference (computed once, not twice!)
            "detect_prob": torch.sigmoid(detect_logit),
            "message_probs": torch.sigmoid(message_logits),
        }


class SlidingWindowDecoder(nn.Module):
    """
    Wraps base decoder with sliding window + top-k aggregation.
    
    BUG FIX from project plan: Returns clip_detect_logit for proper
    BCEWithLogitsLoss training (not just prob).
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
        Detect watermark with sliding window aggregation.
        
        Args:
            audio: (B, T) input audio
        
        Returns:
            dict with clip-level and per-window outputs
        """
        B, T = audio.shape
        
        # Handle short audio
        if T < self.window:
            audio = F.pad(audio, (0, self.window - T))
            T = self.window
        
        n_win = (T - self.window) // self.hop + 1
        windows = audio.unfold(1, self.window, self.hop)
        flat = windows.reshape(-1, self.window)
        
        outputs = self.decoder(flat)
        
        # Reshape outputs to (B, n_win, ...)
        detect = outputs["detect_prob"].reshape(B, n_win)
        message = outputs["message_probs"].reshape(B, n_win, -1)
        model = outputs["model_logits"].reshape(B, n_win, -1)
        
        detect_logits = outputs["detect_logit"].reshape(B, n_win)
        message_logits = outputs["message_logits"].reshape(B, n_win, -1)
        
        # Top-k aggregation (on PROBS for ranking)
        k = min(self.top_k, n_win)
        top_vals, top_idx = torch.topk(detect, k, dim=1)
        clip_detect_prob = top_vals.mean(dim=1)
        
        # BUG FIX: Also aggregate LOGITS for training!
        # Gather top-k logits and average (for BCEWithLogitsLoss)
        top_logits = torch.gather(detect_logits, 1, top_idx)
        clip_detect_logit = top_logits.mean(dim=1)
        
        return {
            # Clip-level (BOTH prob and logit!)
            "clip_detect_prob": clip_detect_prob,
            "clip_detect_logit": clip_detect_logit,  # FIX: For BCEWithLogitsLoss!
            
            # Per-window (for decision rule)
            "all_window_probs": detect,
            "all_message_probs": message,
            "all_model_logits": model,
            
            # Logits (for training)
            "all_window_logits": detect_logits,
            "all_message_logits": message_logits,
            
            "n_windows": n_win,
            "top_k_idx": top_idx,
        }


class ClipDecisionRule:
    """
    Clip-level decision with FWER awareness.
    Thresholds tuned on validation set, not hardcoded.
    
    BUG FIX from project plan: Accepts SINGLE CLIP outputs (not batched).
    For batched inference, call decide() per clip.
    """
    
    def __init__(self, detect_threshold: float = 0.8, preamble_min: int = 15):
        """
        Initialize decision rule.
        
        Args:
            detect_threshold: Minimum detection probability for positive
            preamble_min: Minimum matching preamble bits (out of 16)
        """
        self.detect_threshold = detect_threshold
        self.preamble_min = preamble_min
    
    def decide(self, outputs: dict, codec: 'MessageCodec') -> dict:
        """
        Make clip-level decision.
        
        Expects SINGLE CLIP outputs:
        - clip_detect_prob: scalar or (1,) tensor
        - all_window_probs: (n_win,) tensor
        - all_message_probs: (n_win, 32) tensor
        
        For batched: call this per clip with sliced outputs.
        
        Returns:
            dict with decision result
        """
        # Handle both scalar and tensor
        clip_prob = outputs["clip_detect_prob"]
        if hasattr(clip_prob, 'item'):
            clip_prob = clip_prob.item()
        
        if clip_prob < self.detect_threshold:
            return {"positive": False, "reason": "clip_detect_low"}
        
        # FIX: Explicitly handle per-window indexing
        window_probs = outputs["all_window_probs"]  # (n_win,)
        message_probs = outputs["all_message_probs"]  # (n_win, 32)
        
        n_win = window_probs.shape[0]
        
        valid_windows = []
        for w_idx in range(n_win):
            w_detect = window_probs[w_idx].item()
            w_msg = message_probs[w_idx]  # (32,)
            
            if w_detect < self.detect_threshold:
                continue
            
            result = codec.decode(w_msg)
            # FIX: DecisionRule is the SINGLE source of truth for thresholds
            # We check result["preamble_score"] against self.preamble_min (tuned)
            if result["preamble_score"] * 16 < self.preamble_min:
                continue
            
            valid_windows.append({
                "idx": w_idx,
                "model_id": result["model_id"],
                "confidence": result["confidence"],
            })
        
        if len(valid_windows) == 0:
            return {"positive": False, "reason": "no_valid_windows"}
        
        # Majority vote
        votes = Counter([w["model_id"] for w in valid_windows])
        best_model, count = votes.most_common(1)[0]
        
        return {
            "positive": True,
            "model_id": best_model,
            "vote_count": count,
            "valid_windows": len(valid_windows),
            "clip_detect_prob": clip_prob,
        }


def decide_batch(outputs: dict, codec: 'MessageCodec', rule: ClipDecisionRule) -> list:
    """
    Run decision rule on batched outputs.
    
    Args:
        outputs: Batched outputs from SlidingWindowDecoder
        codec: MessageCodec instance
        rule: ClipDecisionRule instance
    
    Returns:
        List of decisions, one per clip
    """
    B = outputs["clip_detect_prob"].shape[0]
    decisions = []
    
    for b in range(B):
        single = {
            "clip_detect_prob": outputs["clip_detect_prob"][b],
            "all_window_probs": outputs["all_window_probs"][b],
            "all_message_probs": outputs["all_message_probs"][b],
        }
        decisions.append(rule.decide(single, codec))
    
    return decisions
