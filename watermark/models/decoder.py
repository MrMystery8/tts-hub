"""
Watermark Decoder Module

Contains:
- WatermarkDecoder: Pure-torch mel spectrogram (MPS-safe) two-head classifier
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
    LOC_EPS,
    LOC_TOP_M,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_RATIO,
    TOP_K,
)


class WatermarkDecoder(nn.Module):
    """
    Decoder with pure-torch mel frontend (MPS-safe) and a two-head output:

    - `detect`: binary watermark presence
    - `id`: K-way attribution over watermarked samples (model ID)

    The combined (K+1) distribution is derived as:
      P(clean) = 1 - P(wm)
      P(class=i+1) = P(wm) * P(id=i)

    This avoids early training collapse where the single softmax head predicts `clean`
    for everything (dominant class) and starves the attribution signal.
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
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        self.num_ids = self.num_classes - 1
        
        # Register BOTH as buffers
        self.register_buffer('mel_fb', self._create_mel_filterbank())
        self.register_buffer('stft_window', torch.hann_window(n_fft))
        
        # Backbone: preserve time axis so we can localize watermarkness per-frame.
        # Pool only along frequency to keep temporal resolution.
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        feat_ch = 128
        
        # Two-head outputs:
        # - detect: watermark present vs clean
        # - id: model attribution among watermarked samples only
        self.head_loc = nn.Conv1d(feat_ch, 1, kernel_size=1)
        self.head_detect = nn.Linear(feat_ch, 1)
        self.head_id = nn.Linear(feat_ch, self.num_ids)

    @staticmethod
    def _topm_mean(probs: torch.Tensor, *, m: int) -> torch.Tensor:
        """
        Top-M mean pooling over the last dimension.
        probs: (B, T)
        """
        if probs.dim() != 2:
            raise ValueError(f"expected probs (B,T), got shape={tuple(probs.shape)}")
        k = min(max(1, int(m)), int(probs.shape[-1]))
        top, _ = torch.topk(probs, k=k, dim=-1)
        return top.mean(dim=-1)
    
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
        Predict detect + id logits for audio.
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
            
        # Backbone feature map: (B, C, F, T)
        feat2d = self.backbone(mel)
        # Collapse frequency axis, preserve time: (B, C, T)
        feat = feat2d.mean(dim=2)

        loc_logits = self.head_loc(feat).squeeze(1)  # (B, T)
        loc_probs = torch.sigmoid(loc_logits)  # (B, T)

        wm_prob_loc = self._topm_mean(loc_probs, m=int(LOC_TOP_M))  # (B,)

        # Loc-gated pooling for detect/id heads
        w = loc_probs
        w_norm = w / (w.sum(dim=-1, keepdim=True) + float(LOC_EPS))
        feat_pool = (feat * w_norm.unsqueeze(1)).sum(dim=-1)  # (B, C)

        detect_logit = self.head_detect(feat_pool).squeeze(-1)  # (B,)
        detect_prob = torch.sigmoid(detect_logit)  # (B,)

        id_logits = self.head_id(feat_pool)  # (B, K)
        id_probs = torch.softmax(id_logits, dim=-1)  # (B, K)

        # Use localization pooled watermarkness as the primary score for detection metrics/aggregation.
        p_clean = 1.0 - wm_prob_loc
        p_ids = wm_prob_loc.unsqueeze(-1) * id_probs  # (B, K)
        class_probs = torch.cat([p_clean.unsqueeze(-1), p_ids], dim=-1)  # (B, K+1)
        class_logits = torch.log(class_probs.clamp(min=1e-8))

        return {
            # Primary heads
            "detect_logit": detect_logit,  # (B,)
            "detect_prob": detect_prob,  # (B,)
            "loc_logits": loc_logits,  # (B, T)
            "loc_probs": loc_probs,  # (B, T)
            "wm_prob_loc": wm_prob_loc,  # (B,)
            "id_logits": id_logits,  # (B, K)
            "id_probs": id_probs,  # (B, K)

            # Derived combined distribution (compat)
            "class_logits": class_logits,  # (B, K+1) (derived log-probs)
            "class_probs": class_probs,  # (B, K+1)

            # Alias used throughout the codebase for "watermarkedness"
            "wm_prob": wm_prob_loc,  # (B,)
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
        Sliding-window inference with top-k aggregation by detect probability.

        Returns:
          - `clip_class_logits`: (B, C)
          - `clip_class_probs`: (B, C)
          - `clip_wm_prob`: (B,) where wm_prob = 1 - P(clean)
          - `clip_detect_prob`: alias for `clip_wm_prob` (compat)
          - `clip_id_logits`: (B, K)
          - `clip_id_probs`: (B, K)
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
        detect_logits = outputs["detect_logit"].reshape(B, n_win)  # (B, n_win)
        detect_probs = torch.sigmoid(detect_logits)  # (B, n_win)

        id_logits = outputs["id_logits"].reshape(B, n_win, -1)  # (B, n_win, K)
        loc_logits = outputs["loc_logits"]  # (B*n_win, Tloc)
        t_loc = int(loc_logits.shape[-1])
        all_window_loc_logits = loc_logits.reshape(B, n_win, t_loc)
        wm_prob_loc = outputs["wm_prob_loc"].reshape(B, n_win)  # (B, n_win)

        # Top-k aggregation (by watermark probability)
        k = min(self.top_k, n_win)
        _top_probs, top_idx = torch.topk(wm_prob_loc, k, dim=1)

        # Use mean of top-k detect logits as the clip-level detect logit (for detect-head BCE losses).
        top_det_logits = torch.gather(detect_logits, 1, top_idx)  # (B, k)
        clip_detect_logit = top_det_logits.mean(dim=1)  # (B,)
        clip_detect_prob = torch.sigmoid(clip_detect_logit)  # (B,)

        # Gather top-k ID logits and average
        n_ids = id_logits.shape[-1]
        top_id_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_ids)
        top_id_logits = torch.gather(id_logits, 1, top_id_idx)
        clip_id_logits = top_id_logits.mean(dim=1)  # (B, K)
        clip_id_probs = torch.softmax(clip_id_logits, dim=-1)

        # Clip watermark probability from localization pooling.
        top_wm_probs = torch.gather(wm_prob_loc, 1, top_idx)  # (B, k)
        clip_wm_prob = top_wm_probs.mean(dim=1)  # (B,)

        # Derived combined probs/logits (compat)
        p_clean = 1.0 - clip_wm_prob
        p_ids = clip_wm_prob.unsqueeze(-1) * clip_id_probs
        clip_class_probs = torch.cat([p_clean.unsqueeze(-1), p_ids], dim=-1)
        clip_class_logits = torch.log(clip_class_probs.clamp(min=1e-8))
        
        return {
            "clip_class_logits": clip_class_logits,
            "clip_class_probs": clip_class_probs,
            "clip_id_logits": clip_id_logits,
            "clip_id_probs": clip_id_probs,
            "clip_wm_prob": clip_wm_prob,
            "clip_wm_prob_loc": clip_wm_prob,
            "clip_detect_prob": clip_wm_prob,  # compat alias
            "clip_detect_logit": clip_detect_logit,

            # Window-level outputs (for training)
            "all_window_detect_logits": detect_logits,  # (B, n_win)
            "all_window_detect_probs": detect_probs,  # (B, n_win)
            "all_window_loc_logits": all_window_loc_logits,  # (B, n_win, Tloc)
            "all_window_wm_prob_loc": wm_prob_loc,  # (B, n_win)
            "all_window_id_logits": id_logits,  # (B, n_win, K)
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

        id_probs = outputs.get("clip_id_probs")
        id_logits = outputs.get("clip_id_logits")
        if id_probs is None and id_logits is not None:
            id_probs = torch.softmax(id_logits, dim=-1)
        if id_probs is None:
            # fall back to derived class probs if needed
            class_probs = outputs.get("clip_class_probs")
            if class_probs is None:
                raise ValueError("outputs must include clip_id_probs/clip_id_logits or clip_class_probs")
            if class_probs.dim() == 2:
                class_probs = class_probs[0]
            pred_class = int(torch.argmax(class_probs).item())
            positive = clip_wm_prob >= self.wm_threshold and pred_class != int(CLASS_CLEAN)
            reason = "wm_low" if not positive else "wm_high"
            return {
                "positive": bool(positive),
                "reason": reason,
                "clip_wm_prob": float(clip_wm_prob),
                "pred_class": int(pred_class),
                "pred_model_id": int(pred_class - 1) if pred_class != int(CLASS_CLEAN) else None,
            }

        if id_probs.dim() == 2:
            id_probs = id_probs[0]

        pred_id = int(torch.argmax(id_probs).item())
        pred_class = int(CLASS_CLEAN) if clip_wm_prob < self.wm_threshold else int(pred_id + 1)
        positive = clip_wm_prob >= self.wm_threshold
        reason = "wm_low" if not positive else "wm_high"

        return {
            "positive": bool(positive),
            "reason": reason,
            "clip_wm_prob": float(clip_wm_prob),
            "pred_class": int(pred_class),
            "pred_model_id": int(pred_id) if pred_class != int(CLASS_CLEAN) else None,
        }
