"""
Watermark Decoder Module

Contains:
- WatermarkDecoder: Pure-torch mel spectrogram (MPS-safe) watermark detector
- SlidingWindowDecoder: Top-k aggregation wrapper
- ClipDecisionRule: Clip-level decision with majority voting
- decide_batch: Helper for batched inference

Implementation follows WATERMARK_PROJECT_PLAN.md v17, sections 4.4-4.6
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
    N_VERSIONS,
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
        n_versions: int = N_VERSIONS,
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
        self.head_version = nn.Linear(feat_dim, n_versions + 1)  # +1 for unknown
        self.head_pair = nn.Linear(feat_dim, (n_models * n_versions))
    
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
            audio: (B, T) or (B, 1, T) input audio
        
        Returns:
            dict with LOGITS and probs:
                - detect_logit: (B, 1)
                - message_logits: (B, 32)
                - model_logits: (B, n_models+1)
                - detect_prob: (B, 1) sigmoid of detect_logit
                - message_probs: (B, 32) sigmoid of message_logits
        """
        # Adapter for Canonical (B, 1, T) -> (B, T) for STFT
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
            
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
            "version_logits": self.head_version(features),
            "pair_logits": self.head_pair(features),
            
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
            audio: (B, T) or (B, 1, T) input audio
        
        Returns:
            dict with clip-level and per-window outputs
        """
        # Adapter for Canonical (B, 1, T) -> (B, T)
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
            
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
        version = outputs["version_logits"].reshape(B, n_win, -1)
        pair = outputs.get("pair_logits")
        if pair is not None:
            pair = pair.reshape(B, n_win, -1)
        
        detect_logits = outputs["detect_logit"].reshape(B, n_win)
        message_logits = outputs["message_logits"].reshape(B, n_win, -1)
        
        # Top-k aggregation (on PROBS for ranking)
        k = min(self.top_k, n_win)
        top_vals, top_idx = torch.topk(detect, k, dim=1)
        clip_detect_prob = top_vals.mean(dim=1)
        
        # Gather top-k logits and average (for BCEWithLogitsLoss)
        top_logits = torch.gather(detect_logits, 1, top_idx)
        clip_detect_logit = top_logits.mean(dim=1)
        
        # Gather top-k message logits
        # top_idx is (B, K), message_logits is (B, n_win, 32)
        bits = message_logits.shape[-1]
        top_msg_idx = top_idx.unsqueeze(-1).expand(-1, -1, bits)
        top_msg_logits = torch.gather(message_logits, 1, top_msg_idx)
        avg_message_logits = top_msg_logits.mean(dim=1)
        
        # Probs for inference
        avg_message_probs = torch.sigmoid(avg_message_logits)

        # Gather top-k model/version logits for attribution
        n_model_classes = model.shape[-1]
        top_model_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_model_classes)
        top_model_logits = torch.gather(model, 1, top_model_idx)
        avg_model_logits = top_model_logits.mean(dim=1)

        n_version_classes = version.shape[-1]
        top_ver_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_version_classes)
        top_version_logits = torch.gather(version, 1, top_ver_idx)
        avg_version_logits = top_version_logits.mean(dim=1)

        avg_pair_logits = None
        if pair is not None:
            n_pair_classes = pair.shape[-1]
            top_pair_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_pair_classes)
            top_pair_logits = torch.gather(pair, 1, top_pair_idx)
            avg_pair_logits = top_pair_logits.mean(dim=1)
        
        out = {
            # Clip-level (BOTH prob and logit!)
            "clip_detect_prob": clip_detect_prob,
            "clip_detect_logit": clip_detect_logit,
            "avg_message_logits": avg_message_logits,
            "avg_message_probs": avg_message_probs,
            "avg_model_logits": avg_model_logits,
            "avg_version_logits": avg_version_logits,
            
            # Per-window (for decision rule)
            "all_window_probs": detect,
            "all_message_probs": message,
            "all_model_logits": model,
            "all_version_logits": version,
            
            # Logits (for training)
            "all_window_logits": detect_logits,
            "all_message_logits": message_logits,
            
            "n_windows": n_win,
            "top_k_idx": top_idx,
        }
        if pair is not None:
            out["avg_pair_logits"] = avg_pair_logits
            out["all_pair_logits"] = pair
        return out


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
        model_logits = outputs.get("all_model_logits")  # (n_win, n_models+1) optional
        version_logits = outputs.get("all_version_logits")  # (n_win, n_versions+1) optional
        pair_logits = outputs.get("all_pair_logits")  # (n_win, n_pairs) optional
        
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
            
            # Prefer classification heads for attribution if present; fallback to bit-decode.
            if pair_logits is not None:
                w_pair_logits = pair_logits[w_idx]
                pred_pair = int(torch.argmax(w_pair_logits).item())
                pair_conf = float(F.softmax(w_pair_logits, dim=0)[pred_pair].item())
                pred_model = int(pred_pair % N_MODELS)
                pred_version = int(pred_pair // N_MODELS)
                model_conf = pair_conf
                version_conf = pair_conf
            elif model_logits is not None:
                w_model_logits = model_logits[w_idx]
                pred_model = int(torch.argmax(w_model_logits).item())
                model_conf = float(F.softmax(w_model_logits, dim=0)[pred_model].item())
            else:
                pred_model = result["model_id"]
                model_conf = result["confidence"]

            pred_version = None
            version_conf = None
            if pair_logits is not None:
                # Already set above.
                pass
            elif version_logits is not None:
                w_ver_logits = version_logits[w_idx]
                pred_version = int(torch.argmax(w_ver_logits).item())
                version_conf = float(F.softmax(w_ver_logits, dim=0)[pred_version].item())

            valid_windows.append({
                "idx": w_idx,
                "model_id": pred_model,
                "model_conf": model_conf,
                "version": pred_version,
                "version_conf": version_conf,
            })
        
        if len(valid_windows) == 0:
            return {"positive": False, "reason": "no_valid_windows"}
        
        # Majority vote
        votes = Counter([w["model_id"] for w in valid_windows])
        best_model, count = votes.most_common(1)[0]

        # Optional version vote
        versions = [w["version"] for w in valid_windows if w["version"] is not None]
        best_version = None
        if versions:
            best_version = Counter(versions).most_common(1)[0][0]
        
        return {
            "positive": True,
            "model_id": best_model,
            "vote_count": count,
            "valid_windows": len(valid_windows),
            "clip_detect_prob": clip_prob,
            "version": best_version,
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
        if "all_model_logits" in outputs:
            single["all_model_logits"] = outputs["all_model_logits"][b]
        if "all_version_logits" in outputs:
            single["all_version_logits"] = outputs["all_version_logits"][b]
        if "all_pair_logits" in outputs:
            single["all_pair_logits"] = outputs["all_pair_logits"][b]
        decisions.append(rule.decide(single, codec))
    
    return decisions
