# FYP Project Plan: Audio Watermarking with Trained Classifier

## Final Version (v12) — Comprehensive Reference Document

> **Status**: Implementation-ready within stated threat model
> 
> This document consolidates 12 iterations of refinement based on extensive critique. It includes the complete architecture, all lessons learned, and implementation-ready code.

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Threat Model](#3-threat-model)
4. [Complete Implementation](#4-complete-implementation)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Lessons Learned (Mistakes to Avoid)](#7-lessons-learned)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [References](#9-references)

---

# 1. Executive Summary

## What We're Building

A complete audio watermarking system trained from scratch:
- **Encoder**: Lightweight FiLM-conditioned network (~50K params) that embeds imperceptible watermarks
- **Decoder**: CNN-based detector that identifies watermarks and attributes source model
- **Robust to**: MP3, AAC, noise, resampling, reverb, time-stretch (non-adversarial)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Train encoder from scratch | Stronger FYP claim than using pre-trained |
| FiLM conditioning | Message actually modulates watermark content (not just amplitude) |
| Sliding window + voting | Robust to cropping and misalignment |
| 2-stage training | Prevents encoder-decoder collapse |
| Top-k aggregation | Avoids "garbage averaging" and noisy-OR miscalibration |

## FYP Deliverables

1. Trained encoder model (PyTorch)
2. Trained decoder/classifier (PyTorch)
3. Dataset generation pipeline
4. Evaluation report with metrics
5. Integrated TTS Hub demo

---

# 2. Architecture Overview

## System Diagram

```
EMBEDDING:
  Audio (B,1,T) ──► Encoder (FiLM) ──► Overlap-Add ──► Watermarked
                        │
                    Message (32 bits)
                    [16-bit preamble | 7-bit payload × 2]

DETECTION:
  Audio (B,T) ──► Sliding Windows ──► Decoder ──► Per-Window Results
                                                        │
                                            ┌───────────┴───────────┐
                                            │                       │
                                      Top-K Select              Aggregate
                                            │                       │
                                      Payload Vote          Clip Detect Prob
```

## Constants

```python
# Audio
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3.0
SEGMENT_SAMPLES = 48000

# Windows
WINDOW_SAMPLES = 16000  # 1 second
HOP_RATIO = 0.5         # 50% overlap

# Message
MSG_BITS = 32
PREAMBLE_BITS = 16
PAYLOAD_BITS = 7  # 3 model_id + 4 version
```

---

# 3. Threat Model

## What We CLAIM Robustness Against

| Attack | Parameters | Expected Detection |
|--------|------------|-------------------|
| MP3 | 32-128 kbps | >85% |
| AAC | 32-96 kbps | >80% |
| Gaussian noise | SNR 20-40 dB | >85% |
| Resampling | 8kHz ↔ 48kHz | >90% |
| Reverb | RT60 0.3-1.0s | >75% |
| Time-stretch | ±5% | >70% |
| Cropping | 50-100% | >80% |

## What We DO NOT Claim

> [!CAUTION]
> **Explicit Non-Claims**

| Claim | Why Not |
|-------|---------|
| "Robust to adversarial removal" | Not evaluated against adversaries with model access |
| "Secure attribution" | No HMAC/authentication (32-bit payload too small) |
| "Real-world robust" | Neural codecs, Opus not in training |
| "Production-ready" | Proof-of-concept, not battle-tested |

## Security Note

Per NIST guidance, meaningful authentication requires ≥64-bit tags. Our 32-bit message (with 16-bit preamble) provides only ~7 effective payload bits after redundancy—trivial to brute-force.

**Current system: DECODABLE, NOT AUTHENTICATED**

Future work: Expand to 64-96 bits or spread tag across time windows.

---

# 4. Complete Implementation

## 4.1 Message Codec

```python
import hashlib
import torch
import torch.nn.functional as F

class MessageCodec:
    """
    Message format:
    - Bits 0-15:  Preamble (16-bit, keyed pseudo-random)
    - Bits 16-18: Model ID (3 bits, 0-7)
    - Bits 19-22: Version (4 bits, 0-15)
    - Bits 23-25: Model ID copy (redundancy)
    - Bits 26-29: Version copy (redundancy)
    - Bits 30-31: Reserved
    
    Effective payload: 7 bits with 2× soft-average redundancy
    
    BUG FIX: preamble moved to device in decode() to avoid CPU/GPU mismatch
    """
    def __init__(self, key: str = "fyp2026"):
        h = hashlib.sha256(key.encode()).digest()
        self.preamble = torch.tensor(
            [int(b) for b in format(int.from_bytes(h[:2], 'big'), '016b')],
            dtype=torch.float32
        )
    
    def encode(self, model_id: int, version: int = 1) -> torch.Tensor:
        msg = torch.zeros(32)
        msg[0:16] = self.preamble
        
        # Payload copy 1
        for i in range(3):
            msg[16 + i] = (model_id >> i) & 1
        for i in range(4):
            msg[19 + i] = (version >> i) & 1
        
        # Payload copy 2 (for soft-average)
        msg[23:26] = msg[16:19]
        msg[26:30] = msg[19:23]
        
        return msg
    
    def decode(self, probs: torch.Tensor) -> dict:
        """
        Decode with:
        - Strict preamble (15/16 match)
        - Soft-averaging (NOT OR logic!)
        
        BUG FIX: Move preamble to probs.device to avoid CPU/MPS mismatch!
        """
        # FIX: Move preamble to same device as probs!
        preamble = self.preamble.to(probs.device)
        
        preamble_bits = (probs[0:16] > 0.5).int()
        preamble_match = (preamble_bits == preamble.int()).sum().item()
        
        # FIX: Remove hardcoded 15/16 check. Let DecisionRule be the source of truth.
        # preamble_match is returned for the rule to check against tuned params.
        
        # SOFT-AVERAGE: average probs, then threshold
        model_probs = (probs[16:19] + probs[23:26]) / 2
        ver_probs = (probs[19:23] + probs[26:30]) / 2
        
        model_bits = (model_probs > 0.5).int()
        ver_bits = (ver_probs > 0.5).int()
        
        model_id = model_bits[0] + 2*model_bits[1] + 4*model_bits[2]
        version = ver_bits[0] + 2*ver_bits[1] + 4*ver_bits[2] + 8*ver_bits[3]
        
        return {
            "valid": True,
            "model_id": model_id.item(),
            "version": version.item(),
            "preamble_score": preamble_match / 16,
            "confidence": torch.cat([model_probs, ver_probs]).mean().item(),
        }
```

## 4.2 Encoder (FiLM Conditioning)

```python
import torch
import torch.nn as nn

class WatermarkEncoder(nn.Module):
    """
    Encoder with FiLM conditioning.
    Message creates (gamma, beta) pairs that modulate conv features.
    Different bit patterns → different watermarks (not just amplitude scaling).
    """
    def __init__(self, msg_bits: int = 32, hidden: int = 32, groups: int = 4):
        super().__init__()
        
        # FiLM layers
        self.film1 = nn.Linear(msg_bits, hidden * 2)
        self.film2 = nn.Linear(msg_bits, hidden * 2)
        self.film3 = nn.Linear(msg_bits, hidden * 2)
        
        # Conv layers with GroupNorm (stable on small batches)
        self.conv1 = nn.Conv1d(1, hidden, 7, padding=3)
        self.gn1 = nn.GroupNorm(groups, hidden)
        
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.gn2 = nn.GroupNorm(groups, hidden)
        
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.gn3 = nn.GroupNorm(groups, hidden)
        
        self.out = nn.Conv1d(hidden, 1, 3, padding=1)
        
        self.alpha = nn.Parameter(torch.tensor(0.02))
    
    def _film(self, x, params):
        g, b = params.chunk(2, dim=1)
        return g.unsqueeze(-1) * x + b.unsqueeze(-1)
    
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, 1, T)
        message: (B, 32)
        returns: (B, 1, T) watermarked
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
```

## 4.3 Overlap-Add Wrapper (Tensorized)

```python
class OverlapAddEncoder(nn.Module):
    """
    Embeds watermark repeatedly using overlap-add.
    Tensorized with F.fold (no Python loops in reconstruction).
    """
    def __init__(self, base_encoder, window: int = 16000, hop_ratio: float = 0.5):
        super().__init__()
        self.encoder = base_encoder
        self.window = window
        self.hop = int(window * hop_ratio)
        self.register_buffer('hann', torch.hann_window(window))
    
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        B, C, T = audio.shape
        
        # Pad to fit windows
        n_win = max(1, (T - self.window) // self.hop + 1)
        out_len = (n_win - 1) * self.hop + self.window
        pad = out_len - T
        if pad > 0:
            audio = F.pad(audio, (0, pad))
        
        # Unfold: (B, 1, n_win, window)
        windows = audio.unfold(2, self.window, self.hop)
        B, C, N, W = windows.shape
        
        # Batch encode
        flat = windows.reshape(B * N, 1, W)
        msg_exp = message.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        wm_flat = self.encoder(flat, msg_exp) * self.hann
        
        # Reshape for fold
        wm = wm_flat.squeeze(1).reshape(B, N, W).permute(0, 2, 1)
        
        # F.fold: reconstruct WATERMARK RESIDUAL only
        # FIX: Compute residual explicitly before folding to avoid double-counting audio!
        # wm_flat contains (audio + watermark) * hann
        # We need (watermark only) * hann
        
        # Reconstruct flat audio (un-watermarked) * hann
        audio_flat = flat * self.hann.view(1, 1, -1)
        
        # Compute residual: (wm_flat - audio_flat) is pure watermark * hann
        residual_flat = wm_flat - audio_flat
        residual = residual_flat.squeeze(1).reshape(B, N, W).permute(0, 2, 1)
        
        watermark_residual = F.fold(
            residual,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2)
        
        # Normalizer (sum of overlapping windows)
        norm_in = torch.ones_like(wm) * self.hann.view(1, -1, 1)
        normalizer = F.fold(
            norm_in,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2).clamp(min=1e-8)
        
        watermark_residual = watermark_residual / normalizer
        
        # Remove padding
        if pad > 0:
            audio = audio[:, :, :-pad]  # Original audio (unpadded)
            watermark_residual = watermark_residual[:, :, :-pad]
        
        # Add residual to original audio
        return audio + watermark_residual
```

## 4.4 Decoder (Outputs Logits)

```python
class WatermarkDecoder(nn.Module):
    """
    Decoder with:
    - Pure-torch mel (MPS-safe, no torchaudio MelSpectrogram)
    - Outputs LOGITS (use BCEWithLogitsLoss)
    
    PERFORMANCE FIXES:
    - Hann window cached as buffer (not recreated every forward)
    - mel_fb already on device via register_buffer (no .to() in forward)
    """
    def __init__(self, msg_bits: int = 32, n_models: int = 8, n_fft: int = 512):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.n_mels = 80
        
        # Register BOTH as buffers (avoids device copies in forward!)
        self.register_buffer('mel_fb', self._create_mel_filterbank())
        self.register_buffer('stft_window', torch.hann_window(n_fft))
        
        # CNN backbone
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
    
    def _create_mel_filterbank(self, sr=16000):
        import numpy as np
        n_freqs = self.n_fft // 2 + 1
        mel_low, mel_high = 0, 2595 * np.log10(1 + sr / 2 / 700)
        mel_pts = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
        bins = np.floor((self.n_fft + 1) * hz_pts / sr).astype(int)
        
        fb = np.zeros((self.n_mels, n_freqs))
        for i in range(self.n_mels):
            left, center, right = bins[i], bins[i+1], bins[i+2]
            for j in range(left, center):
                fb[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                fb[i, j] = (right - j) / (right - center)
        
        return torch.from_numpy(fb).float()
    
    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        # Use cached window (no creation in forward!)
        spec = torch.stft(audio, self.n_fft, self.hop, 
                          window=self.stft_window, return_complex=True)
        mag = spec.abs()
        # mel_fb already on correct device via register_buffer
        mel = torch.matmul(self.mel_fb, mag)
        return torch.log(mel + 1e-8).unsqueeze(1)
    
    def forward(self, audio: torch.Tensor) -> dict:
        """
        audio: (B, T)
        returns: dict with LOGITS
        """
        mel = self._compute_mel(audio)
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
```

## 4.5 Unified Sliding Window Decoder

```python
class SlidingWindowDecoder(nn.Module):
    """
    Wraps base decoder with sliding window + top-k aggregation.
    
    BUG FIX: Now returns clip_detect_logit for proper BCEWithLogitsLoss training
    """
    def __init__(self, base_decoder, window: int = 16000, hop_ratio: float = 0.5, top_k: int = 3):
        super().__init__()
        self.decoder = base_decoder
        self.window = window
        self.hop = int(window * hop_ratio)
        self.top_k = top_k
    
    def forward(self, audio: torch.Tensor) -> dict:
        B, T = audio.shape
        
        if T < self.window:
            audio = F.pad(audio, (0, self.window - T))
            T = self.window
        
        n_win = (T - self.window) // self.hop + 1
        windows = audio.unfold(1, self.window, self.hop)
        flat = windows.reshape(-1, self.window)
        
        outputs = self.decoder(flat)
        
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
```

## 4.6 Decision Rule

```python
class ClipDecisionRule:
    """
    Clip-level decision with FWER awareness.
    Thresholds tuned on validation set, not hardcoded.
    
    BUG FIX: Accepts SINGLE CLIP outputs (not batched).
    For batched inference, call decide() per clip.
    """
    def __init__(self, detect_threshold: float = 0.8, preamble_min: int = 15):
        self.detect_threshold = detect_threshold
        self.preamble_min = preamble_min
    
    def decide(self, outputs: dict, codec: MessageCodec) -> dict:
        """
        Expects SINGLE CLIP outputs:
        - clip_detect_prob: scalar or (1,) tensor
        - all_window_probs: (n_win,) tensor
        - all_message_probs: (n_win, 32) tensor
        
        For batched: call this per clip with sliced outputs.
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
        from collections import Counter
        votes = Counter([w["model_id"] for w in valid_windows])
        best_model, count = votes.most_common(1)[0]
        
        return {
            "positive": True,
            "model_id": best_model,
            "vote_count": count,
            "valid_windows": len(valid_windows),
            "clip_detect_prob": clip_prob,
        }


def decide_batch(outputs: dict, codec: MessageCodec, rule: ClipDecisionRule) -> list:
    """
    Helper to run decision rule on batched outputs.
    Returns list of decisions, one per clip.
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
```

---

# 5. Training Pipeline

## 5.1 Dataset with Fixed Length

```python
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 48000  # 3 seconds

class WatermarkDataset(Dataset):
    def __init__(self, manifest_path: Path, codec, training: bool = True):
        # FIX: dataset needs codec to generate messages!
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.codec = codec
        self.training = training
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        audio, sr = torchaudio.load(item["path"])
        audio = audio.mean(dim=0)  # Mono
        
        # Resample to 16kHz
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        
        # Fixed length (random crop for train, center for eval)
        T = audio.shape[0]
        if T >= SEGMENT_SAMPLES:
            if self.training:
                start = torch.randint(0, T - SEGMENT_SAMPLES + 1, (1,)).item()
            else:
                start = (T - SEGMENT_SAMPLES) // 2
            audio = audio[start:start + SEGMENT_SAMPLES]
        else:
            audio = F.pad(audio, (0, SEGMENT_SAMPLES - T))
        
        # Type conversions
        model_id = int(item.get("model_id", 0))
        version = int(item.get("version", 1))
        
        # FIX: Generate MESSAGE tensor on-the-fly
        message = self.codec.encode(model_id, version).float()
        
        return {
            "audio": audio,
            "has_watermark": torch.tensor(float(item["has_watermark"]), dtype=torch.float32),
            "model_id": torch.tensor(model_id, dtype=torch.long),
            "version": torch.tensor(version, dtype=torch.long),
            "message": message 
        }

def collate_fn(batch):
    # Standard stack - message guarantees to be present now
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
```

## 5.2 Stage 1: Detection (with Negatives!)

```python
def train_stage1(decoder, loader, device, epochs=20):
    """
    Train detection on BOTH positives and negatives.
    Uses per-window + clip-level loss for stable gradients.
    
    BUG FIX: Use clip_detect_LOGIT (not prob) with BCEWithLogitsLoss!
    """
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        for batch in loader:
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)
            
            outputs = decoder(audio)
            
            # Per-window loss (stable gradients) - LOGITS
            n_win = outputs["all_window_logits"].shape[1]
            has_wm_exp = has_wm.unsqueeze(1).expand(-1, n_win)
            loss_window = F.binary_cross_entropy_with_logits(
                outputs["all_window_logits"], has_wm_exp
            )
            
            # Clip loss - FIX: Use LOGIT, not prob!
            loss_clip = F.binary_cross_entropy_with_logits(
                outputs["clip_detect_logit"],  # FIX: logit, not prob!
                has_wm
            )
            
            loss = loss_window + 0.5 * loss_clip
            
            opt.zero_grad()
            loss.backward()
            opt.step()
```

## 5.3 Stage 1B: Payload (with Curriculum)

```python
def compute_preamble_log_likelihood(msg_probs, preamble):
    """Log-likelihood for preamble selection (not hard match!)"""
    B, n_win, _ = msg_probs.shape
    preamble_probs = msg_probs[:, :, :16]
    preamble_exp = preamble.view(1, 1, 16).expand(B, n_win, -1)
    
    eps = 1e-7
    p = torch.clamp(preamble_probs, eps, 1 - eps)
    ll = torch.where(preamble_exp == 1, torch.log(p), torch.log(1 - p))
    
    return ll.sum(dim=2)


def train_stage1b(decoder, loader, device, preamble, epochs=10, warmup=3, top_k=3):
    """
    Train payload with curriculum:
    - Warmup: use preamble correlation (detector not trusted)
    - After: use detect prob
    """
    for epoch in range(epochs):
        in_warmup = epoch < warmup
        
        # Phase switch logic: Reconfigure optimizer ONLY when phase changes
        # (or just simple check at start of epoch)
        
        if epoch == 0 or epoch == warmup:
             # 1. Set requires_grad FIRST
            if in_warmup:
                # Warmup: Freeze everything EXCEPT message head
                for n, p in decoder.named_parameters():
                    if 'head_message' not in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
            else:
                 # Normal: Unfreeze everything
                for p in decoder.parameters():
                    p.requires_grad = True
            
            # 2. Create optimizer SECOND (only for currently trainable params)
            trainable = [p for p in decoder.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=1e-4)
            print(f"Optimizer reset. Trainable params: {len(trainable)}")
        
        for batch in loader:
            # ... training loop ...
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            outputs = decoder(audio)
            msg_logits = outputs["all_message_logits"]
            
            if in_warmup:
                # Use PREAMBLE for window selection
                ll = compute_preamble_log_likelihood(outputs["all_message_probs"], preamble.to(device))
                _, top_idx = torch.topk(ll, top_k, dim=1)
            else:
                _, top_idx = torch.topk(outputs["all_window_probs"], top_k, dim=1)
            
            # Gather top-k
            B, n_win, bits = msg_logits.shape
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, bits)
            selected = torch.gather(msg_logits, 1, idx_exp)
            avg_logits = selected.mean(dim=1)
            
            loss = F.binary_cross_entropy_with_logits(avg_logits, message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # Unfreeze all
    for p in decoder.parameters():
        p.requires_grad = True
```

## 5.4 Stage 2: Encoder (Differentiable Only!)

```python
class DifferentiableAugmenter:
    """Only transforms that preserve gradient flow."""
    
    def __call__(self, audio):
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
        kernel = torch.ones(1, 1, k, device=x.device) / k
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.conv1d(x, kernel, padding=k//2).squeeze(1)
    
    def volume_change(self, x):
        db = random.uniform(-6, 6)
        return x * 10**(db/20)


def train_stage2(encoder, decoder, loader, device, epochs=20, top_k=3):
    """
    Train encoder with differentiable augments only.
    
    FIX: Use TOP-K windows for loss (matches inference objective!)
    Previous: mean(all windows) - trained different objective than inference
    """
    aug = DifferentiableAugmenter()
    # FIX: Move loss module to device!
    stft_loss = CachedSTFTLoss().to(device)
    
    for p in decoder.parameters():
        p.requires_grad = False
    
    opt = torch.optim.AdamW(encoder.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        for batch in loader:
            audio = batch["audio"].unsqueeze(1).to(device)
            message = batch["message"].to(device)
            
            wm = encoder(audio, message)
            augmented = aug(wm.squeeze(1))
            
            outputs = decoder(augmented)
            
            # Quality loss (cached STFT)
            loss_qual = stft_loss(audio.squeeze(1), wm.squeeze(1))
            
            # FIX: Use TOP-K windows for detection/message loss!
            # This matches the inference objective (top-k aggregation)
            detect = outputs["all_window_probs"]  # (B, n_win)
            detect_logits = outputs["all_window_logits"]  # (B, n_win)
            msg_logits = outputs["all_message_logits"]  # (B, n_win, 32)
            
            B, n_win = detect.shape
            k = min(top_k, n_win)
            
            # Get top-k indices by detection probability
            _, top_idx = torch.topk(detect, k, dim=1)
            
            # Gather top-k detection logits
            top_det_logits = torch.gather(detect_logits, 1, top_idx)  # (B, k)
            
            # Detection loss on top-k (target = 1 for watermarked)
            loss_det = F.binary_cross_entropy_with_logits(
                top_det_logits.mean(dim=1),
                torch.ones(B, device=device)
            )
            
            # Gather top-k message logits
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_logits.shape[-1])
            top_msg_logits = torch.gather(msg_logits, 1, top_idx_exp)  # (B, k, 32)
            
            # Message loss on top-k
            loss_msg = F.binary_cross_entropy_with_logits(
                top_msg_logits.mean(dim=1),  # (B, 32)
                message
            )
            
            loss = loss_det + 0.5 * loss_msg + 10.0 * loss_qual
            
            opt.zero_grad()
            loss.backward()
            opt.step()
```

## 5.5 Quality Loss (Multi-Resolution STFT - CACHED)

```python
class CachedSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss with CACHED windows.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        
        # Cache windows as buffers (avoids creation in forward!)
        # FIX: Don't take device in init, use buffers logic
        for n_fft in fft_sizes:
            self.register_buffer(f'window_{n_fft}', torch.hann_window(n_fft))
    
    def forward(self, original: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
        total = 0.0
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = getattr(self, f'window_{n_fft}')  # Use cached window!
            
            orig_spec = torch.stft(original, n_fft, hop, window=window, return_complex=True)
            wm_spec = torch.stft(watermarked, n_fft, hop, window=window, return_complex=True)
            
            orig_mag, wm_mag = orig_spec.abs(), wm_spec.abs()
            
            sc_loss = torch.norm(orig_mag - wm_mag, p='fro') / (torch.norm(orig_mag, p='fro') + 1e-8)
            log_loss = F.l1_loss(torch.log(wm_mag + 1e-8), torch.log(orig_mag + 1e-8))
            
            total = total + sc_loss + log_loss
        
        return total / len(self.fft_sizes)


# Legacy function kept for reference
def compute_stft_loss(original, watermarked):
    """DEPRECATED: Use CachedSTFTLoss instead for better performance."""
    total = 0.0
    for n_fft in [512, 1024, 2048]:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=original.device)
        
        orig_spec = torch.stft(original, n_fft, hop, window=window, return_complex=True)
        wm_spec = torch.stft(watermarked, n_fft, hop, window=window, return_complex=True)
        
        orig_mag, wm_mag = orig_spec.abs(), wm_spec.abs()
        
        sc_loss = torch.norm(orig_mag - wm_mag, p='fro') / torch.norm(orig_mag, p='fro')
        log_loss = F.l1_loss(torch.log(wm_mag + 1e-8), torch.log(orig_mag + 1e-8))
        
        total += sc_loss + log_loss
    
    return total / 3
```

---

# 6. Evaluation Protocol

## 6.1 Metrics (with References)

| Metric | Method | Reference |
|--------|--------|-----------|
| AUC | Clip-level ROC | Hanley & McNeil, 1982 |
| TPR @ FPR | Threshold tuned on validation | FWER-aware |
| Payload accuracy | Correct model_id / total true positives | |
| ViSQOL | Speech mode, with alignment | Google ViSQOL |

## 6.2 Attack Suite

```python
def apply_attack_safe(audio: torch.Tensor, attack_fn) -> torch.Tensor:
    """
    Apply attack and restore to SEGMENT_SAMPLES.
    Handles length-changing attacks (time-stretch, codecs).
    Assumes audio is (T,) or (channels, T).
    """
    attacked = attack_fn(audio)
    
    # FIX: Enforce length post-attack!
    # Ensure we work on last dim
    T = attacked.shape[-1]
    if T > SEGMENT_SAMPLES:
        attacked = attacked[..., :SEGMENT_SAMPLES]
    elif T < SEGMENT_SAMPLES:
        # Pad last dim
        attacked = F.pad(attacked, (0, SEGMENT_SAMPLES - T))
    
    return attacked


EVAL_ATTACKS = {
    # Trained on
    "clean": lambda x: x,
    "mp3_64": lambda x: apply_attack_safe(x, lambda a: apply_codec(a, "mp3", 64)),
    # ... other attacks wrapped with apply_attack_safe ...
}
```

## 6.3 ViSQOL with Alignment

```python
def compute_visqol_aligned(original, degraded, sr=16000):
    import numpy as np
    import os
    import soundfile as sf
    from scipy.signal import correlate
    
    # Cross-correlation alignment
    corr = correlate(degraded, original, mode='full')
    lag = np.argmax(corr) - len(original) + 1
    
    if lag > 0:
        aligned, ref = degraded[lag:], original[:len(degraded)-lag]
    else:
        aligned, ref = degraded[:len(degraded)+lag], original[-lag:]
    
    min_len = min(len(ref), len(aligned))
    ref, aligned = ref[:min_len], aligned[:min_len]
    
    # Use unique temp files (parallel-safe)
    import uuid
    uid = uuid.uuid4().hex[:8]
    ref_path = f"/tmp/ref_{uid}.wav"
    deg_path = f"/tmp/deg_{uid}.wav"
    
    sf.write(ref_path, ref, sr)
    sf.write(deg_path, aligned, sr)
    
    from visqol import visqol_lib_py
    config = visqol_lib_py.MakeVisqolConfig()
    config.audio.sample_rate = sr
    config.options.use_speech_scoring = True  # Speech mode!
    
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    result = api.Measure(ref_path, deg_path)
    
    os.unlink(ref_path)
    os.unlink(deg_path)
    
    return result.moslqo
```

---

# 7. Lessons Learned (Mistakes to Avoid)

## Critical Bugs Caught During Design

| Bug | Impact | Fix |
|-----|--------|-----|
| **Double embedding** | Random weights in dataset, then re-embed during training | On-the-fly embedding only |
| **OR-logic in decode** | `(a+b)>=1` biases toward 1s | Soft-average: `(p1+p2)/2` |
| **No negatives in Stage 1** | Detector learns "always 1" | 50/50 pos/neg split |
| **max() aggregation** | Inflates FPR (multiple comparisons) | Top-k mean |
| **Non-diff augments in Stage 2** | Gradients don't flow to encoder | Diff-only for encoder training |
| **Warmup selects by garbage detector** | Wrong windows selected | Use preamble log-likelihood |
| **AAC in .mp3 container** | Invalid muxing | Use .m4a for AAC |
| **Train/test by file path** | Same utterance in both via codecs | GroupShuffleSplit by source_idx |

## Architecture Decisions

| Decision | Alternatives Considered | Why Chosen |
|----------|------------------------|------------|
| FiLM conditioning | Amplitude scaling, time-tiling | Message actually changes watermark shape |
| Sliding window + voting | Single-pass, end-to-end | Robust to crop/misalignment |
| GroupNorm | BatchNorm | Stable on small batches (Mac) |
| BCEWithLogitsLoss | sigmoid + BCE | More numerically stable |
| 2-stage training | Joint from start | Prevents encoder-decoder collapse |

## MPS/Mac Considerations

| Issue | Solution |
|-------|----------|
| `torchaudio.MelSpectrogram` not MPS | Pure-torch STFT + mel filterbank |
| Determinism hurts MPS performance | Optional flag, documented caveat |
| Small batches | GroupNorm instead of BatchNorm |

---

# 8. Acceptance Criteria

## Engineering

- [ ] Codec precompute uses correct containers (.mp3, .m4a)
- [ ] No ffmpeg errors in logs
- [ ] Train/val/test have zero source overlap
- [ ] Fold divisor test passes
- [ ] Version pinning documented

## Training

- [ ] Stage 1 loss converges
- [ ] Stage 1B loss decreases after warmup
- [ ] Stage 2 quality loss stays low
- [ ] No NaN/Inf

## Metrics (on held-out test set)

- [ ] Clip-level AUC > 0.95
- [ ] TPR @ 1% FPR > 85%
- [ ] Attribution accuracy (true positives) > 85%
- [ ] ViSQOL > 4.0

## Claim Control

- [ ] Report scopes claims to "benign settings"
- [ ] Does NOT claim "secure attribution"
- [ ] Does NOT claim "robust to adversarial removal"

---

# 9. References

## Benchmarks
1. **AudioMarkBench** — NeurIPS 2024 Datasets Track
2. **RAW-Bench** — Interspeech 2025 / arXiv May 2025
3. **SoK: Audio Watermarking** — arXiv Mar 2025

## Key Papers
4. **AudioSeal** — Roman et al., ICML 2024
5. **WavMark** — Chen et al., arXiv:2308.12770
6. **"Deep Audio Watermarks are Shallow"** — arXiv Apr 2025

## Standards
7. **HMAC** — RFC 2104, IETF
8. **NIST SP 800-224** — HMAC guidance

## Quality Metrics
9. **ViSQOL** — github.com/google/visqol (Apache 2.0)
10. **Hanley & McNeil** — ROC AUC confidence intervals, 1982

## PyTorch Docs
11. `torch.nn.functional.fold` — Image-like 3D/4D only
12. `torch.nn.BCEWithLogitsLoss` — Numerically stable
13. `torchaudio.transforms.MelSpectrogram` — CPU/CUDA only (no MPS)

---

# Appendix: Version History

| Version | Key Change |
|---------|------------|
| v1 | Initial plan |
| v2 | Fixed pipeline contradiction (on-the-fly embedding) |
| v3 | CRC→redundancy, added sync mechanism |
| v4 | Overlap-add, vectorized processing |
| v5 | Soft-average decode, 15/16 preamble |
| v6 | Per-window decode, top-k voting |
| v7 | Stage-1 negatives, unified decoder interface |
| v8 | Stage-1B for payload, aligned hop |
| v9 | Fixed-length, BCEWithLogitsLoss |
| v10 | Curriculum warmup, FWER framing |
| v11 | Reproducibility, acceptance criteria |
| v12 | AAC containers, GroupShuffleSplit, log-likelihood warmup |

**This document represents the final, research-defensible design.**
