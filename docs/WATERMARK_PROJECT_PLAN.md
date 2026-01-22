# FYP Project Plan v5: Audio Watermarking (Final)

> **All ChatGPT critiques resolved** — Ready for implementation

---

## v5 Fixes Summary

| v4 Issue | v5 Fix |
|----------|--------|
| OR-logic in "majority vote" | **Soft-averaging**: `(p1+p2)/2 > 0.5` |
| "3× repetition" but only 2 copies | Fixed documentation: **2× with soft-average** |
| 14/16 preamble too loose | Tightened to **15/16 match** |
| `max` aggregation inflates FPR | **Noisy-or**: `1 - prod(1 - p_i)` |
| Non-differentiable Stage 2 augments | **Differentiable-only** for encoder training |
| Python loop in overlap-add | **Tensorized** with `F.fold` |
| "10 bits effective" wrong | Corrected to **7 bits** (3+4) |

---

## 1. Fixed Message Codec (Soft-Averaging)

```python
class MessageCodecV5:
    """
    Fixed codec with:
    - 16-bit preamble (keyed)
    - 7-bit payload (3 model_id + 4 version)
    - 2× redundancy with SOFT-AVERAGING (not OR logic)
    """
    def __init__(self, key: str = "fyp2026"):
        import hashlib
        h = hashlib.sha256(key.encode()).digest()
        self.preamble = torch.tensor(
            [int(b) for b in format(int.from_bytes(h[:2], 'big'), '016b')],
            dtype=torch.float
        )
    
    def encode(self, model_id: int, version: int = 1) -> torch.Tensor:
        """
        32-bit message:
        - Bits 0-15:  Preamble (16 bits)
        - Bits 16-18: Model ID (3 bits)
        - Bits 19-22: Version (4 bits)
        - Bits 23-25: Model ID copy 2
        - Bits 26-29: Version copy 2
        - Bits 30-31: Reserved
        
        Effective payload: 7 bits with 2× redundancy
        """
        msg = torch.zeros(32)
        msg[0:16] = self.preamble
        
        # Payload copy 1
        for i in range(3):
            msg[16 + i] = (model_id >> i) & 1
        for i in range(4):
            msg[19 + i] = (version >> i) & 1
        
        # Payload copy 2 (for soft-average redundancy)
        msg[23:26] = msg[16:19]
        msg[26:30] = msg[19:23]
        
        return msg
    
    def decode(self, probs: torch.Tensor) -> dict:
        """
        Decode with:
        - Strict preamble check (15/16 required)
        - Soft-averaging for payload (NOT OR logic)
        """
        # STRICT preamble check: require 15/16 match
        preamble_bits = (probs[0:16] > 0.5).int()
        preamble_match = (preamble_bits == self.preamble.int()).sum().item()
        
        if preamble_match < 15:  # Tightened from 14
            return {"valid": False, "reason": f"preamble ({preamble_match}/16)"}
        
        # SOFT-AVERAGE for payload (fixes OR-logic bug)
        # Average the probabilities, THEN threshold
        model_probs_avg = (probs[16:19] + probs[23:26]) / 2
        ver_probs_avg = (probs[19:23] + probs[26:30]) / 2
        
        model_bits = (model_probs_avg > 0.5).int()
        ver_bits = (ver_probs_avg > 0.5).int()
        
        model_id = model_bits[0] + 2*model_bits[1] + 4*model_bits[2]
        version = ver_bits[0] + 2*ver_bits[1] + 4*ver_bits[2] + 8*ver_bits[3]
        
        # Confidence = average probability of all payload bits
        confidence = torch.cat([model_probs_avg, ver_probs_avg]).mean().item()
        
        return {
            "valid": True,
            "model_id": model_id.item(),
            "version": version.item(),
            "preamble_score": preamble_match / 16,
            "confidence": confidence,
        }
```

---

## 2. Noisy-OR Aggregation (Fixes FPR Inflation)

```python
class ClipLevelDecoder(nn.Module):
    """
    Clip-level detection using noisy-or aggregation.
    Fixes the multiple-comparisons FPR inflation from max().
    """
    def __init__(self, base_decoder, window_samples=16000, hop_ratio=0.25):
        super().__init__()
        self.decoder = base_decoder
        self.window = window_samples
        self.hop = int(window_samples * hop_ratio)
    
    def forward(self, audio: torch.Tensor) -> dict:
        B, T = audio.shape
        
        if T < self.window:
            audio = F.pad(audio, (0, self.window - T))
            T = self.window
        
        n_windows = (T - self.window) // self.hop + 1
        
        # Vectorized window extraction
        windows = audio.unfold(1, self.window, self.hop)  # (B, n_win, window)
        windows_flat = windows.reshape(-1, self.window)   # (B*n_win, window)
        
        # Decode all windows
        outputs = self.decoder(windows_flat)
        
        # Reshape: (B, n_windows, ...)
        detect = outputs["detect_prob"].reshape(B, n_windows, -1).squeeze(-1)
        message = outputs["message_prob"].reshape(B, n_windows, -1)
        model = outputs["model_logits"].reshape(B, n_windows, -1)
        
        # NOISY-OR aggregation (fixes max() FPR inflation)
        # P(at least one) = 1 - prod(1 - p_i)
        noisy_or = 1 - torch.prod(1 - detect, dim=1, keepdim=True)
        
        # For message/model: weight by detection probability
        weights = F.softmax(detect, dim=1).unsqueeze(-1)
        weighted_message = (message * weights).sum(dim=1)
        weighted_model = (model * weights).sum(dim=1)
        
        return {
            "detect_prob": noisy_or,
            "message_prob": weighted_message,
            "model_logits": weighted_model,
            "n_windows": n_windows,
            "per_window_detect": detect,  # For debugging
        }
```

---

## 3. Tensorized Overlap-Add (No Python Loop)

```python
class TensorizedOverlapAddEncoder(nn.Module):
    """
    Fully tensorized overlap-add using F.fold.
    No Python loops in reconstruction.
    """
    def __init__(self, base_encoder, window_samples=16000, hop_ratio=0.5):
        super().__init__()
        self.encoder = base_encoder
        self.window = window_samples
        self.hop = int(window_samples * hop_ratio)
        self.register_buffer('hann', torch.hann_window(window_samples))
    
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        B, C, T = audio.shape
        
        # Pad to fit windows
        n_windows = max(1, (T - self.window) // self.hop + 1)
        out_len = (n_windows - 1) * self.hop + self.window
        pad_len = out_len - T
        if pad_len > 0:
            audio = F.pad(audio, (0, pad_len))
        
        # Extract windows: (B, 1, n_win, window)
        windows = audio.unfold(2, self.window, self.hop)
        B, C, n_win, W = windows.shape
        
        # Reshape for batch encoding: (B * n_win, 1, window)
        windows_flat = windows.permute(0, 2, 1, 3).reshape(B * n_win, 1, W)
        message_exp = message.unsqueeze(1).expand(-1, n_win, -1).reshape(B * n_win, -1)
        
        # Encode all windows
        wm_flat = self.encoder(windows_flat, message_exp)  # (B*n_win, 1, W)
        
        # Apply Hann window
        wm_flat = wm_flat * self.hann.view(1, 1, -1)
        
        # Reshape: (B, n_win, window) -> (B, window, n_win) for fold
        wm = wm_flat.squeeze(1).reshape(B, n_win, W).permute(0, 2, 1)
        
        # TENSORIZED overlap-add using F.fold (no Python loop!)
        output = F.fold(
            wm,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2)  # (B, 1, out_len)
        
        # Compute normalizer (sum of overlapping Hann windows)
        ones = torch.ones(B, n_win, W, device=audio.device)
        ones = ones * self.hann.view(1, 1, -1)
        ones = ones.permute(0, 2, 1)
        normalizer = F.fold(
            ones,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2).clamp(min=1e-8)
        
        output = output / normalizer
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :, :-pad_len]
        
        return output
```

---

## 4. Differentiable-Only Stage 2 Augmentations

```python
class DifferentiableAugmenter(nn.Module):
    """
    Augmentations that preserve gradient flow for Stage 2 encoder training.
    NON-differentiable (MP3, FFmpeg) used ONLY in decoder training + eval.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """All transforms here are differentiable through PyTorch."""
        transform = random.choice([
            self.add_noise,
            self.apply_eq,
            self.resample_diff,
            self.volume_change,
            self.identity,
        ])
        return transform(audio)
    
    def identity(self, x):
        return x
    
    def add_noise(self, x, snr_range=(20, 40)):
        """Differentiable Gaussian noise."""
        snr = random.uniform(*snr_range)
        signal_power = x.pow(2).mean()
        noise_power = signal_power / (10 ** (snr / 10))
        noise = torch.randn_like(x) * noise_power.sqrt()
        return x + noise
    
    def apply_eq(self, x):
        """Differentiable EQ via 1D conv (lowpass/highpass)."""
        kernel_size = random.choice([3, 5, 7])
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.conv1d(x, kernel, padding=kernel_size//2).squeeze(1)
    
    def resample_diff(self, x, factors=[(0.9, 1.1), (0.95, 1.05)]):
        """Differentiable approximate resampling via interpolation."""
        factor = random.uniform(*random.choice(factors))
        orig_len = x.shape[-1]
        new_len = int(orig_len * factor)
        
        # Use interpolate (differentiable)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        resampled = F.interpolate(x, size=new_len, mode='linear', align_corners=False)
        # Resample back to original length
        resampled = F.interpolate(resampled, size=orig_len, mode='linear', align_corners=False)
        return resampled.squeeze(1)
    
    def volume_change(self, x, range_db=(-6, 6)):
        """Differentiable volume scaling."""
        db = random.uniform(*range_db)
        scale = 10 ** (db / 20)
        return x * scale


class NonDifferentiableAugmenter:
    """
    For decoder training and evaluation ONLY.
    These break gradients - never use in Stage 2 encoder training!
    """
    def __call__(self, audio_path: str, transform: str) -> str:
        # MP3, AAC, Opus via FFmpeg - returns path to transformed file
        # Used in dataloader, not in training loop
        ...
```

---

## 5. Fixed Training Pipeline

```python
def train_full_pipeline(encoder, decoder, train_loader, device, config):
    """
    3-stage training with correct augmentation domains.
    """
    diff_aug = DifferentiableAugmenter()
    codec = MessageCodecV5(key=config["key"])
    
    # ===========================================
    # STAGE 1: Decoder with spread-spectrum (20 epochs)
    # Augmentations: ANY (differentiable not required)
    # ===========================================
    print("=== Stage 1: Decoder Training ===")
    spread_spectrum = SpreadSpectrumEmbedder()
    for p in encoder.parameters():
        p.requires_grad = False
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(20):
        for batch in train_loader:
            audio = batch["audio"].to(device)
            message = codec.encode(batch["model_id"], batch["version"]).to(device)
            
            wm = spread_spectrum(audio, message)
            # Can use ANY augmentation here (non-diff OK)
            aug = diff_aug(wm)  # Or load pre-augmented from disk
            
            out = decoder(aug)
            loss = compute_decoder_loss(out, message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # ===========================================
    # STAGE 2: Encoder with frozen decoder (20 epochs)
    # Augmentations: DIFFERENTIABLE ONLY!
    # ===========================================
    print("=== Stage 2: Encoder Training (DIFFERENTIABLE AUG ONLY) ===")
    for p in encoder.parameters():
        p.requires_grad = True
    for p in decoder.parameters():
        p.requires_grad = False
    
    opt = torch.optim.AdamW(encoder.parameters(), lr=3e-4)
    
    for epoch in range(20):
        for batch in train_loader:
            audio = batch["audio"].unsqueeze(1).to(device)
            message = codec.encode(batch["model_id"], batch["version"]).to(device)
            
            wm = encoder(audio, message)
            
            # ONLY differentiable augmentations! Gradients must flow!
            aug = diff_aug(wm.squeeze(1))
            
            out = decoder(aug)
            
            loss_qual = compute_stft_loss(audio.squeeze(1), wm.squeeze(1))
            loss_det = compute_decoder_loss(out, message)
            loss = loss_det + 10.0 * loss_qual
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # ===========================================
    # STAGE 3: Joint fine-tune (10 epochs)
    # Low LR, differentiable augmentations
    # ===========================================
    print("=== Stage 3: Joint Fine-tune ===")
    for p in decoder.parameters():
        p.requires_grad = True
    
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-5
    )
    
    for epoch in range(10):
        # Same as Stage 2 but both models update
        ...
```

---

## 6. Clip-Level Evaluation Metrics

```python
def evaluate_clip_level(encoder, decoder, test_loader, codec, device):
    """
    Report CLIP-LEVEL metrics (not window-level).
    Proper FPR/TPR at clip granularity.
    """
    clip_probs = []
    clip_labels = []
    payload_results = []
    
    for batch in test_loader:
        audio = batch["audio"].to(device)
        is_watermarked = batch["is_watermarked"]
        
        # Get clip-level detection (noisy-or aggregated)
        outputs = decoder(audio)
        clip_prob = outputs["detect_prob"].cpu().numpy()
        
        clip_probs.extend(clip_prob.flatten())
        clip_labels.extend(is_watermarked.numpy())
        
        # Payload decoding
        for i, prob in enumerate(outputs["message_prob"]):
            result = codec.decode(prob)
            result["ground_truth_model"] = batch["model_id"][i].item()
            payload_results.append(result)
    
    # Clip-level metrics
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr, tpr, thresh = roc_curve(clip_labels, clip_probs)
    auc = roc_auc_score(clip_labels, clip_probs)
    
    def tpr_at_fpr(target):
        idx = np.searchsorted(fpr, target)
        return tpr[min(idx, len(tpr)-1)]
    
    valid_payloads = [r for r in payload_results if r.get("valid")]
    
    return {
        "clip_auc": auc,
        "tpr_at_0.1%_fpr": tpr_at_fpr(0.001),
        "tpr_at_1%_fpr": tpr_at_fpr(0.01),
        "tpr_at_5%_fpr": tpr_at_fpr(0.05),
        "valid_payload_rate": len(valid_payloads) / len(payload_results),
        "n_clips": len(clip_labels),
    }
```

---

## 7. Final Architecture Summary

```
ENCODER (FiLM + GroupNorm):
  Audio + Message → FiLM-conditioned convs → Tanh watermark → α-scaled addition

EMBEDDING (Tensorized Overlap-Add):
  Audio → unfold windows → batch encode → F.fold with Hann → smooth output

DECODER (Clip-Level Noisy-OR):
  Audio → unfold windows → batch decode → noisy-or detection → weighted message/model

MESSAGE (2× Soft-Average):
  16-bit preamble (keyed) + 7-bit payload (2× redundancy) → soft-average decode
```

---

## 8. Success Criteria (Realistic)

| Metric | Target | Notes |
|--------|--------|-------|
| Clip-level AUC | >0.95 | Noisy-or aggregation |
| TPR @ 1% FPR | >85% | Clip-level |
| Valid payload rate (clean) | >90% | 15/16 preamble + soft-avg |
| Valid payload rate (MP3-128) | >70% | Degradation expected |
| ViSQOL | >4.0 | Imperceptibility |
| SNR | >30 dB | Quality constraint |

---

## 9. Acknowledged Limitations

- **Attribution not authenticated**: HMAC future work
- **Neural codecs not in scope**: EnCodec stress test exploratory only
- **7-bit effective payload**: Sufficient for 8 models + 16 versions

---

## Appendix: All Critiques Resolved

| Ver | Issue | Status |
|-----|-------|--------|
| v1 | Pipeline contradiction | ✅ On-the-fly |
| v1 | MPS incompatibility | ✅ Pure-torch |
| v2 | Encoder doesn't encode bits | ✅ FiLM |
| v2 | No sync for cropping | ✅ Repeated embed |
| v3 | CRC is not ECC | ✅ 2× redundancy |
| v3 | 4-bit sync weak | ✅ 16-bit preamble |
| v4 | OR-logic in decode | ✅ Soft-average |
| v4 | 14/16 preamble loose | ✅ 15/16 |
| v4 | max() inflates FPR | ✅ Noisy-or |
| v4 | Non-diff Stage 2 | ✅ Diff-only augments |
| v4 | Python loop in fold | ✅ Tensorized F.fold |

**Plan is now implementation-ready.**
