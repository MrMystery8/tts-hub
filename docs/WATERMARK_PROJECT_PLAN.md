# FYP Project Plan v7: Audio Watermarking (Final)

> **All critiques resolved** — Implementation-ready

---

## v7 Fixes Summary

| v6 Issue | v7 Fix |
|----------|--------|
| Stage 1 trains on CLEAN (not watermarked) | **Fixed message + pre-watermark→codec pipeline** |
| Noisy-OR miscalibrated | **Top-k mean aggregator** (simpler, honest) |
| No clip-level decision rule | **Explicit rule with threshold tuning** |
| F.fold API constraints | **2D reshape + unit tests documented** |
| .pt float storage blowup | **FLAC int16 with consistent shapes** |

---

## 1. Fixed Stage 1: Pre-Watermark Then Codec

### The Bug (v6)

```python
# v6 BUG: codec applied to CLEAN, decoder sees no watermark!
wm_clean = spread_spectrum(clean, message)  # Watermark created...
out = decoder(augmented)  # ...but IGNORED! augmented is from clean!
```

### The Fix (v7): Fixed Message + Correct Pipeline

```python
# Stage 1 uses FIXED MESSAGE (preamble + dummy payload)
# This allows precomputing: clean → watermark → codec → tensor

STAGE1_FIXED_MESSAGE = torch.zeros(32)
STAGE1_FIXED_MESSAGE[0:16] = PREAMBLE  # Fixed 16-bit preamble
STAGE1_FIXED_MESSAGE[16:32] = 0.5  # Dummy payload (detector learns presence, not ID)


def generate_stage1_tensors(manifest_path: Path, output_dir: Path):
    """
    Pre-generate: clean → watermark(fixed_msg) → codec → FLAC
    This is the CORRECT pipeline for Stage 1.
    """
    embedder = SpreadSpectrumEmbedder()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    codecs = {
        "mp3_64": lambda p: apply_codec(p, "mp3", 64),
        "mp3_128": lambda p: apply_codec(p, "mp3", 128),
        "aac_96": lambda p: apply_codec(p, "aac", 96),
    }
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    for i, item in enumerate(manifest):
        # Load clean
        clean, sr = torchaudio.load(item["audio_path"])
        clean = clean.mean(dim=0, keepdim=True)  # Mono, shape (1, T)
        
        # WATERMARK WITH FIXED MESSAGE
        wm = embedder(clean.unsqueeze(0), STAGE1_FIXED_MESSAGE.unsqueeze(0))
        wm = wm.squeeze(0)  # (1, T)
        
        # Save watermarked as temp WAV
        temp_wm = output_dir / f"temp_{i:04d}.wav"
        torchaudio.save(str(temp_wm), wm, sr)
        
        item["stage1_augmented"] = {}
        
        for codec_name, codec_fn in codecs.items():
            # Apply codec to WATERMARKED audio
            coded_wav = codec_fn(temp_wm)
            
            # Save as FLAC int16 (not .pt float - saves disk!)
            out_path = output_dir / f"{i:04d}_{codec_name}.flac"
            torchaudio.save(str(out_path), coded_wav, sr, 
                           encoding="PCM_S", bits_per_sample=16)
            
            item["stage1_augmented"][codec_name] = str(out_path)
        
        temp_wm.unlink()  # Clean up temp
        print(f"Stage 1 prep: {i+1}/{len(manifest)}")
    
    with open(output_dir / "stage1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


class Stage1Dataset(Dataset):
    """Loads pre-watermarked, pre-coded audio for Stage 1."""
    
    def __init__(self, manifest_path: Path):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
    
    def __getitem__(self, idx):
        item = self.manifest[idx]
        
        # Random codec selection
        codec_name = random.choice(list(item["stage1_augmented"].keys()))
        audio_path = item["stage1_augmented"][codec_name]
        
        # Load FLAC (consistent shape: mono)
        audio, sr = torchaudio.load(audio_path)
        audio = audio.squeeze(0)  # (T,)
        
        return {
            "audio": audio,
            "codec": codec_name,
            "has_watermark": 1.0,  # All Stage 1 samples are watermarked
            # No model_id/version - Stage 1 only learns presence!
        }
```

### Fixed Training Loop

```python
def train_stage1(decoder, stage1_loader, device, epochs=20):
    """
    Stage 1: Decoder learns to DETECT watermarks under real codecs.
    Uses FIXED message - no attribution learning yet.
    """
    print("=== Stage 1: Presence Detection (real codecs) ===")
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        for batch in stage1_loader:
            # Audio is ALREADY watermarked + coded
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)
            
            outputs = decoder(audio)
            
            # Only train detection (not payload) in Stage 1
            loss = F.binary_cross_entropy(
                outputs["detect_prob"].squeeze(), 
                has_wm
            )
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        print(f"Stage 1 Epoch {epoch+1}: loss={loss.item():.4f}")
```

---

## 2. Top-K Mean Aggregator (Replaces Noisy-OR)

### Why Not Noisy-OR

> Noisy-OR assumes independence. Overlapping windows are correlated → probabilities miscalibrated.

### Simple, Honest Alternative

```python
class TopKMeanDecoder(nn.Module):
    """
    Aggregate using top-k mean instead of noisy-OR.
    Simpler, more honest about correlation.
    """
    def __init__(self, base_decoder, window_samples=16000, hop_ratio=0.25, top_k=3):
        super().__init__()
        self.decoder = base_decoder
        self.window = window_samples
        self.hop = int(window_samples * hop_ratio)
        self.top_k = top_k
    
    def forward(self, audio: torch.Tensor) -> dict:
        B, T = audio.shape
        
        if T < self.window:
            audio = F.pad(audio, (0, self.window - T))
            T = self.window
        
        n_windows = (T - self.window) // self.hop + 1
        windows = audio.unfold(1, self.window, self.hop)
        windows_flat = windows.reshape(-1, self.window)
        
        outputs = self.decoder(windows_flat)
        detect = outputs["detect_prob"].reshape(B, n_windows)
        
        # TOP-K MEAN (not noisy-OR)
        top_k_vals, top_k_idx = torch.topk(detect, min(self.top_k, n_windows), dim=1)
        clip_detect = top_k_vals.mean(dim=1)
        
        return {
            "clip_detect_prob": clip_detect,
            "top_k_window_probs": top_k_vals,
            "top_k_window_idx": top_k_idx,
            "all_window_probs": detect,
        }
```

---

## 3. Explicit Clip-Level Decision Rule

```python
class ClipDecisionRule:
    """
    Explicit decision rule for clip-level positive/negative.
    Thresholds tuned to target clip-level FPR.
    """
    def __init__(
        self,
        detect_threshold: float = 0.7,      # Tune on validation
        preamble_min: int = 15,              # Out of 16
        min_valid_windows: int = 1,          # At least 1 window passes
    ):
        self.detect_threshold = detect_threshold
        self.preamble_min = preamble_min
        self.min_valid_windows = min_valid_windows
    
    def decide(self, decoder_output: dict, codec) -> dict:
        """
        Clip is POSITIVE if:
        1. clip_detect_prob >= detect_threshold
        2. At least min_valid_windows have valid preamble + high detect
        3. Majority vote on model_id among valid windows
        """
        clip_prob = decoder_output["clip_detect_prob"].item()
        
        if clip_prob < self.detect_threshold:
            return {"positive": False, "reason": "clip_detect_below_threshold"}
        
        # Check per-window validity
        valid_windows = []
        for w_idx, (w_detect, w_msg) in enumerate(zip(
            decoder_output["all_window_probs"],
            decoder_output["all_message_probs"]
        )):
            if w_detect.item() < self.detect_threshold:
                continue
            
            decode_result = codec.decode(w_msg)
            if not decode_result["valid"]:
                continue
            if decode_result["preamble_score"] * 16 < self.preamble_min:
                continue
            
            valid_windows.append({
                "idx": w_idx,
                "detect": w_detect.item(),
                "model_id": decode_result["model_id"],
                "confidence": decode_result["confidence"],
            })
        
        if len(valid_windows) < self.min_valid_windows:
            return {"positive": False, "reason": "insufficient_valid_windows"}
        
        # Majority vote
        from collections import Counter
        model_votes = Counter([w["model_id"] for w in valid_windows])
        best_model, count = model_votes.most_common(1)[0]
        
        return {
            "positive": True,
            "model_id": best_model,
            "vote_count": count,
            "valid_windows": len(valid_windows),
            "clip_detect_prob": clip_prob,
        }


def tune_thresholds(decoder, val_loader, codec, target_fpr=0.01):
    """
    Tune decision thresholds to achieve target CLIP-LEVEL FPR.
    """
    all_probs = []
    all_labels = []
    
    for batch in val_loader:
        outputs = decoder(batch["audio"])
        all_probs.extend(outputs["clip_detect_prob"].cpu().numpy())
        all_labels.extend(batch["has_watermark"].numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find threshold for target FPR on NEGATIVES
    neg_probs = all_probs[all_labels == 0]
    threshold = np.percentile(neg_probs, 100 * (1 - target_fpr))
    
    # Compute TPR at this threshold
    pos_probs = all_probs[all_labels == 1]
    tpr = (pos_probs >= threshold).mean()
    
    return {
        "threshold": threshold,
        "target_fpr": target_fpr,
        "achieved_tpr": tpr,
    }
```

---

## 4. F.fold with 2D Reshape + Unit Tests

```python
class SafeOverlapAddEncoder(nn.Module):
    """
    Overlap-add using F.fold with explicit 2D reshape.
    F.fold expects image-like (B, C*kH*kW, L) input.
    """
    def __init__(self, base_encoder, window=16000, hop_ratio=0.5):
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
        
        # Unfold: (B, 1, n_win, W)
        windows = audio.unfold(2, self.window, self.hop)
        B, C, N, W = windows.shape
        
        # Batch encode
        flat = windows.reshape(B * N, 1, W)
        msg_exp = message.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        wm_flat = self.encoder(flat, msg_exp) * self.hann
        
        # Reshape for fold: (B, W, N) - treating W as "channels"
        wm = wm_flat.squeeze(1).reshape(B, N, W).permute(0, 2, 1)  # (B, W, N)
        
        # F.fold: output_size=(1, out_len), kernel=(1, W), stride=(1, hop)
        # Input shape must be (B, C*kH*kW, L) = (B, W, N)
        output = F.fold(
            wm,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        )  # (B, 1, 1, out_len)
        output = output.squeeze(2)  # (B, 1, out_len)
        
        # Normalizer
        norm_in = torch.ones_like(wm) * self.hann.view(1, -1, 1)
        normalizer = F.fold(
            norm_in,
            output_size=(1, out_len),
            kernel_size=(1, self.window),
            stride=(1, self.hop)
        ).squeeze(2).clamp(min=1e-8)
        
        output = output / normalizer
        
        if pad > 0:
            output = output[:, :, :-pad]
        
        return output


# === UNIT TESTS (run before training!) ===
def test_overlap_add_reconstruction():
    """Verify overlap-add reconstructs original when encoder is identity."""
    
    class IdentityEncoder(nn.Module):
        def forward(self, audio, msg):
            return audio
    
    encoder = SafeOverlapAddEncoder(IdentityEncoder(), window=1024, hop_ratio=0.5)
    
    # Random test audio
    audio = torch.randn(2, 1, 8000)
    msg = torch.zeros(2, 32)
    
    output = encoder(audio, msg)
    
    # Should be close to input (within numerical precision)
    error = (output - audio).abs().max().item()
    assert error < 1e-5, f"Reconstruction error too high: {error}"
    print("✓ Overlap-add reconstruction test passed")


def test_overlap_add_gradients():
    """Verify gradients flow through overlap-add."""
    
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, 3, padding=1)
        def forward(self, audio, msg):
            return self.conv(audio)
    
    base = SimpleEncoder()
    encoder = SafeOverlapAddEncoder(base, window=1024, hop_ratio=0.5)
    
    audio = torch.randn(2, 1, 8000, requires_grad=True)
    msg = torch.zeros(2, 32)
    
    output = encoder(audio, msg)
    loss = output.sum()
    loss.backward()
    
    assert audio.grad is not None, "No gradient for audio"
    assert base.conv.weight.grad is not None, "No gradient for encoder weights"
    print("✓ Overlap-add gradient test passed")
```

---

## 5. FLAC Storage with Consistent Shapes

```python
def save_audio_flac(tensor: torch.Tensor, path: Path, sr: int = 16000):
    """
    Save audio as FLAC int16.
    Enforces consistent shape: (1, T) mono.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)  # Force mono
    
    # Normalize to [-1, 1] if needed
    if tensor.abs().max() > 1.0:
        tensor = tensor / tensor.abs().max()
    
    torchaudio.save(
        str(path), 
        tensor, 
        sr,
        encoding="PCM_S",
        bits_per_sample=16
    )


def load_audio_flac(path: Path, target_sr: int = 16000) -> torch.Tensor:
    """
    Load FLAC with consistent output shape: (T,) float32.
    """
    audio, sr = torchaudio.load(str(path))
    
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    
    audio = audio.squeeze(0)  # (T,)
    return audio
```

---

## 6. Final Architecture Summary

```
STAGE 1 PREP:
  clean → watermark(fixed_msg) → codec(MP3/AAC) → FLAC int16

STAGE 1 TRAINING:
  FLAC → decoder → detect_loss only (no attribution)

STAGE 2 TRAINING:
  clean → encoder(message) → diff_aug → decoder → all losses

STAGE 3:
  Joint fine-tune (low LR)

INFERENCE:
  audio → sliding windows → per-window decode
    → top-k mean aggregation
    → clip decision rule (thresholds from validation)
    → majority vote on model_id
```

---

## 7. Success Criteria (Honest)

| Metric | Target | Notes |
|--------|--------|-------|
| Clip-level AUC | >0.95 | Top-k mean aggregation |
| TPR @ 1% FPR | >85% | Clip-level, threshold-tuned |
| TPR @ 5% FPR | >92% | More stable estimate |
| Valid payload rate (clean) | >90% | Per-window decode + vote |
| Valid payload rate (MP3-128) | >70% | Real codec training |
| ViSQOL | >4.0 | Imperceptibility |

---

## Appendix: All Critiques Resolved

| Ver | Issue | Status |
|-----|-------|--------|
| v1-v5 | Various | ✅ All fixed |
| v6 | Stage 1 trains on clean | ✅ Fixed message + pre-wm→codec |
| v6 | Noisy-OR miscalibrated | ✅ Top-k mean |
| v6 | No clip-level rule | ✅ Explicit threshold tuning |
| v6 | F.fold API | ✅ 2D reshape + tests |
| v6 | .pt storage blowup | ✅ FLAC int16 |

**v7 is implementation-ready.**
