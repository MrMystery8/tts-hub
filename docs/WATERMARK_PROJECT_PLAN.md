# FYP Project Plan v3: Audio Watermarking (Final)

> **Final Version** — Fixes encoder bit-encoding, adds sync mechanism, addresses all ChatGPT critiques

---

## Summary of v3 Changes

| v2 Issue | v3 Fix |
|----------|--------|
| Encoder doesn't encode bits | **FiLM conditioning** - message modulates conv features |
| No sync for cropping | **Repeated embedding + sliding window decoding** |
| 300 samples too small | Target **600+ samples** (1-2 hours) |
| BatchNorm unstable | Switch to **GroupNorm** |
| Joint training collapse risk | **2-stage training** (decoder first) |
| No forgery testing | Add **copy attack** evaluation |
| PESQ licensing | Use **ViSQOL** (open-source) |

---

## 1. Fixed Encoder with FiLM Conditioning

### The Problem (v2)

```python
# v2 - WRONG: Just scales amplitude, doesn't encode bits
msg_scale = torch.sigmoid(msg_embed.mean(dim=1))  # Single scalar!
watermark = watermark * msg_scale  # Same pattern, different volume
```

### The Fix (v3) - FiLM Modulation

```python
class WatermarkEncoderV3(nn.Module):
    """
    Encoder with FiLM conditioning - message ACTUALLY modulates watermark content.
    """
    def __init__(self, msg_bits: int = 32, hidden: int = 32):
        super().__init__()
        
        # Message to FiLM parameters (gamma, beta per layer)
        self.film1 = nn.Linear(msg_bits, hidden * 2)  # gamma + beta
        self.film2 = nn.Linear(msg_bits, hidden * 2)
        self.film3 = nn.Linear(msg_bits, hidden * 2)
        
        # Conv layers (no BatchNorm - use GroupNorm for Mac stability)
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=7, padding=3)
        self.gn1 = nn.GroupNorm(4, hidden)
        
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(4, hidden)
        
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(4, hidden)
        
        self.out_conv = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
        
        self.alpha = nn.Parameter(torch.tensor(0.02))
        
    def _apply_film(self, x, film_params):
        """Apply FiLM: gamma * x + beta"""
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta
        
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Each message bit pattern produces a DIFFERENT watermark signal.
        """
        B, _, T = audio.shape
        
        # Layer 1 + FiLM
        x = self.conv1(audio)
        x = self.gn1(x)
        x = self._apply_film(x, self.film1(message))  # Message modulates features!
        x = torch.relu(x)
        
        # Layer 2 + FiLM
        x = self.conv2(x)
        x = self.gn2(x)
        x = self._apply_film(x, self.film2(message))
        x = torch.relu(x)
        
        # Layer 3 + FiLM
        x = self.conv3(x)
        x = self.gn3(x)
        x = self._apply_film(x, self.film3(message))
        x = torch.relu(x)
        
        # Output watermark
        watermark = torch.tanh(self.out_conv(x))
        
        # Add with constrained strength
        alpha = torch.clamp(self.alpha, 0.01, 0.1)
        return audio + alpha * watermark
```

### Why This Works

- Each message creates **unique (gamma, beta)** pairs at every layer
- Different bit patterns → different intermediate feature activations → different watermark shapes
- Decoder can now actually **recover which bits** were embedded

---

## 2. Sync Mechanism for Cropping Robustness

### The Problem (v2)

```
Original:    [====WATERMARK====]
After crop:  [==TERMARK====]  ← Decoder expects aligned start, fails
```

### The Fix (v3) - Repeated Embedding + Sliding Window

```python
class SyncAwareEncoder(nn.Module):
    """Embeds watermark REPEATEDLY for crop robustness."""
    
    def __init__(self, base_encoder, window_samples: int = 16000):
        super().__init__()
        self.encoder = base_encoder
        self.window = window_samples  # 1 second at 16kHz
        
    def forward(self, audio: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        B, C, T = audio.shape
        
        # Embed watermark in each window
        watermarked = audio.clone()
        for start in range(0, T, self.window):
            end = min(start + self.window, T)
            if end - start < self.window // 2:
                continue  # Skip very short trailing segments
                
            segment = audio[:, :, start:end]
            
            # Pad if needed
            if segment.shape[-1] < self.window:
                segment = F.pad(segment, (0, self.window - segment.shape[-1]))
            
            wm_segment = self.encoder(segment, message)
            watermarked[:, :, start:end] = wm_segment[:, :, :end-start]
        
        return watermarked


class SyncAwareDecoder(nn.Module):
    """Decodes using sliding windows + voting."""
    
    def __init__(self, base_decoder, window_samples: int = 16000, hop: int = 8000):
        super().__init__()
        self.decoder = base_decoder
        self.window = window_samples
        self.hop = hop
        
    def forward(self, audio: torch.Tensor) -> dict:
        B, T = audio.shape
        
        all_detect = []
        all_bits = []
        all_model = []
        
        # Sliding window
        for start in range(0, T - self.window + 1, self.hop):
            segment = audio[:, start:start + self.window]
            out = self.decoder(segment)
            all_detect.append(out["detect_prob"])
            all_bits.append(out["message_prob"])
            all_model.append(out["model_logits"])
        
        if not all_detect:
            # Fallback for very short audio
            return self.decoder(audio)
        
        # Aggregate: take max detection, majority vote on bits
        detect_prob = torch.stack(all_detect).max(dim=0)[0]
        message_prob = torch.stack(all_bits).mean(dim=0)  # Average probabilities
        model_logits = torch.stack(all_model).mean(dim=0)  # Average logits
        
        return {
            "detect_prob": detect_prob,
            "message_prob": message_prob,
            "model_logits": model_logits,
            "n_windows": len(all_detect),
        }
```

### Why This Works

- Watermark repeats every 1 second → any 1-second crop contains full watermark
- Sliding window decoder tests multiple positions → finds watermark even if misaligned
- Voting/averaging across windows improves robustness

---

## 3. Two-Stage Training (Safer)

### Why Joint Training Can Collapse

> "Jointly training encoder+decoder often collapses: encoder learns 'cheat' noise; decoder overfits." — Referenced from AudioSeal GitHub issues

### v3 Training Strategy

```
STAGE 1: Train Decoder Only (~20 epochs)
├── Use simple spread-spectrum embedding (deterministic)
├── Apply heavy augmentation
├── Learn robust features for detection
└── Freeze decoder

STAGE 2: Train Encoder Only (~20 epochs)
├── Freeze decoder
├── Train encoder for imperceptibility + robustness
├── Focus on quality loss + detection pass-through
└── Freeze encoder

STAGE 3: Joint Fine-tuning (~10 epochs)
├── Unfreeze both
├── Low learning rate (1e-5)
├── Small updates to align encoder-decoder
└── Final checkpoint
```

### Implementation

```python
def train_two_stage(encoder, decoder, train_loader, device, config):
    
    # Stage 1: Decoder with fixed spread-spectrum
    print("=== Stage 1: Training Decoder ===")
    for param in encoder.parameters():
        param.requires_grad = False
    
    simple_embedder = SpreadSpectrumEmbedder()  # Deterministic
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(20):
        for batch in train_loader:
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            # Embed with simple method
            watermarked = simple_embedder(audio, message)
            augmented = augmenter(watermarked)
            
            outputs = decoder(augmented)
            loss = compute_decoder_loss(outputs, message)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Stage 2: Encoder with frozen decoder
    print("=== Stage 2: Training Encoder ===")
    for param in encoder.parameters():
        param.requires_grad = True
    for param in decoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4)
    
    for epoch in range(20):
        for batch in train_loader:
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            watermarked = encoder(audio, message)
            augmented = augmenter(watermarked)
            
            outputs = decoder(augmented)
            
            loss_quality = compute_stft_loss(audio, watermarked)
            loss_detect = compute_decoder_loss(outputs, message)
            loss = loss_detect + 10.0 * loss_quality
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Stage 3: Joint fine-tuning
    print("=== Stage 3: Joint Fine-tune ===")
    for param in decoder.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-5  # Very low LR
    )
    
    for epoch in range(10):
        # Same training loop with both models
        ...
```

---

## 4. Message/Payload Design (with ECC)

### Structure

```
32-bit payload:
├── Bits 0-3:   Sync pattern (fixed: 1010)
├── Bits 4-6:   Model ID (0-7)
├── Bits 7-10:  Version (0-15)
├── Bits 11-15: Timestamp hash (0-31)
├── Bits 16-23: Payload data (8 bits custom)
├── Bits 24-31: CRC-8 checksum
```

### Validation Logic

```python
def validate_payload(decoded_bits: torch.Tensor) -> dict:
    """Validate decoded payload with sync check and CRC."""
    bits = (decoded_bits > 0.5).int()
    
    # Check sync pattern (bits 0-3 should be 1010)
    sync = bits[0:4]
    expected_sync = torch.tensor([1, 0, 1, 0])
    sync_valid = (sync == expected_sync).all()
    
    if not sync_valid:
        return {"valid": False, "reason": "sync_mismatch"}
    
    # Extract fields
    model_id = bits[4] + 2*bits[5] + 4*bits[6]
    version = bits[7] + 2*bits[8] + 4*bits[9] + 8*bits[10]
    
    # Verify CRC
    payload_bits = bits[0:24]
    received_crc = bits[24:32]
    computed_crc = compute_crc8(payload_bits)
    
    crc_valid = (received_crc == computed_crc).all()
    
    return {
        "valid": crc_valid,
        "model_id": model_id.item(),
        "version": version.item(),
        "confidence": decoded_bits.mean().item(),
    }
```

---

## 5. Expanded Evaluation Suite

### Standard Transforms (Keep)

MP3-64/128/320, AAC-96, Noise SNR 20/30 dB, Resample, Reverb, Time-stretch, Crop

### NEW: Forgery/Copy Attack Tests

```python
def test_copy_attack(encoder, decoder, clean_audio, watermarked_audio):
    """
    Copy attack: Can we paste watermark residual onto clean audio?
    """
    # Extract "watermark residual"
    residual = watermarked_audio - clean_audio
    
    # Apply to different clean audio
    other_clean = load_random_clean_audio()
    forged = other_clean + residual
    
    # Does detector fire?
    outputs = decoder(forged)
    
    return {
        "forgery_detected": outputs["detect_prob"].item() > 0.5,
        "forgery_prob": outputs["detect_prob"].item(),
    }

def test_model_id_spoof(encoder, decoder, audio):
    """
    Spoof attack: Can attacker embed a different model_id?
    Without keyed MAC, this is possible - document as limitation.
    """
    # Embed with wrong model_id
    fake_message = create_message(model_id=99)
    spoofed = encoder(audio, fake_message)
    
    outputs = decoder(spoofed)
    decoded_id = outputs["model_logits"].argmax()
    
    return {
        "spoof_successful": decoded_id == 99,
        "note": "Keyed MAC required to prevent spoofing (future work)",
    }
```

### Quality Metrics (Use ViSQOL, not PESQ)

```python
# ViSQOL is open-source: https://github.com/google/visqol
from visqol import visqol_lib_py

def compute_quality_metrics(original, watermarked, sr=16000):
    """Compute audio quality using ViSQOL (open-source)."""
    
    # Save temp files
    sf.write("/tmp/orig.wav", original, sr)
    sf.write("/tmp/wm.wav", watermarked, sr)
    
    # ViSQOL
    config = visqol_lib_py.MakeVisqolConfig()
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    
    result = api.Measure("/tmp/orig.wav", "/tmp/wm.wav")
    
    # SNR
    noise = watermarked - original
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    
    return {
        "visqol": result.moslqo,
        "snr_db": snr,
    }
```

---

## 6. Updated Timeline (Realistic)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | FiLM encoder, sync decoder, basic training | E2E forward pass works |
| **Week 2** | 2-stage training, dataset expansion (600+ samples) | Stage 1 decoder trained |
| **Week 3** | Full training pipeline, augmentation suite | Both stages complete |
| **Week 4** | Evaluation: transforms + forgery tests | Metrics table complete |
| **Week 5** | Integration, write-up, quality metrics | Final report |

---

## 7. Success Criteria (Targets, Not Promises)

> Per AudioMarkBench + SoK, no method achieves perfect robustness. These are **hypotheses to test**, not guarantees.

| Metric | Target | Notes |
|--------|--------|-------|
| Detection (clean) | >95% | Should be easy |
| Detection (MP3-128) | >85% | Standard benchmark |
| Detection (cropped 50%) | >75% | Enabled by sync mechanism |
| BER (clean) | <5% | Bits should be recoverable |
| BER (MP3-128) | <15% | Some degradation expected |
| Model attribution | >85% | On watermarked samples |
| ViSQOL | >4.0 | Imperceptibility |
| SNR | >30 dB | Standard target |

---

## 8. Acknowledged Limitations

### Out of Scope (Future Work)

- **Neural codec attacks** (EnCodec, DAC) — too heavy for FYP
- **White-box attacks** — attacker with model weights
- **Keyed MAC for anti-spoofing** — would prevent model_id forging

### Known Weaknesses

> "Deep Audio Watermarks are Shallow" (2025) shows removal is possible with minimal quality loss in no-box settings. Our system targets practical scenarios, not adversarial robustness.

---

## Summary: v3 vs v2

| Aspect | v2 | v3 |
|--------|----|----|
| Bit encoding | ❌ Just scales amplitude | ✅ FiLM conditioning |
| Crop robustness | ❌ No sync | ✅ Repeated embed + sliding window |
| Training stability | ⚠️ Joint (risky) | ✅ 2-stage (safer) |
| Dataset size | 300 samples | 600+ samples |
| Forgery testing | ❌ Missing | ✅ Added |
| Quality metric | PESQ (licensed) | ViSQOL (open-source) |

**Rating target: 9/10 research alignment, 8/10 implementation readiness**
