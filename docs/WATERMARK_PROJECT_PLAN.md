# FYP Project Plan v8: Audio Watermarking (Final)

> **All blockers resolved** — Implementation-ready

---

## v8 Fixes Summary

| v7 Issue | v8 Fix |
|----------|--------|
| Stage-1 has no negatives | **Add clean→codec negatives** (50/50 split) |
| Decoder output interface mismatch | **Unified output: detect + message per window** |
| Fixed message uses 0.5 | **Binary zeros** |
| Window/hop mismatch | **Aligned: 0.5 for both** |
| Payload not trained under codec | **Stage-1B: varying messages** |
| Threshold tuning ignores full rule | **Tune with entire decision function** |

---

## 1. Stage-1 with Negatives

```python
def generate_stage1_tensors_v8(manifest_path: Path, output_dir: Path):
    """
    Generate BOTH positives and negatives for Stage 1.
    Positive: clean → watermark(fixed_msg) → codec → FLAC
    Negative: clean → codec → FLAC (no watermark)
    """
    embedder = SpreadSpectrumEmbedder()
    
    # Fixed binary message (not 0.5!)
    FIXED_MESSAGE = torch.zeros(32)
    FIXED_MESSAGE[0:16] = PREAMBLE  # Preamble only, payload = 0
    
    codecs = ["mp3_64", "mp3_128", "aac_96"]
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    stage1_samples = []
    
    for i, item in enumerate(manifest):
        clean, sr = torchaudio.load(item["audio_path"])
        clean = clean.mean(dim=0, keepdim=True)
        
        # === POSITIVE: watermarked + coded ===
        wm = embedder(clean.unsqueeze(0), FIXED_MESSAGE.unsqueeze(0)).squeeze(0)
        
        for codec in codecs:
            # Save watermarked + coded
            pos_path = output_dir / f"{i:04d}_{codec}_pos.flac"
            coded_wm = apply_codec(wm, codec)
            save_audio_flac(coded_wm, pos_path, sr)
            
            stage1_samples.append({
                "path": str(pos_path),
                "has_watermark": 1.0,
                "codec": codec,
            })
        
        # === NEGATIVE: clean + coded (no watermark) ===
        for codec in codecs:
            neg_path = output_dir / f"{i:04d}_{codec}_neg.flac"
            coded_clean = apply_codec(clean, codec)
            save_audio_flac(coded_clean, neg_path, sr)
            
            stage1_samples.append({
                "path": str(neg_path),
                "has_watermark": 0.0,  # NEGATIVE!
                "codec": codec,
            })
        
        print(f"Stage 1 prep: {i+1}/{len(manifest)}")
    
    with open(output_dir / "stage1_manifest.json", "w") as f:
        json.dump(stage1_samples, f, indent=2)
    
    print(f"Generated {len(stage1_samples)} samples (50% pos, 50% neg)")


class Stage1DatasetV8(Dataset):
    """Stage 1 dataset with BOTH positives and negatives."""
    
    def __init__(self, manifest_path: Path):
        with open(manifest_path) as f:
            self.samples = json.load(f)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = load_audio_flac(item["path"])
        
        return {
            "audio": audio,
            "has_watermark": torch.tensor(item["has_watermark"]),
            "codec": item["codec"],
        }
    
    def __len__(self):
        return len(self.samples)
```

---

## 2. Unified Decoder Output Interface

```python
class UnifiedDecoder(nn.Module):
    """
    Returns BOTH detect and message probs per window.
    Fixes interface mismatch with ClipDecisionRule.
    """
    def __init__(self, base_decoder, window=16000, hop_ratio=0.5, top_k=3):
        super().__init__()
        self.decoder = base_decoder
        self.window = window
        self.hop = int(window * hop_ratio)  # ALIGNED with encoder
        self.top_k = top_k
    
    def forward(self, audio: torch.Tensor) -> dict:
        B, T = audio.shape
        
        if T < self.window:
            audio = F.pad(audio, (0, self.window - T))
            T = self.window
        
        n_win = (T - self.window) // self.hop + 1
        windows = audio.unfold(1, self.window, self.hop)
        windows_flat = windows.reshape(-1, self.window)
        
        outputs = self.decoder(windows_flat)
        
        # Reshape: (B, n_win, ...)
        detect = outputs["detect_prob"].reshape(B, n_win)
        message = outputs["message_prob"].reshape(B, n_win, -1)
        model = outputs["model_logits"].reshape(B, n_win, -1)
        
        # Top-k mean for clip-level detection
        top_k_vals, top_k_idx = torch.topk(detect, min(self.top_k, n_win), dim=1)
        clip_detect = top_k_vals.mean(dim=1)
        
        return {
            # Clip-level
            "clip_detect_prob": clip_detect,
            
            # Per-window (BOTH detect and message!)
            "all_window_probs": detect,        # (B, n_win)
            "all_message_probs": message,      # (B, n_win, 32)
            "all_model_logits": model,         # (B, n_win, n_models)
            
            # Metadata
            "n_windows": n_win,
            "top_k_idx": top_k_idx,
        }
```

---

## 3. Stage-1B: Payload Under Codec (Varying Messages)

```python
def generate_stage1b_tensors(manifest_path: Path, output_dir: Path, codec: MessageCodecV5):
    """
    Stage-1B: Train payload decoding under codec.
    Uses VARYING messages (not fixed) so decoder learns to recover bits.
    """
    embedder = SpreadSpectrumEmbedder()
    codecs = ["mp3_128"]  # Focus on one codec for payload training
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    samples = []
    
    for i, item in enumerate(manifest):
        # Random model_id and version for each sample
        model_id = random.randint(0, 7)
        version = random.randint(0, 15)
        message = codec.encode(model_id, version)
        
        clean, sr = torchaudio.load(item["audio_path"])
        clean = clean.mean(dim=0, keepdim=True)
        
        wm = embedder(clean.unsqueeze(0), message.unsqueeze(0)).squeeze(0)
        
        for codec_name in codecs:
            out_path = output_dir / f"{i:04d}_{codec_name}_payload.flac"
            coded = apply_codec(wm, codec_name)
            save_audio_flac(coded, out_path, sr)
            
            samples.append({
                "path": str(out_path),
                "message": message.tolist(),
                "model_id": model_id,
                "version": version,
            })
    
    with open(output_dir / "stage1b_manifest.json", "w") as f:
        json.dump(samples, f, indent=2)


def train_stage1b(decoder, stage1b_loader, device, epochs=10):
    """
    Stage-1B: Train decoder to recover PAYLOAD under codecs.
    Separate from Stage-1 detection training.
    """
    print("=== Stage 1B: Payload under Codec ===")
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-4)  # Lower LR
    
    for epoch in range(epochs):
        for batch in stage1b_loader:
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            outputs = decoder(audio)
            
            # Train message recovery
            loss = F.binary_cross_entropy(
                outputs["message_prob"],
                message
            )
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        print(f"Stage 1B Epoch {epoch+1}: loss={loss.item():.4f}")
```

---

## 4. Full Decision Rule Threshold Tuning

```python
def tune_thresholds_full_rule(decoder, val_loader, codec, decision_rule, target_fpr=0.01):
    """
    Tune thresholds using the ENTIRE decision function.
    Not just clip_detect_prob, but preamble validity too.
    """
    all_decisions = []
    all_labels = []
    
    for batch in val_loader:
        outputs = decoder(batch["audio"])
        
        for i in range(len(batch["audio"])):
            # Run full decision rule
            single_output = {
                "clip_detect_prob": outputs["clip_detect_prob"][i:i+1],
                "all_window_probs": outputs["all_window_probs"][i:i+1],
                "all_message_probs": outputs["all_message_probs"][i:i+1],
            }
            
            decision = decision_rule.decide(single_output, codec)
            all_decisions.append(decision["positive"])
            all_labels.append(batch["has_watermark"][i].item())
    
    all_decisions = np.array(all_decisions)
    all_labels = np.array(all_labels)
    
    # Compute FPR/TPR for current thresholds
    neg_mask = all_labels == 0
    pos_mask = all_labels == 1
    
    current_fpr = all_decisions[neg_mask].mean()
    current_tpr = all_decisions[pos_mask].mean()
    
    print(f"Current: FPR={current_fpr:.3f}, TPR={current_tpr:.3f}")
    
    # Grid search over threshold values
    best_threshold = None
    best_tpr = 0
    
    for detect_thresh in np.arange(0.5, 0.95, 0.05):
        decision_rule.detect_threshold = detect_thresh
        
        decisions = []
        for batch in val_loader:
            outputs = decoder(batch["audio"])
            for i in range(len(batch["audio"])):
                single_output = {k: v[i:i+1] for k, v in outputs.items()}
                d = decision_rule.decide(single_output, codec)
                decisions.append(d["positive"])
        
        decisions = np.array(decisions)
        fpr = decisions[neg_mask].mean()
        tpr = decisions[pos_mask].mean()
        
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_threshold = detect_thresh
    
    print(f"Best: threshold={best_threshold}, TPR={best_tpr:.3f} @ FPR<={target_fpr}")
    decision_rule.detect_threshold = best_threshold
    
    return {"threshold": best_threshold, "tpr": best_tpr}
```

---

## 5. Aligned Window/Hop (0.5 for Both)

```python
# ALIGNED: both encoder and decoder use hop_ratio=0.5
WINDOW_SAMPLES = 16000  # 1 second at 16kHz
HOP_RATIO = 0.5         # 50% overlap

encoder = SafeOverlapAddEncoder(base_encoder, window=WINDOW_SAMPLES, hop_ratio=HOP_RATIO)
decoder = UnifiedDecoder(base_decoder, window=WINDOW_SAMPLES, hop_ratio=HOP_RATIO, top_k=3)
```

---

## 6. Complete Training Pipeline

```python
def train_full_pipeline_v8(encoder, decoder, config):
    """
    Complete 4-stage training pipeline.
    """
    codec = MessageCodecV8()
    
    # === STAGE 1: Detection under codec (with negatives!) ===
    print("=== Stage 1: Detection (pos + neg) ===")
    stage1_dataset = Stage1DatasetV8(config["stage1_manifest"])
    stage1_loader = DataLoader(stage1_dataset, batch_size=16, shuffle=True)
    
    for p in encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(20):
        for batch in stage1_loader:
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)
            
            outputs = decoder(audio)
            loss = F.binary_cross_entropy(outputs["clip_detect_prob"], has_wm)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # === STAGE 1B: Payload under codec ===
    print("=== Stage 1B: Payload recovery ===")
    stage1b_dataset = Stage1BDataset(config["stage1b_manifest"])
    stage1b_loader = DataLoader(stage1b_dataset, batch_size=16, shuffle=True)
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    
    for epoch in range(10):
        for batch in stage1b_loader:
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            outputs = decoder(audio)
            loss = F.binary_cross_entropy(outputs["all_message_probs"].mean(dim=1), message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # === STAGE 2: Encoder (differentiable only) ===
    print("=== Stage 2: Encoder ===")
    for p in encoder.parameters():
        p.requires_grad = True
    for p in decoder.parameters():
        p.requires_grad = False
    
    # ... (same as v7) ...
    
    # === STAGE 3: Joint fine-tune ===
    print("=== Stage 3: Joint ===")
    for p in decoder.parameters():
        p.requires_grad = True
    
    # ... (same as v7) ...
```

---

## 7. Success Criteria (Updated)

| Metric | Target | Notes |
|--------|--------|-------|
| Clip-level AUC | >0.95 | With negatives in training |
| TPR @ 1% FPR | >85% | Full rule threshold tuning |
| TPR @ 5% FPR | >92% | More stable |
| Valid payload rate (clean) | >90% | Per-window decode |
| **Valid payload rate (MP3-128)** | **>70%** | **Stage-1B trained** |
| ViSQOL | >4.0 | Imperceptibility |

---

## Appendix: All Critiques Resolved

| Ver | Issue | Status |
|-----|-------|--------|
| v1-v6 | Various | ✅ Fixed |
| v7 | Stage-1 no negatives | ✅ 50/50 pos/neg |
| v7 | Decoder interface mismatch | ✅ Unified output |
| v7 | Fixed message uses 0.5 | ✅ Binary zeros |
| v7 | Window/hop mismatch | ✅ Aligned 0.5 |
| v7 | Payload not trained under codec | ✅ Stage-1B |
| v7 | Threshold tuning incomplete | ✅ Full rule tuning |

**v8 is implementation-ready.**
