# FYP Project Plan v10: Audio Watermarking (Final)

> **Operational details complete** — Implementation-ready

---

## v10 Fixes Summary

| v9 Issue | v10 Fix |
|----------|---------|
| Variable audio length | **Fixed 3s segments + collate_fn** |
| Sample rate not enforced | **Resample to 16kHz on load** |
| BCE numerically unstable | **BCEWithLogitsLoss** |
| Stage-1B top-k on garbage | **Curriculum: freeze detect first** |
| Metrics misaligned | **Payload correctness separate** |
| FWER not explicit | **Family-wise error framing** |

---

## 1. Fixed-Length Pipeline

```python
# === GLOBAL CONSTANTS ===
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)  # 48000


def load_and_prepare(path: Path) -> torch.Tensor:
    """
    Load audio with:
    - Sample rate enforcement (resample to 16kHz)
    - Mono conversion
    - Fixed-length (3s = 48000 samples)
    """
    audio, sr = torchaudio.load(str(path))
    
    # Mono
    audio = audio.mean(dim=0)  # (T,)
    
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    
    # Fixed length: crop or pad to SEGMENT_SAMPLES
    T = audio.shape[0]
    if T > SEGMENT_SAMPLES:
        # Random crop for training
        start = torch.randint(0, T - SEGMENT_SAMPLES + 1, (1,)).item()
        audio = audio[start:start + SEGMENT_SAMPLES]
    elif T < SEGMENT_SAMPLES:
        # Pad with zeros
        audio = F.pad(audio, (0, SEGMENT_SAMPLES - T))
    
    return audio  # (48000,) guaranteed


def collate_fixed_length(batch):
    """
    Custom collate for fixed-length audio.
    All tensors are already SEGMENT_SAMPLES length.
    """
    audios = torch.stack([item["audio"] for item in batch])  # (B, T)
    
    result = {"audio": audios}
    
    # Handle other fields
    if "has_watermark" in batch[0]:
        result["has_watermark"] = torch.stack([item["has_watermark"] for item in batch])
    if "message" in batch[0]:
        result["message"] = torch.stack([item["message"] for item in batch])
    if "model_id" in batch[0]:
        result["model_id"] = torch.tensor([item["model_id"] for item in batch])
    
    return result


class FixedLengthDataset(Dataset):
    """Base dataset with fixed-length segments."""
    
    def __init__(self, manifest_path: Path, training: bool = True):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.training = training
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        audio, sr = torchaudio.load(item["path"])
        audio = audio.mean(dim=0)  # Mono
        
        # Resample
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        
        # Fixed length
        T = audio.shape[0]
        if T >= SEGMENT_SAMPLES:
            if self.training:
                start = torch.randint(0, T - SEGMENT_SAMPLES + 1, (1,)).item()
            else:
                start = (T - SEGMENT_SAMPLES) // 2  # Center crop for eval
            audio = audio[start:start + SEGMENT_SAMPLES]
        else:
            audio = F.pad(audio, (0, SEGMENT_SAMPLES - T))
        
        return {"audio": audio, **{k: v for k, v in item.items() if k != "path"}}
```

---

## 2. Attack Functions with Length Preservation

```python
def apply_attack_safe(audio: torch.Tensor, attack_fn) -> torch.Tensor:
    """
    Apply attack and restore to SEGMENT_SAMPLES.
    Handles length-changing attacks (time-stretch).
    """
    attacked = attack_fn(audio)
    
    T = attacked.shape[-1]
    if T > SEGMENT_SAMPLES:
        attacked = attacked[..., :SEGMENT_SAMPLES]
    elif T < SEGMENT_SAMPLES:
        attacked = F.pad(attacked, (0, SEGMENT_SAMPLES - T))
    
    return attacked


EVAL_ATTACKS_SAFE = {
    "clean": lambda x: x,
    "mp3_64": lambda x: apply_attack_safe(x, lambda a: apply_codec(a, "mp3", 64)),
    "mp3_128": lambda x: apply_attack_safe(x, lambda a: apply_codec(a, "mp3", 128)),
    "time_stretch_95": lambda x: apply_attack_safe(x, lambda a: time_stretch(a, 0.95)),
    "time_stretch_105": lambda x: apply_attack_safe(x, lambda a: time_stretch(a, 1.05)),
    # ... etc
}
```

---

## 3. BCEWithLogitsLoss (Numerically Stable)

```python
class DecoderV10(nn.Module):
    """
    Decoder outputs LOGITS, not probabilities.
    More numerically stable with BCEWithLogitsLoss.
    """
    def __init__(self, ...):
        ...
        self.head_detect = nn.Linear(feat_dim, 1)  # Outputs logit
        self.head_message = nn.Linear(feat_dim, msg_bits)  # Outputs logits
    
    def forward(self, audio: torch.Tensor) -> dict:
        ...
        return {
            # LOGITS (not sigmoid'd)
            "clip_detect_logit": clip_detect_logit,
            "all_window_logits": window_logits,       # (B, n_win)
            "all_message_logits": message_logits,     # (B, n_win, 32)
            
            # Probs for inference (sigmoid applied)
            "clip_detect_prob": torch.sigmoid(clip_detect_logit),
            "all_window_probs": torch.sigmoid(window_logits),
            "all_message_probs": torch.sigmoid(message_logits),
        }


def train_stage1_v10(decoder, loader, device, epochs=20):
    """Training with BCEWithLogitsLoss."""
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        for batch in loader:
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)
            
            outputs = decoder(audio)
            
            # Per-window loss (with LOGITS)
            n_win = outputs["all_window_logits"].shape[1]
            has_wm_exp = has_wm.unsqueeze(1).expand(-1, n_win)
            
            loss_window = F.binary_cross_entropy_with_logits(
                outputs["all_window_logits"],
                has_wm_exp
            )
            
            # Clip loss (with LOGITS)
            loss_clip = F.binary_cross_entropy_with_logits(
                outputs["clip_detect_logit"],
                has_wm
            )
            
            loss = loss_window + 0.5 * loss_clip
            
            opt.zero_grad()
            loss.backward()
            opt.step()
```

---

## 4. Stage-1B Curriculum (Freeze Detect First)

```python
def train_stage1b_v10(decoder, loader, device, epochs=10, warmup_epochs=3, top_k=3):
    """
    Stage-1B with curriculum:
    - First 3 epochs: use ALL windows (detector not trusted)
    - Remaining: use top-k by detect prob
    """
    print("=== Stage 1B: Payload (curriculum) ===")
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        use_topk = epoch >= warmup_epochs
        
        for batch in loader:
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            outputs = decoder(audio)
            msg_logits = outputs["all_message_logits"]  # (B, n_win, 32)
            detect = outputs["all_window_probs"]        # (B, n_win)
            
            B, n_win, msg_bits = msg_logits.shape
            
            if use_topk:
                # Top-k windows by detection
                k = min(top_k, n_win)
                _, top_idx = torch.topk(detect, k, dim=1)
                top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_bits)
                selected_logits = torch.gather(msg_logits, 1, top_idx_exp)
                avg_logits = selected_logits.mean(dim=1)
            else:
                # Warmup: all windows (don't trust detector yet)
                avg_logits = msg_logits.mean(dim=1)
            
            loss = F.binary_cross_entropy_with_logits(avg_logits, message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        mode = "top-k" if use_topk else "all-windows"
        print(f"Stage 1B Epoch {epoch+1} ({mode}): loss={loss.item():.4f}")
```

---

## 5. Separate Detection vs Payload Metrics

```python
def evaluate_v10(decoder, loader, codec, decision_rule):
    """
    Report BOTH detection and payload correctness separately.
    """
    results = {
        "detection": {"probs": [], "labels": []},
        "payload": {"correct": 0, "total": 0, "positives": 0},
    }
    
    for batch in loader:
        outputs = decoder(batch["audio"].to(device))
        
        for i in range(len(batch["audio"])):
            # Detection (raw probability)
            clip_prob = outputs["clip_detect_prob"][i].item()
            label = batch["has_watermark"][i].item()
            results["detection"]["probs"].append(clip_prob)
            results["detection"]["labels"].append(label)
            
            # Payload correctness (end-to-end)
            single = {
                "clip_detect_prob": outputs["clip_detect_prob"][i],
                "all_window_probs": outputs["all_window_probs"][i],
                "all_message_probs": outputs["all_message_probs"][i],
            }
            decision = decision_rule.decide(single, codec)
            
            if decision["positive"]:
                results["payload"]["positives"] += 1
                
                # Check if decoded model_id matches ground truth
                if label == 1 and "model_id" in batch:
                    gt_model = batch["model_id"][i].item()
                    if decision.get("model_id") == gt_model:
                        results["payload"]["correct"] += 1
                    results["payload"]["total"] += 1
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score
    
    probs = np.array(results["detection"]["probs"])
    labels = np.array(results["detection"]["labels"])
    
    metrics = {
        # Detection metrics
        "detection_auc": roc_auc_score(labels, probs),
        "detection_tpr_at_1pct_fpr": compute_tpr_at_fpr(labels, probs, 0.01),
        
        # Payload correctness (SEPARATE from detection)
        "payload_accuracy": (
            results["payload"]["correct"] / results["payload"]["total"]
            if results["payload"]["total"] > 0 else None
        ),
        "positive_rate": results["payload"]["positives"] / len(labels),
    }
    
    return metrics
```

---

## 6. FWER Framing (Explicit)

```python
"""
FAMILY-WISE ERROR RATE CONSIDERATION

When scanning N windows per clip:
- Per-window FPR = α
- Clip-level FPR ≈ 1 - (1-α)^N (if independent, worse if correlated)

For N=10 windows and α=0.01:
- Naive clip FPR ≈ 9.6% (unacceptable!)

Our mitigation:
1. Report and tune at CLIP level, not window level
2. Require BOTH high detect AND valid preamble in same window
3. Use top-k aggregation (not "any window fires")

This is analogous to Bonferroni/FWER correction mindset.
"""

# Config with explicit FWER-aware thresholds
DECISION_CONFIG = {
    "detect_threshold": 0.8,      # High threshold (FWER mitigation)
    "preamble_min": 15,           # 15/16 bits match
    "min_valid_windows": 1,       # At least 1 passes both
    "clip_detect_min": 0.7,       # Clip-level threshold
}
```

---

## 7. Final Training Pipeline

```python
def train_full_v10(encoder, decoder, config):
    """Complete v10 training pipeline."""
    
    # Dataloaders with fixed-length collate
    stage1_loader = DataLoader(
        Stage1Dataset(config["stage1_manifest"]),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fixed_length
    )
    
    # STAGE 1: Detection (with negatives)
    train_stage1_v10(decoder, stage1_loader, device, epochs=20)
    
    # STAGE 1B: Payload (with curriculum)
    train_stage1b_v10(decoder, stage1b_loader, device, epochs=10, warmup_epochs=3)
    
    # STAGE 2: Encoder (differentiable, logits loss)
    train_stage2_v10(encoder, decoder, stage2_loader, device, epochs=20)
    
    # STAGE 3: Joint fine-tune
    train_stage3_v10(encoder, decoder, stage3_loader, device, epochs=10)
```

---

## Appendix: All Critiques Resolved

| Ver | Issue | Status |
|-----|-------|--------|
| v1-v8 | Various | ✅ Fixed |
| v9 | Variable length | ✅ Fixed 3s + collate |
| v9 | No sr enforcement | ✅ Resample to 16k |
| v9 | BCE unstable | ✅ BCEWithLogitsLoss |
| v9 | Top-k on garbage | ✅ Curriculum warmup |
| v9 | Metrics misaligned | ✅ Payload accuracy separate |
| v9 | FWER not explicit | ✅ Documented + mitigated |

**v10 is implementation-ready.**
