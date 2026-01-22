# FYP Project Plan v9: Audio Watermarking (Final)

> **All practical failure modes addressed** — Implementation-ready

---

## v9 Fixes Summary

| v8 Issue | v9 Fix |
|----------|--------|
| clip_detect loss → topk gradient | **Per-window BCE auxiliary loss** |
| Stage-1B averages all windows | **Payload loss on top-k windows only** |
| Shape inconsistency | **Uniform (B, T) mono everywhere** |
| tune_thresholds slices ints | **Only pass tensor keys** |
| Attack suite too narrow | **Expanded eval: resample, reverb, noise** |
| Success criteria unmeasurable | **Stated negative count + confidence** |

---

## 1. Per-Window Auxiliary Detection Loss

```python
def train_stage1_v9(decoder, stage1_loader, device, epochs=20):
    """
    Stage 1 with BOTH:
    - Per-window BCE (stable gradients, robust learning)
    - Clip-level BCE (for aggregation calibration)
    """
    print("=== Stage 1: Detection (per-window + clip) ===")
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in stage1_loader:
            audio = batch["audio"].to(device)  # (B, T) - enforced shape
            has_wm = batch["has_watermark"].to(device)  # (B,)
            
            outputs = decoder(audio)
            
            # CLIP-LEVEL loss (for aggregation)
            loss_clip = F.binary_cross_entropy(
                outputs["clip_detect_prob"],
                has_wm
            )
            
            # PER-WINDOW auxiliary loss (stable gradients!)
            # Expand label to all windows: (B,) -> (B, n_win)
            n_win = outputs["all_window_probs"].shape[1]
            has_wm_expanded = has_wm.unsqueeze(1).expand(-1, n_win)
            
            loss_window = F.binary_cross_entropy(
                outputs["all_window_probs"],
                has_wm_expanded
            )
            
            # Combined: per-window + clip (weighted)
            loss = loss_window + 0.5 * loss_clip
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
        
        print(f"Stage 1 Epoch {epoch+1}: loss={total_loss/len(stage1_loader):.4f}")
```

---

## 2. Stage-1B Payload Loss on Top-K Windows Only

```python
def train_stage1b_v9(decoder, stage1b_loader, device, epochs=10, top_k=3):
    """
    Stage-1B: Payload loss ONLY on high-confidence windows.
    Fixes the garbage-averaging problem.
    """
    print("=== Stage 1B: Payload (top-k windows) ===")
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in stage1b_loader:
            audio = batch["audio"].to(device)      # (B, T)
            message = batch["message"].to(device)  # (B, 32)
            
            outputs = decoder(audio)
            
            detect = outputs["all_window_probs"]     # (B, n_win)
            msg_probs = outputs["all_message_probs"] # (B, n_win, 32)
            
            B, n_win, msg_bits = msg_probs.shape
            
            # Get top-k windows by detection score
            k = min(top_k, n_win)
            _, top_idx = torch.topk(detect, k, dim=1)  # (B, k)
            
            # Gather top-k message probs: (B, k, 32)
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_bits)
            top_msg_probs = torch.gather(msg_probs, 1, top_idx_exp)
            
            # Average over top-k windows only
            avg_top_msg = top_msg_probs.mean(dim=1)  # (B, 32)
            
            # BCE on top-k average (not all windows!)
            loss = F.binary_cross_entropy(avg_top_msg, message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
        
        print(f"Stage 1B Epoch {epoch+1}: loss={total_loss/len(stage1b_loader):.4f}")
```

---

## 3. Uniform Shape Enforcement

```python
# === CANONICAL SHAPE: (B, T) for all audio tensors ===

class Stage1DatasetV9(Dataset):
    """Enforces mono (T,) output, DataLoader gives (B, T)."""
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        audio, sr = torchaudio.load(item["path"])
        
        # Enforce mono, squeeze to (T,)
        audio = audio.mean(dim=0)  # (channels, T) -> (T,)
        
        return {
            "audio": audio,  # (T,) float32
            "has_watermark": torch.tensor(item["has_watermark"], dtype=torch.float32),
        }


class Stage1BDatasetV9(Dataset):
    """Payload dataset with proper tensor conversion."""
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        audio, sr = torchaudio.load(item["path"])
        audio = audio.mean(dim=0)  # (T,)
        
        # Convert message list to tensor!
        message = torch.tensor(item["message"], dtype=torch.float32)
        
        return {
            "audio": audio,       # (T,)
            "message": message,   # (32,)
            "model_id": item["model_id"],
        }


class UnifiedDecoderV9(nn.Module):
    """Expects input shape (B, T)."""
    
    def forward(self, audio: torch.Tensor) -> dict:
        # Validate input shape
        assert audio.dim() == 2, f"Expected (B, T), got {audio.shape}"
        B, T = audio.shape
        
        # ... rest of implementation ...
```

---

## 4. Fixed Threshold Tuning (Tensor Keys Only)

```python
def tune_thresholds_v9(decoder, val_loader, codec, decision_rule, target_fpr=0.01):
    """
    Threshold tuning with ONLY tensor keys passed to decision rule.
    Fixes the 'slicing int' crash.
    """
    all_decisions = []
    all_labels = []
    
    for batch in val_loader:
        outputs = decoder(batch["audio"].to(device))
        
        for i in range(len(batch["audio"])):
            # Only pass TENSOR keys that the rule needs
            single_output = {
                "clip_detect_prob": outputs["clip_detect_prob"][i],
                "all_window_probs": outputs["all_window_probs"][i],
                "all_message_probs": outputs["all_message_probs"][i],
                # NOT: n_windows (int), top_k_idx (not needed)
            }
            
            decision = decision_rule.decide(single_output, codec)
            all_decisions.append(1 if decision["positive"] else 0)
            all_labels.append(int(batch["has_watermark"][i].item()))
    
    # ... rest of implementation same as v8 ...
```

---

## 5. Expanded Evaluation Attack Suite

```python
EVAL_ATTACKS = {
    # Codec (trained on)
    "mp3_64": lambda x: apply_codec(x, "mp3", 64),
    "mp3_128": lambda x: apply_codec(x, "mp3", 128),
    "mp3_320": lambda x: apply_codec(x, "mp3", 320),
    "aac_96": lambda x: apply_codec(x, "aac", 96),
    
    # Noise (partial training)
    "noise_snr20": lambda x: add_noise(x, snr=20),
    "noise_snr30": lambda x: add_noise(x, snr=30),
    
    # NEW: Evaluation-only attacks (not trained on)
    "resample_8k": lambda x: resample_round_trip(x, 16000, 8000),
    "resample_48k": lambda x: resample_round_trip(x, 16000, 48000),
    "reverb_small": lambda x: apply_reverb(x, rt60=0.3),
    "reverb_large": lambda x: apply_reverb(x, rt60=0.8),
    "time_stretch_95": lambda x: time_stretch(x, factor=0.95),
    "time_stretch_105": lambda x: time_stretch(x, factor=1.05),
    "background_noise": lambda x: mix_with_noise(x, snr=15),
}


def evaluate_per_attack(decoder, test_loader, codec, decision_rule):
    """
    Per-attack evaluation following AudioMarkBench style.
    Reports clip-level metrics for each attack.
    """
    results = {}
    
    for attack_name, attack_fn in EVAL_ATTACKS.items():
        clip_probs = []
        clip_labels = []
        valid_payloads = []
        
        for batch in test_loader:
            # Apply attack
            attacked = attack_fn(batch["audio"])
            
            outputs = decoder(attacked.to(device))
            
            for i in range(len(batch["audio"])):
                clip_probs.append(outputs["clip_detect_prob"][i].item())
                clip_labels.append(int(batch["has_watermark"][i].item()))
                
                # Check payload validity
                single = {
                    "clip_detect_prob": outputs["clip_detect_prob"][i],
                    "all_window_probs": outputs["all_window_probs"][i],
                    "all_message_probs": outputs["all_message_probs"][i],
                }
                decision = decision_rule.decide(single, codec)
                valid_payloads.append(decision.get("positive", False))
        
        # Compute metrics
        from sklearn.metrics import roc_auc_score
        
        results[attack_name] = {
            "auc": roc_auc_score(clip_labels, clip_probs) if len(set(clip_labels)) > 1 else None,
            "valid_payload_rate": sum(valid_payloads) / len(valid_payloads),
            "n_samples": len(clip_labels),
        }
        
        print(f"{attack_name}: AUC={results[attack_name]['auc']:.3f}, "
              f"Payload={results[attack_name]['valid_payload_rate']:.1%}")
    
    return results
```

---

## 6. Success Criteria with Confidence

```python
def compute_metrics_with_confidence(y_true, y_prob, n_bootstrap=1000):
    """
    Compute metrics with 95% confidence intervals via bootstrap.
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np
    
    aucs = []
    tprs_at_1pct = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        y_t = np.array(y_true)[idx]
        y_p = np.array(y_prob)[idx]
        
        if len(set(y_t)) < 2:
            continue
        
        aucs.append(roc_auc_score(y_t, y_p))
        
        # TPR @ 1% FPR
        neg = y_p[y_t == 0]
        pos = y_p[y_t == 1]
        thresh = np.percentile(neg, 99)
        tpr = (pos >= thresh).mean()
        tprs_at_1pct.append(tpr)
    
    return {
        "auc_mean": np.mean(aucs),
        "auc_95ci": (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)),
        "tpr_1pct_mean": np.mean(tprs_at_1pct),
        "tpr_1pct_95ci": (np.percentile(tprs_at_1pct, 2.5), np.percentile(tprs_at_1pct, 97.5)),
    }
```

### Success Criteria (with sample size)

| Metric | Target | Required Samples | Notes |
|--------|--------|------------------|-------|
| AUC | >0.95 | 200+ pos, 200+ neg | 95% CI <±0.03 |
| TPR @ 1% FPR | >85% | 500+ neg | Stable estimate |
| TPR @ 5% FPR | >92% | 200+ neg | More stable |
| Valid payload (clean) | >90% | 100+ samples | |
| Valid payload (MP3-128) | >70% | 100+ samples | Stage-1B trained |

---

## Appendix: All Critiques Resolved

| Ver | Issue | Status |
|-----|-------|--------|
| v1-v7 | Various | ✅ Fixed |
| v8 | clip_detect topk gradients | ✅ Per-window aux loss |
| v8 | Stage-1B all-window avg | ✅ Top-k window loss |
| v8 | Shape (B,T) vs (B,1,T) | ✅ Uniform (B,T) |
| v8 | tune slices ints | ✅ Tensor keys only |
| v8 | Narrow attack suite | ✅ Expanded eval |
| v8 | Unmeasurable criteria | ✅ CI + sample sizes |

**v9 is implementation-ready.**
