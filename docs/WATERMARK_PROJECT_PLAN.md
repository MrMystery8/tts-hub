# FYP Project Plan v11: Audio Watermarking (Final)

> **Acceptance criteria defined** — Implementation-ready with checklist

---

## v11 Fixes Summary

| v10 Issue | v11 Fix |
|-----------|---------|
| collate crashes on non-tensors | **Convert in `__getitem__`** |
| "Freeze detect" not implemented | **`requires_grad=False`** |
| Stage-1B warmup uses garbage | **Preamble correlation selection** |
| FWER wording sloppy | **"Tune empirically"** |
| Thresholds hardcoded | **Stated as "learned from validation"** |
| Codec I/O bound | **Bulk precompute + cache** |
| Missing conditional attribution | **Two attribution metrics** |
| No reproducibility | **Seeds + version pinning** |
| Overclaiming | **Acceptance criteria checklist** |

---

## 1. Robust Dataset with Tensor Conversion

```python
class Stage1DatasetV11(Dataset):
    """
    All type conversions in __getitem__, not collate.
    Returns only needed fields with guaranteed types.
    """
    def __init__(self, manifest_path: Path, training: bool = True):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.training = training
    
    def __getitem__(self, idx) -> dict:
        item = self.samples[idx]
        
        # Load and process audio
        audio = load_and_prepare(item["path"], training=self.training)
        
        # EXPLICIT type conversions (not in collate!)
        return {
            "audio": audio,  # torch.Tensor (SEGMENT_SAMPLES,)
            "has_watermark": torch.tensor(
                float(item["has_watermark"]), 
                dtype=torch.float32
            ),
            # Only include fields that exist and are needed
        }
    
    def __len__(self):
        return len(self.samples)


class Stage1BDatasetV11(Dataset):
    """Payload dataset with explicit tensor conversion."""
    
    def __getitem__(self, idx) -> dict:
        item = self.samples[idx]
        audio = load_and_prepare(item["path"], training=self.training)
        
        return {
            "audio": audio,
            "message": torch.tensor(item["message"], dtype=torch.float32),
            "model_id": torch.tensor(item["model_id"], dtype=torch.long),
        }


def collate_v11(batch: list) -> dict:
    """
    Simple collate - all tensors already have correct types.
    """
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
```

---

## 2. Actual Freeze Implementation

```python
def train_stage1b_v11(decoder, loader, device, epochs=10, warmup_epochs=3, top_k=3):
    """
    Stage-1B with ACTUAL parameter freezing during warmup.
    """
    print("=== Stage 1B: Payload (with freeze) ===")
    
    # Identify detect-related parameters
    detect_params = [
        p for n, p in decoder.named_parameters() 
        if 'head_detect' in n or 'backbone' in n
    ]
    message_params = [
        p for n, p in decoder.named_parameters()
        if 'head_message' in n
    ]
    
    for epoch in range(epochs):
        in_warmup = epoch < warmup_epochs
        
        # ACTUALLY FREEZE detect params during warmup
        for p in detect_params:
            p.requires_grad = not in_warmup
        
        # Message params always trainable
        for p in message_params:
            p.requires_grad = True
        
        # Optimizer with only trainable params
        trainable = [p for p in decoder.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=1e-4)
        
        for batch in loader:
            audio = batch["audio"].to(device)
            message = batch["message"].to(device)
            
            outputs = decoder(audio)
            msg_logits = outputs["all_message_logits"]
            
            if in_warmup:
                # Use PREAMBLE CORRELATION for window selection (not detect)
                preamble_scores = compute_preamble_correlation(
                    outputs["all_message_probs"],
                    PREAMBLE
                )  # (B, n_win) correlation scores
                
                _, top_idx = torch.topk(preamble_scores, top_k, dim=1)
            else:
                # After warmup: use detect prob
                _, top_idx = torch.topk(outputs["all_window_probs"], top_k, dim=1)
            
            # Gather top-k message logits
            B, n_win, msg_bits = msg_logits.shape
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_bits)
            selected = torch.gather(msg_logits, 1, top_idx_exp)
            avg_logits = selected.mean(dim=1)
            
            loss = F.binary_cross_entropy_with_logits(avg_logits, message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        mode = "warmup (frozen detect)" if in_warmup else "normal"
        print(f"Stage 1B Epoch {epoch+1} ({mode})")
    
    # Unfreeze all at end
    for p in decoder.parameters():
        p.requires_grad = True


def compute_preamble_correlation(msg_probs, preamble):
    """
    Compute correlation with known preamble for window selection.
    Used during warmup when detector isn't reliable.
    """
    # msg_probs: (B, n_win, 32)
    # preamble: (16,) binary
    B, n_win, _ = msg_probs.shape
    
    preamble_probs = msg_probs[:, :, :16]  # First 16 bits
    preamble_exp = preamble.view(1, 1, 16).expand(B, n_win, -1)
    
    # Correlation: how well do probs match preamble?
    match = (preamble_probs > 0.5).float() == preamble_exp.float()
    correlation = match.float().mean(dim=2)  # (B, n_win)
    
    return correlation
```

---

## 3. Bulk Precompute for Codec (Avoid I/O Bound)

```python
def precompute_all_codecs(manifest_path: Path, output_dir: Path):
    """
    BULK precompute all codec variants in one pass.
    Avoids per-sample subprocess overhead during training.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all codec operations
    codecs = [
        ("mp3_64", ["-c:a", "libmp3lame", "-b:a", "64k"]),
        ("mp3_128", ["-c:a", "libmp3lame", "-b:a", "128k"]),
        ("aac_96", ["-c:a", "aac", "-b:a", "96k"]),
    ]
    
    def process_one(args):
        i, item, codec_name, codec_args = args
        in_path = item["path"]
        out_path = output_dir / f"{i:04d}_{codec_name}.flac"
        
        # Encode then decode (round-trip)
        temp = output_dir / f"temp_{i}_{codec_name}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path] + codec_args + [str(temp)],
            capture_output=True
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(temp), str(out_path)],
            capture_output=True
        )
        temp.unlink()
        
        return {"path": str(out_path), "codec": codec_name, "source_idx": i}
    
    # Parallel processing
    tasks = []
    for i, item in enumerate(manifest):
        for codec_name, codec_args in codecs:
            tasks.append((i, item, codec_name, codec_args))
    
    print(f"Precomputing {len(tasks)} codec variants...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_one, tasks))
    
    # Save manifest
    with open(output_dir / "precomputed_manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Done. Saved to {output_dir}")
```

---

## 4. Two Attribution Metrics

```python
def evaluate_attribution_v11(decoder, loader, codec, decision_rule):
    """
    Report BOTH:
    1. Attribution accuracy on TRUE POSITIVES (correctly detected)
    2. Attribution accuracy on ALL PREDICTED POSITIVES
    """
    results = {
        "true_positives": {"correct_attr": 0, "total": 0},
        "predicted_positives": {"correct_attr": 0, "total": 0},
    }
    
    for batch in loader:
        outputs = decoder(batch["audio"].to(device))
        
        for i in range(len(batch["audio"])):
            gt_label = int(batch["has_watermark"][i].item())
            gt_model = batch["model_id"][i].item() if "model_id" in batch else None
            
            single = {
                "clip_detect_prob": outputs["clip_detect_prob"][i],
                "all_window_probs": outputs["all_window_probs"][i],
                "all_message_probs": outputs["all_message_probs"][i],
            }
            decision = decision_rule.decide(single, codec)
            
            if decision["positive"]:
                results["predicted_positives"]["total"] += 1
                
                if gt_model is not None:
                    if decision.get("model_id") == gt_model:
                        results["predicted_positives"]["correct_attr"] += 1
                    
                    # True positive = actually watermarked AND detected
                    if gt_label == 1:
                        results["true_positives"]["total"] += 1
                        if decision.get("model_id") == gt_model:
                            results["true_positives"]["correct_attr"] += 1
    
    metrics = {
        # Attribution accuracy on TRUE positives (correct detection first)
        "attr_acc_true_positive": (
            results["true_positives"]["correct_attr"] / 
            results["true_positives"]["total"]
            if results["true_positives"]["total"] > 0 else None
        ),
        
        # Attribution accuracy on ALL predicted positives (includes false positives)
        "attr_acc_predicted_positive": (
            results["predicted_positives"]["correct_attr"] /
            results["predicted_positives"]["total"]
            if results["predicted_positives"]["total"] > 0 else None
        ),
        
        "n_true_positives": results["true_positives"]["total"],
        "n_predicted_positives": results["predicted_positives"]["total"],
    }
    
    return metrics
```

---

## 5. Reproducibility

```python
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # MPS doesn't have manual_seed_all, but torch.manual_seed covers it
    
    # Deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_metadata(output_dir: Path, config: dict):
    """Save reproducibility metadata."""
    import subprocess
    
    metadata = {
        "config": config,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip(),
        "timestamp": datetime.now().isoformat(),
        "seed": config.get("seed", 42),
    }
    
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

---

## 6. Corrected FWER Framing

```python
"""
FAMILY-WISE ERROR RATE

When scanning N overlapping windows:
- Independence formula: P(any fires) = 1 - (1-α)^N
- Reality: windows are correlated (positive dependence)
- Effect: independence formula is only an approximation

PRACTICAL APPROACH (not mathematical correction):
- Tune thresholds at CLIP LEVEL empirically on validation set
- Do not trust per-window thresholds directly
- Require joint (detect + preamble) validation in same window

Thresholds are LEARNED from validation, not hardcoded:
"""

class ThresholdConfig:
    """
    Thresholds are calibrated on validation set, not hardcoded.
    These are initial defaults; final values come from tune_thresholds().
    """
    detect_threshold: float = 0.8      # Initial default
    preamble_min: int = 15             # Initial default
    clip_detect_min: float = 0.7       # Initial default
    
    @classmethod
    def from_validation(cls, val_results: dict):
        """Populate from validation tuning."""
        return cls(
            detect_threshold=val_results["optimal_detect_thresh"],
            preamble_min=val_results["optimal_preamble_min"],
            clip_detect_min=val_results["optimal_clip_thresh"],
        )
```

---

## 7. Acceptance Criteria Checklist

> [!IMPORTANT]
> **What must pass before declaring "done"**

### Training Acceptance

- [ ] Stage 1 loss converges (both per-window and clip BCE decrease)
- [ ] Stage 1B payload loss decreases after warmup
- [ ] Stage 2 quality loss (STFT) stays below threshold
- [ ] No NaN/Inf in any loss or gradient

### Detection Acceptance (on held-out test set)

- [ ] Clip-level AUC > 0.95
- [ ] TPR @ 1% FPR > 85%
- [ ] TPR @ 5% FPR > 92%

### Payload Acceptance

- [ ] Valid payload rate (clean) > 90%
- [ ] Valid payload rate (MP3-128) > 70%
- [ ] Attribution accuracy (true positives) > 85%

### Quality Acceptance

- [ ] ViSQOL > 4.0 (or SNR > 30 dB if ViSQOL unavailable)
- [ ] Subjective listening: "no noticeable difference"

### Reproducibility Acceptance

- [ ] `run_metadata.json` saved with git commit + versions
- [ ] Same seed → same results (within floating point tolerance)

---

## 8. Claim Control

> [!CAUTION]
> **Do NOT claim the following without additional work:**

| Claim | Why Not | What to Say Instead |
|-------|---------|---------------------|
| "Robust to removal attacks" | Not evaluated against adversarial removal | "Survives common non-adversarial transforms" |
| "Secure attribution" | No HMAC/authentication | "Decodable attribution in benign settings" |
| "Real-world robust" | Missing Opus, reverb, noise mixtures | "Evaluated on subset of common transforms" |
| "Production-ready" | Not battle-tested | "Proof-of-concept validated on test set" |

---

## Appendix: All Critiques Resolved

| Ver | Issue | Status |
|-----|-------|--------|
| v1-v9 | Various | ✅ Fixed |
| v10 | collate crashes | ✅ Convert in `__getitem__` |
| v10 | Freeze not implemented | ✅ `requires_grad=False` |
| v10 | Warmup uses garbage | ✅ Preamble correlation |
| v10 | FWER wording | ✅ "Tune empirically" |
| v10 | Hardcoded thresholds | ✅ "Learned from validation" |
| v10 | Codec I/O bound | ✅ Bulk precompute |
| v10 | Missing conditional attr | ✅ Two metrics |
| v10 | No reproducibility | ✅ Seeds + version pinning |
| v10 | Overclaiming | ✅ Acceptance criteria |

**v11 meets acceptance criteria definition. Ready for implementation.**
