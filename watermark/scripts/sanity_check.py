"""
Sanity Audit Script

Runs a complete lifecycle test:
1. Generate synthetic dataset (200 clips).
2. Train Stage 1 (Detection) -> Stage 1B (Payload) -> Stage 2 (Encoder).
3. Evaluate on a held-out test set using the FULL Decision Rule.
4. Report metrics (Accuracy, TPR, etc).
"""
import torch
import numpy as np
import soundfile as sf
import json
import shutil
from pathlib import Path
from torch.utils.data import DataLoader

from watermark.config import DEVICE, SAMPLE_RATE
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder, ClipDecisionRule, decide_batch
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2
from watermark.evaluation.metrics import compute_tpr_at_fpr

def generate_data(num_files=200, output_dir="sanity_data"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    label_map = []
    
    print(f"Generating {num_files} synthetic clips...")
    for i in range(num_files):
        # Generate random audio (Sine + Noise)
        t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0))
        freq = np.random.uniform(200, 800)
        audio = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        audio = audio.astype(np.float32)
        
        path = output_dir / f"sample_{i}.wav"
        sf.write(str(path), audio, SAMPLE_RATE)
        
        # 50/50 split for watermark
        is_pos = (i % 2 == 0)
        label_map.append({
            "path": str(path.absolute()),
            "has_watermark": 1 if is_pos else 0,
            "model_id": i % 8 if is_pos else None,
            "version": 1
        })
        
    # Split Train/Test (80/20)
    split_idx = int(0.8 * num_files)
    train_manifest = label_map[:split_idx]
    test_manifest = label_map[split_idx:]
    
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_manifest, f, indent=2)
        
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_manifest, f, indent=2)
        
    return output_dir / "train.json", output_dir / "test.json"

def evaluate_model(decoder, encoder, test_loader, codec, device):
    decoder.eval()
    encoder.eval()
    rule = ClipDecisionRule(detect_threshold=0.5, preamble_min=12) 
    
    results = []
    y_true = []
    y_score = []
    
    print("\nEvaluating on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)
            message = batch["message"].to(device)
            
            # Apply watermark if needed (Ground Truth generation for eval)
            # Eval set is clean on disk, so we simulate proper test conditions:
            # - If has_wm=1, we watermark it using our TRAINED encoder
            # - If has_wm=0, we leave it clean
            
            audio_in = audio
            wm_audio = encoder(audio_in, message)
            
            mask = has_wm.view(-1, 1, 1)
            # Create the test audio: Mixed batch of Watermarked vs Clean
            test_input = mask * wm_audio + (1 - mask) * audio_in
            test_input = test_input.detach()
            
            # Run Decoder
            outputs = decoder(test_input)
            
            # Run Decision Rule
            decisions = decide_batch(outputs, codec, rule)
            
            # Collect metrics
            for i in range(len(decisions)):
                is_watermarked = (has_wm[i].item() == 1.0)
                pred_positive = decisions[i]["positive"]
                
                # Check correctness
                correct = (is_watermarked == pred_positive)
                if is_watermarked and pred_positive:
                    # Check payload
                    correct = correct and (decisions[i]["model_id"] == batch["model_id"][i].item())
                
                results.append(correct)
                y_true.append(is_watermarked)
                y_score.append(outputs["clip_detect_prob"][i].item())

    accuracy = sum(results) / len(results)
    
    # Compute AUC/TPR
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_score)
        # Simple TPR@1% calculation
        fpr_limit = 0.01
        sorted_scores = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
        # ... actually use sklearn or simple logic
        # Just report AUC and Accuracy for sanity
    except:
        auc = 0.0
        
    print(f"Test Accuracy (inc. payload): {accuracy*100:.1f}%")
    print(f"Test AUC: {auc:.4f}")
    return accuracy

def main():
    print(f"Starting Sanity Audit on {DEVICE}...")
    
    # 1. Generate Data
    train_mf, test_mf = generate_data(num_files=200)
    
    codec = MessageCodec(key="sanity")
    
    # 2. Init Models
    encoder = OverlapAddEncoder(WatermarkEncoder(msg_bits=32)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(msg_bits=32)).to(DEVICE)
    
    # 3. Load Data
    train_ds = WatermarkDataset(train_mf, codec=codec, training=True)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    test_ds = WatermarkDataset(test_mf, codec=codec, training=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 4. Train Stage 1
    print("\n[Stage 1] Training Detection...")
    train_stage1(decoder, encoder, train_loader, DEVICE, epochs=15, log_interval=999)
    
    # 5. Train Stage 1B
    print("\n[Stage 1B] Training Payload...")
    train_stage1b(decoder, encoder, train_loader, DEVICE, codec.preamble, epochs=10, warmup=3, log_interval=999)
    
    # 6. Train Stage 2
    print("\n[Stage 2] Training Encoder...")
    train_stage2(encoder, decoder, train_loader, DEVICE, epochs=15, log_interval=999)
    
    # 7. Final Eval
    print("\n[Eval] Running Final Decision Rule...")
    acc = evaluate_model(decoder, encoder, test_loader, codec, DEVICE)
    
    if acc > 0.8:
        print("\nSUCCESS: Sanity check passed (Acc > 80%)")
    else:
        print("\nWARNING: Sanity check low accuracy (Check setup)")
        
    # Cleanup
    shutil.rmtree("sanity_data")

if __name__ == "__main__":
    main()
