
import torch
import torch.nn.functional as F
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import soundfile as sf
import torchaudio

from watermark.config import DEVICE, SAMPLE_RATE, MSG_BITS
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2
from watermark.evaluation.attacks import ATTACKS, apply_attack_safe
from watermark.utils.io import load_audio

def load_test_audio(paths, device="cpu"):
    """Load test audio files using robust loader."""
    clips = []
    print(f"Loading {len(paths)} test clips...")
    for i, p in enumerate(paths):
        try:
            # Enforce canonical (1, T) and SR on load
            audio = load_audio(p, target_sr=SAMPLE_RATE)
            
            # SAFETY: Clamp length to 10s max
            MAX_LEN = 160000 
            if audio.shape[-1] > MAX_LEN:
               audio = audio[..., :MAX_LEN]
            
            clips.append(audio.to(device))
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
            
    return clips

def run_mini_benchmark(train_manifest="mini_benchmark_train.json", test_manifest="mini_benchmark_test.json"):
    print(f"Loading manifests...")
    with open(test_manifest) as f:
        test_paths = json.load(f)
        
    print(f"Training on {train_manifest} | Testing on {len(test_paths)} clips")
    
    # === Setup ===
    codec = MessageCodec()
    base_encoder = WatermarkEncoder(msg_bits=32)
    encoder = OverlapAddEncoder(base_encoder).to(DEVICE)
    
    base_decoder = WatermarkDecoder(msg_bits=32)
    decoder = SlidingWindowDecoder(base_decoder).to(DEVICE)
    
    # === Training ===
    # Use reduced settings for mini-benchmark but enough to converge
    EPOCHS = 8 
    print(f"\n=== Training Stages ({EPOCHS} epochs each) ===")
    
    dataset = WatermarkDataset(train_manifest, codec=codec, training=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # Stage 1: Detection
    train_stage1(decoder, encoder, loader, DEVICE, epochs=EPOCHS, log_interval=10)
    
    # Stage 1b: Payload
    preamble = codec.get_preamble()
    train_stage1b(decoder, encoder, loader, DEVICE, preamble, epochs=EPOCHS, log_interval=10)
    
    # Stage 2: Encoder
    train_stage2(encoder, decoder, loader, DEVICE, epochs=EPOCHS, log_interval=10)
    
    print("\n=== Evaluation Phase (CPU) ===")
    # Move to CPU for evaluation stability
    encoder.cpu().eval()
    decoder.cpu().eval()
    
    # Flatten test clips
    test_clips = load_test_audio(test_paths, device="cpu")
    print(f"Loaded {len(test_clips)} valid test clips.")
    
    results = {}
    
    # Removed AAC to prevent torchaudio crashes. Clean, Noise, Resample, Reverb.
    attacks_to_test = ["clean", "noise_white_20db", "resample_8k", "reverb"]
    
    print(f"{'Attack':<20} | {'AUC':<6} | {'TPR@1%':<8} | {'BitAcc':<8} | {'ExactMat':<8}")
    print("-" * 65)
    
    for attack_name in attacks_to_test:
        print(f"Testing {attack_name}...", end="", flush=True)
        if attack_name not in ATTACKS:
             print(f"Skipping {attack_name} (not found)")
             continue
             
        attack_fn = ATTACKS[attack_name]
        
        preds = []     # (score, label)
        bit_accs = []
        exact_matches = 0
        total_pos = 0
        
        for i, clip in enumerate(test_clips):
            # Debug heartbeat
            if i % 20 == 0: print(".", end="", flush=True)
            
            # clip is (1, T) from load_audio
            # C = clip.shape[0] (should be 1)
            
            # 1. Negative Sample (Clean + Attack)
            try:
                # clip is (1, T). apply_attack_safe expects (C, T). Matches.
                neg_audio = apply_attack_safe(clip, attack_fn)
                
                with torch.no_grad():
                    # Decoder expects (B, T) or (B, 1, T).
                    # neg_audio is (1, T). Treat as B=1.
                    out_neg = decoder(neg_audio)
                    score_neg = torch.sigmoid(out_neg["clip_detect_logit"]).item()
                    preds.append((score_neg, 0))
            except Exception as e:
                print(f"[Err Neg {i}: {e}]", end="")
                preds.append((0.0, 0)) # Default to 0

            # 2. Positive Sample (Watermarked + Attack)
            # Generate random message
            msg = torch.randint(0, 2, (1, 32), device="cpu").float()
            
            try:
                with torch.no_grad():
                    # Encoder expects (B, 1, T). clip is (1, T). Unsqueeze(0).
                    wm_audio = encoder(clip.unsqueeze(0), msg)
                    
                    # wm_audio is (1, 1, T). Squeeze to (1, T) for attack.
                    wm_attacked = apply_attack_safe(wm_audio.squeeze(0), attack_fn)
                    
                    out_pos = decoder(wm_attacked)
                    score_pos = torch.sigmoid(out_pos["clip_detect_logit"]).item()
                    preds.append((score_pos, 1))
                    
                    decoded_msg = torch.sigmoid(out_pos["avg_message_logits"]) > 0.5
                    
                    # Exact match
                    is_exact = torch.equal(decoded_msg, msg.bool())
                    
                    # Bit acc
                    acc = (decoded_msg == msg.bool()).float().mean().item()
                    
                    if score_pos > 0.5: # Conditional on detection
                        bit_accs.append(acc)
                        if is_exact:
                            exact_matches += 1
                        total_pos += 1
            except Exception as e:
                print(f"[Err Pos {i}: {e}]", end="")
                preds.append((0.0, 1)) # Default to fail detection if crash

        # Compute Metrics
        print(" Computing...", end="")
        scores, labels = zip(*preds)
        
        try:
             auc = roc_auc_score(labels, scores)
             fpr, tpr, thresholds = roc_curve(labels, scores)
             tpr_at_1 = np.interp(0.01, fpr, tpr)
        except:
             auc = 0.5
             tpr_at_1 = 0.0
        
        # Payload metrics
        avg_bit_acc = np.mean(bit_accs) if len(bit_accs) > 0 else 0.0
        exact_match_rate = exact_matches / total_pos if total_pos > 0 else 0.0
        
        results[attack_name] = {
            "auc": auc,
            "tpr_1": tpr_at_1,
            "bit_acc": avg_bit_acc,
            "exact_match": exact_match_rate
        }
        
        print(f"\r{attack_name:<20} | {auc:.4f} | {tpr_at_1:.4f}   | {avg_bit_acc:.4f}   | {exact_match_rate:.4f}")
        
    return results

if __name__ == "__main__":
    if not Path("mini_benchmark_train.json").exists():
        print("Manifests not found. Run download_data.py first.")
    else:
        run_mini_benchmark()
