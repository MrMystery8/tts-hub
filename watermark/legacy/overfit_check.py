"""
Overfit Check Script

Goal: Force the model to learn detection (AUC > 0.95) on a tiny dataset.
Strategy:
1. Small Dataset: 50 clips of PINK NOISE (easier than sine).
2. Relaxed Constraints: Quality loss weight = 0.0 (temporarily).
3. Stronger Embed: Increase alpha range.
4. Monitoring: Log TRAINING AUC per epoch to verify convergence.
5. Verification: Check if AUC uses probabilities.

LEGACY NOTE:
This script targets the old bit-payload pipeline and is not maintained under multiclass attribution.
"""
import torch
import numpy as np
import soundfile as sf
import json
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from watermark.config import DEVICE, SAMPLE_RATE
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage2 import train_stage2
# Stage 1B less critical for detection debug

def generate_pink_noise(samples):
    """Generate pink noise (1/f) for better masking potential."""
    uneven = samples % 2
    X = np.random.randn(samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
    S = np.arange(len(X)) + 1  # Filter
    y = (np.fft.irfft(X / S)).real
    if uneven: y = y[:-1]
    return y.astype(np.float32) * 0.1

def generate_overfit_data(num_files=50, output_dir="overfit_data"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    label_map = []
    print(f"Generating {num_files} PINK NOISE clips (easier masking)...")
    
    for i in range(num_files):
        audio = generate_pink_noise(int(SAMPLE_RATE * 3.0))
        path = output_dir / f"sample_{i}.wav"
        sf.write(str(path), audio, SAMPLE_RATE)
        
        is_pos = (i % 2 == 0)
        label_map.append({
            "path": str(path.absolute()),
            "has_watermark": 1 if is_pos else 0,
            "model_id": i % 8 if is_pos else None,
            "version": 1
        })
        
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(label_map, f, indent=2)
    return output_dir / "manifest.json"

def debug_stage2(encoder, decoder, loader, device, epochs=50):
    """
    Modified Stage 2 for OVERFITTING:
    - Quality Weight = 0.0 (Unconstrained!)
    - Log Training AUC every epoch
    """
    print(f"\n[Overfit Stage 2] Training Encoder (Quality Weight = 0.0)...")
    
    # Unfreeze encoder, Freeze decoder
    encoder.train()
    decoder.eval() # Use fixed decoder logic (or pre-trained?)
    # Wait, normally Stage 2 assumes a TRAINED decoder. 
    # But if Stage 1 failed (AUC 0.5), we have a garbage decoder.
    # We must train JOINTLY or Iteratively for this overfit test?
    # Actually, Stage 1 trains Decoder on Fixed Encoder. 
    # Stage 2 trains Encoder on Fixed Decoder.
    # If Encoder is weak (init), Stage 1 learns weak signal.
    # If Decoder is weak (init/S1 fail), Stage 2 can't learn.
    # RECOMMENDATION: Train Stage 2 with Decoder UNFrozen? Or Alternating?
    # For simplicity, let's stick to pipeline: S1 -> S2 -> S1... but here just S1 (Decoder) implies 'Fixed Encoder'.
    # If Fixed Encoder (random init) outputs visible artifacts, S1 SHOULD learn.
    
    return

def debug_stage1(decoder, encoder, loader, device, epochs=50):
    """
    Modified Stage 1 for OVERFITTING:
    - Track AUC
    """
    print(f"\n[Overfit Stage 1] Training Detection (Decoder) on Random Encoder...")
    decoder.train()
    encoder.eval() 
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3) # Higher LR
    
    for epoch in range(epochs):
        all_probs = []
        all_targets = []
        epoch_loss = 0
        
        for batch in loader:
            audio = batch["audio"].to(device)
            has_wm = batch["has_watermark"].to(device)
            message = batch["message"].to(device)
            
            # Embed with STRONGER alpha (hack: modify encoder output?)
            # Encoder has learned alpha? It starts at 0.02.
            # Let's Force it scaling manually for this test if needed.
            # But verifying 'random init' learnability first.
            
            with torch.no_grad():
                wm_audio = encoder(audio, message)
                # FORCE HIGH STRENGTH for sanity check?
                # The 'alpha' inside encoder handles strength. Random init alpha ~0.02.
                # If we assume random weights + 0.02 is enough signal?
                # Maybe 0.02 is too small for random weights to show through SINE/PinkNoise?
                pass
            
            mask = has_wm.view(-1, 1, 1)
            inp = (mask * wm_audio + (1 - mask) * audio).detach()
            
            outputs = decoder(inp)
            logits = outputs["clip_detect_logit"]
            probs = outputs["clip_detect_prob"]
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, has_wm)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            all_probs.extend(probs.detach().cpu().numpy())
            all_targets.extend(has_wm.cpu().numpy())
            
        # Compute AUC
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.5
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | AUC: {auc:.4f}")
            
        # Early stop if converged
        if auc > 0.99:
            print(f"Converged at Epoch {epoch+1}! AUC: {auc:.4f}")
            return True
            
    return False

def main():
    print(f"Starting OVERFIT CHECK on {DEVICE}...")
    
    # 1. Gen Pink Noise
    manifest_path = generate_overfit_data(num_files=50) # 50 files
    codec = MessageCodec(key="overfit")
    
    # 2. Init Models
    # Increase base strength of encoder manually?
    # We can patch 'alpha' after init.
    enc_base = WatermarkEncoder(msg_bits=32)
    with torch.no_grad():
        enc_base.alpha.fill_(0.1) # FORCE 0.1 (Max allowed in plan)
        print("Forcing Encoder Alpha to 0.1 for high visibility.")
        
    encoder = OverlapAddEncoder(enc_base).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(msg_bits=32)).to(DEVICE)
    
    ds = WatermarkDataset(manifest_path, codec=codec, training=True)
    loader = DataLoader(ds, batch_size=10, shuffle=True, collate_fn=collate_fn)
    
    # 3. Train Stage 1 (Decoder Only)
    # If Detection works, AUC should hit 1.0 quickly on 50 noise files.
    success = debug_stage1(decoder, encoder, loader, DEVICE, epochs=50)
    
    if not success:
        print("\nFAILURE: Stage 1 did not overfit even with Alpha=0.1 on Pink Noise.")
        print("Possible causes: Decoder architecture too weak? Inputs normalized wrong?")
        return
        
    print("\nSUCCESS: Stage 1 Overfitted (Decoder can see the watermark).")
    
    # 4. Train Stage 2 (Encoder Only) - Removing Quality Handcuffs
    # Now we train encoder to OPTIMIZE availability using the trained decoder.
    # We set quality loss weight to 0 to see if it learns to cheat/maximize detection.
    
    print(f"\n[Overfit Stage 2] Training Encoder (Qual=0.0)...")
    encoder.train()
    decoder.eval()
    opt_enc = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    
    from watermark.training.losses import CachedSTFTLoss
    stft_loss = CachedSTFTLoss().to(DEVICE)
    
    for epoch in range(50):
        all_probs = []
        all_targets = []
        
        for batch in loader:
            audio = batch["audio"].to(DEVICE)
            message = batch["message"].to(DEVICE)
            
            # Embed
            wm = encoder(audio, message)
            
            # Decode
            outputs = decoder(wm.squeeze(1))
            
            # Loss: MAXIMIZE detection only (ignore quality)
            # Target = 1.0
            det_logits = outputs["clip_detect_logit"]
            loss_det = torch.nn.functional.binary_cross_entropy_with_logits(
                det_logits, torch.ones_like(det_logits)
            )
            
            # AUX LOSS (All windows)
            loss_aux = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["all_window_logits"],
                torch.ones_like(outputs["all_window_logits"])
            )
            
            loss = loss_det + 0.5 * loss_aux
            
            opt_enc.zero_grad()
            loss.backward()
            opt_enc.step()
            
            all_probs.extend(outputs["clip_detect_prob"].detach().cpu().numpy())
            # Targets are all 1s here for statistics
            
        avg_prob = sum(all_probs)/len(all_probs)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1} | Det Loss: {loss_det.item():.4f} | Avg Prob: {avg_prob:.4f}")
            
    print(f"Final Encoder Avg Prob: {avg_prob:.4f}")
    if avg_prob > 0.9:
        print("SUCCESS: Stage 2 learned to trigger decoder.")
    else:
        print("WARNING: Stage 2 struggled to trigger decoder.")

    # Cleanup
    shutil.rmtree("overfit_data")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
