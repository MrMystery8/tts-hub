"""
Strict Verification Scripts
1. Budget Stress Test: Force high alpha to ensure EnergyBudgetLoss triggers.
2. Two-head decoder plumbing test: Encoder/Decoder forward shapes.
"""
import torch
from watermark.training.losses import EnergyBudgetLoss
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.config import DEVICE, N_CLASSES

def test_budget():
    print("\n[Test] Energy Budget Loss Stress Test")
    
    # 1. Setup Loss
    target_db = -30.0
    loss_fn = EnergyBudgetLoss(target_db=target_db, limit_type="hard")
    
    # 2. Create Dummy Audio
    # (B, 1, T)
    B, T = 4, 16000
    audio = torch.randn(B, 1, T).to(DEVICE)
    power_orig = audio.pow(2).mean()
    
    print(f"Original Power: {power_orig.item():.6f}")
    
    # 3. Create "Violating" Watermark
    # Make perturbation huge (e.g. 0dB relative to signal, or even louder)
    # Target is -30dB (~0.001 power ratio).
    # If we add noise with amplitude 0.5 * orig, power ratio is 0.25 (-6dB).
    # This SHOULD trigger loss.
    
    noise = torch.randn_like(audio) * 0.5 * audio.std() # ~ -6dB
    wm_violating = audio + noise
    
    loss_val = loss_fn(audio, wm_violating)
    
    # Calculate measured dB
    diff = wm_violating - audio
    p_wm = diff.pow(2).mean()
    p_orig = audio.pow(2).mean()
    ratio = p_wm / p_orig
    measured_db = 10 * torch.log10(ratio)
    
    print(f"Violating Case:")
    print(f"  Measured Ratio: {ratio.item():.6f}")
    print(f"  Measured dB:    {measured_db.item():.2f} dB")
    print(f"  Loss Value:     {loss_val.item():.6f}")
    
    if loss_val.item() > 0.0:
        print(">> PASSED: Loss activated for violating input.")
    else:
        print(">> FAILED: Loss did not activate!")

    # 4. Create "Compliant" Watermark
    # -40dB noise
    scale = 10**(-40/20) # Amplitude scale
    noise_compliant = torch.randn_like(audio) * scale * audio.std()
    wm_compliant = audio + noise_compliant
    
    loss_compliant = loss_fn(audio, wm_compliant)
    
    diff_c = wm_compliant - audio
    measured_db_c = 10 * torch.log10(diff_c.pow(2).mean() / audio.pow(2).mean())
    
    print(f"Compliant Case (-40dB target):")
    print(f"  Measured dB:    {measured_db_c.item():.2f} dB")
    print(f"  Loss Value:     {loss_compliant.item():.6f}")
    
    if loss_compliant.item() == 0.0:
        print(">> PASSED: Loss is zero for compliant input.")
    else:
        print(">> FAILED: Loss activated for compliant input!")


def test_two_head_plumbing():
    print("\n[Test] Two-head Decoder Plumbing")

    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES)).to(DEVICE)

    # 3s audio @ 16k
    audio = torch.randn(2, 1, 48000, device=DEVICE)
    y_class = torch.tensor([1, 2], device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        wm = encoder(audio, y_class)
        out = decoder(wm)

    assert out["clip_class_logits"].shape == (2, N_CLASSES)
    assert out["clip_class_probs"].shape == (2, N_CLASSES)
    assert out["clip_id_logits"].shape == (2, N_CLASSES - 1)
    assert out["clip_id_probs"].shape == (2, N_CLASSES - 1)
    assert out["clip_wm_prob"].shape == (2,)
    print(">> PASSED: Shapes OK.")

if __name__ == "__main__":
    test_budget()
    test_two_head_plumbing()
