
import torch
import torch.nn.functional as F
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.config import DEVICE, N_CLASSES

def main():
    print(f"Sanity Check A: Conditioning Realism")
    print(f"Device: {DEVICE}")
    print(f"N_CLASSES: {N_CLASSES}")
    
    # 1. Init model
    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES)).to(DEVICE)
    encoder.eval()
    
    # 2. Fake Audio
    # (1, 1, 48000) - 3 seconds at 16khz
    B, C, T = 1, 1, 48000
    audio = torch.randn(B, C, T).to(DEVICE)
    
    # 3. Two different classes
    class_1 = torch.tensor([1], device=DEVICE)
    class_2 = torch.tensor([2], device=DEVICE)
    
    print("Running inference class=1...")
    with torch.no_grad():
        out1 = encoder(audio, class_1)
        
    print("Running inference class=2...")
    with torch.no_grad():
        out2 = encoder(audio, class_2)
        
    # 4. Compute delta
    diff = (out1 - out2).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"\n--- Results ---")
    print(f"Mean |out1 - out2|: {mean_diff:.8f}")
    print(f"Max  |out1 - out2|: {max_diff:.8f}")
    
    if mean_diff < 1e-9:
        print("\n[FAIL] CRITICAL: Encoder output is identical for different classes!")
    else:
        print("\n[PASS] Encoder produces distinct watermarks for different classes.")
        
if __name__ == "__main__":
    main()
