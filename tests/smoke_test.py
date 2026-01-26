
import torch
import unittest
import numpy as np
import tempfile
import os
import soundfile as sf
from pathlib import Path

from watermark.utils.io import load_audio, save_audio
from watermark.evaluation.attacks import ATTACKS, apply_attack_safe
from watermark.config import SEGMENT_SAMPLES, SAMPLE_RATE

class TestEngineeringContracts(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic 1s sine wave @ 16kHz
        self.sr = 16000
        t = np.linspace(0, 1, self.sr)
        self.audio_np = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Canonical Tensor (1, T)
        self.audio_tensor = torch.from_numpy(self.audio_np).unsqueeze(0)
        
    def test_io_contract(self):
        """Test strict I/O shape contract (1, T)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            
            # Save using our util (accepts tensor)
            save_audio(str(path), self.audio_tensor, self.sr)
            
            # Load back
            loaded = load_audio(str(path), target_sr=self.sr)
            
            # Checks
            self.assertEqual(loaded.dim(), 2, "Loaded audio must match (1, T) dim")
            self.assertEqual(loaded.shape[0], 1, "Loaded audio must be Mono (1, T)")
            self.assertEqual(loaded.shape[1], self.sr, "Length mismatch")
            self.assertTrue(torch.is_tensor(loaded), "Must return Tensor")
            self.assertEqual(loaded.dtype, torch.float32, "Must be float32")
            
    def test_io_resampling(self):
        """Test loader handles resampling correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_44k.wav"
            
            # Create proper 44.1k data (1s)
            sr_orig = 44100
            t = np.linspace(0, 1, sr_orig)
            audio_44k = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            
            # Save as 44.1k using soundfile directly
            sf.write(str(path), audio_44k, sr_orig)
            
            # Load with target 16k
            loaded = load_audio(str(path), target_sr=16000)
            
            # Checks
            self.assertEqual(loaded.shape[1], 16000, f"Resampling failed length check: {loaded.shape}")
            self.assertAlmostEqual(loaded.max().item(), 0.5, delta=0.1, msg="Amplitude preserved")

    def test_attacks_contract(self):
        """Test ALL attacks preserve shape (1, T) and finite values."""
        
        print(f"\nTesting {len(ATTACKS)} attacks...")
        
        for name, attack_fn in ATTACKS.items():
            with self.subTest(attack=name):
                print(f"  -> Testing {name}")
                
                # Apply Safe Wrapper
                out = apply_attack_safe(self.audio_tensor, attack_fn)
                
                # Strict Assertions
                self.assertEqual(out.shape, self.audio_tensor.shape, 
                                 f"Attack {name} violated shape contract")
                self.assertTrue(torch.isfinite(out).all(), 
                                f"Attack {name} produced NaNs/Infs")
                self.assertEqual(out.device, self.audio_tensor.device,
                                 f"Attack {name} changed device")

    def test_encoder_contract(self):
        """Test OverlapAddEncoder preserves strict length (no rounding errors)."""
        from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
        from watermark.config import N_CLASSES
        
        base = WatermarkEncoder(num_classes=N_CLASSES)
        encoder = OverlapAddEncoder(base)
        
        # Test various lengths, including primes and awkward sizes
        lengths = [16000, 16001, 22050, 48000, 100] 
        class_id = torch.randint(0, N_CLASSES, (1,), dtype=torch.long)
        
        print("\nTesting Encoder Lengths...")
        for L in lengths:
            # (1, 1, L)
            audio = torch.randn(1, 1, L)
            with torch.no_grad():
                out = encoder(audio, class_id)
            
            self.assertEqual(out.shape[-1], L, f"Encoder changed length from {L} to {out.shape[-1]}")
            print(f"  -> L={L} Passed")

if __name__ == '__main__':
    unittest.main()
