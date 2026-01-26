"""
Unit tests for Training Infrastructure
"""
import torch
import pytest
import tempfile
import json
import os
from pathlib import Path
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.losses import CachedSTFTLoss


class TestWatermarkDataset:
    """Tests for WatermarkDataset."""
    
    def test_dataset_loading(self):
        """Should load audio and generate multiclass label."""
        # Create temp audio and manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            audio_path = tmpdir / "test.wav"
            import torchaudio
            # torchaudio.save defaults to torchcodec in newer versions which fails if not installed
            # We explicitly use soundfile backend or just use soundfile directly if needed
            # But just having soundfile installed might be enough for backend="soundfile"
            try:
                torchaudio.save(audio_path, torch.randn(1, 16000), 16000, backend="soundfile")
            except Exception:
                # Fallback if backend arg not supported (older torchaudio) 
                # or if it still fails.
                import soundfile as sf
                sf.write(audio_path, torch.randn(16000).numpy(), 16000)
            
            manifest = [{
                "path": str(audio_path),
                "has_watermark": 1,
                "model_id": 3,
                "version": 2
            }]
            manifest_path = tmpdir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            dataset = WatermarkDataset(str(manifest_path), training=True)
            
            item = dataset[0]
            assert "audio" in item
            # Canonical audio contract is (C, T) where C=1 (mono).
            assert item["audio"].shape == (1, 48000)  # Default segment samples
            assert item["has_watermark"].item() == 1.0
            assert item["model_id"].item() == 3
            assert item["version"].item() == 2
            assert item["y_class"].item() == 4  # class = model_id + 1

    def test_collate_fn(self):
        """Should stack items correctly."""
        batch = [
            {"a": torch.tensor([1]), "b": torch.tensor([2])},
            {"a": torch.tensor([3]), "b": torch.tensor([4])},
        ]
        collated = collate_fn(batch)
        assert collated["a"].shape == (2, 1)
        assert collated["b"].shape == (2, 1)


class TestCachedSTFTLoss:
    """Tests for CachedSTFTLoss."""
    
    def test_forward_pass(self):
        """Should compute positive loss."""
        loss_fn = CachedSTFTLoss(fft_sizes=[512])
        x = torch.randn(2, 16000)
        y = x + 0.01 * torch.randn(2, 16000)
        
        loss = loss_fn(x, y)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_buffers_registered(self):
        """Should have window buffers."""
        loss_fn = CachedSTFTLoss(fft_sizes=[512, 1024])
        assert hasattr(loss_fn, 'window_512')
        assert hasattr(loss_fn, 'window_1024')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
