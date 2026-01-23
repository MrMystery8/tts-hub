"""
Watermark Dataset Module

Handles loading audio, on-the-fly watermarking, and message generation.
Implementation follows WATERMARK_PROJECT_PLAN.md v16, section 5.1.
"""
import json
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING
import random

from watermark.config import SAMPLE_RATE, SEGMENT_SAMPLES

if TYPE_CHECKING:
    from watermark.models.codec import MessageCodec


class WatermarkDataset(Dataset):
    """
    Dataset for watermarking training/eval.
    
    Features:
    - Fixed length segments (3s)
    - Random crop (train) / Center crop (eval)
    - On-the-fly message generation via Codec
    - Resampling to target SAMPLE_RATE
    """
    
    def __init__(self, manifest_path: str, codec: 'MessageCodec', training: bool = True):
        """
        Args:
            manifest_path: Path to JSON manifest file.
            codec: MessageCodec instance for message generation.
            training: If True, uses random cropping. Else, uses center cropping.
        """
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        self.codec = codec
        self.training = training
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        # Load audio (mono)
        try:
            # Try loading with soundfile backend explicitly to avoid torchcodec issues
            try:
                audio, sr = torchaudio.load(item["path"], backend="soundfile")
            except Exception:
                 # Fallback to soundfile direct load if backend arg fails
                import soundfile as sf
                data, sr = sf.read(item["path"])
                audio = torch.from_numpy(data).float()
                if audio.dim() == 2:
                    audio = audio.t() # (channels, time)
                else:
                    audio = audio.unsqueeze(0) # (1, time)

            audio = audio.mean(dim=0)  # Mono
        except Exception as e:
            # Fallback for missing/corrupt files (should filter before training, but safety here)
            # Create silence if file fails
            print(f"Error loading {item.get('path')}: {e}")
            audio = torch.zeros(SEGMENT_SAMPLES)
            sr = SAMPLE_RATE
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        
        # Crop/Pad to fixed length
        T = audio.shape[0]
        if T >= SEGMENT_SAMPLES:
            if self.training:
                # Random crop
                start = torch.randint(0, T - SEGMENT_SAMPLES + 1, (1,)).item()
            else:
                # Center crop
                start = (T - SEGMENT_SAMPLES) // 2
            audio = audio[start : start + SEGMENT_SAMPLES]
        else:
            # Pad
            audio = F.pad(audio, (0, SEGMENT_SAMPLES - T))
        
        # Metadata parsing
        model_id = int(item.get("model_id", 0)) if item.get("model_id") is not None else 0
        version = int(item.get("version", 1)) if item.get("version") is not None else 1
        has_watermark = float(item.get("has_watermark", 0))
        
        # Generate MESSAGE tensor on-the-fly
        # Even if clean (has_watermark=0), we generate a message so tensors have constant shape.
        # Loss will ignore message for negative samples if implemented correctly.
        message = self.codec.encode(model_id, version).float()
        
        return {
            "audio": audio,
            "has_watermark": torch.tensor(has_watermark, dtype=torch.float32),
            "model_id": torch.tensor(model_id, dtype=torch.long),
            "version": torch.tensor(version, dtype=torch.long),
            "message": message 
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack batch items."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
