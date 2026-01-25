"""
Watermark Dataset Module

Handles loading audio, on-the-fly watermarking, and message generation.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.1.
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
from watermark.config import N_MODELS, N_VERSIONS

if TYPE_CHECKING:
    from watermark.models.codec import MessageCodec


from watermark.utils.io import load_audio

class WatermarkDataset(Dataset):
    """
    Dataset for watermarking training/eval.
    Now uses robust load_audio for canonical (1, T) format.
    """
    
    def __init__(self, manifest_path: str, codec: 'MessageCodec', training: bool = True):
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        self.codec = codec
        self.training = training
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        path = item["path"]
        
        try:
            # Canonical load: (1, T) @ 16k
            audio = load_audio(path, target_sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            audio = torch.zeros(1, SEGMENT_SAMPLES)
        
        # Audio is (1, T)
        # Crop/Pad to fixed length
        T = audio.shape[-1]
        
        if T >= SEGMENT_SAMPLES:
            if self.training:
                # Random crop
                start = torch.randint(0, T - SEGMENT_SAMPLES + 1, (1,)).item()
            else:
                # Center crop
                start = (T - SEGMENT_SAMPLES) // 2
            audio = audio[..., start : start + SEGMENT_SAMPLES]
        else:
            # Pad at end
            audio = F.pad(audio, (0, SEGMENT_SAMPLES - T))
        

        
        # Metadata parsing
        raw_model_id = item.get("model_id", None)
        raw_version = item.get("version", None)
        model_id = int(raw_model_id) if raw_model_id is not None else -1
        version = int(raw_version) if raw_version is not None else -1
        has_watermark = float(item.get("has_watermark", 0))
        has_labels = (model_id >= 0) and (version >= 0)
        
        # Generate MESSAGE tensor on-the-fly
        # Even if clean (has_watermark=0), we generate a message so tensors have constant shape.
        # Loss will ignore message for negative samples if implemented correctly.
        if has_labels:
            message = self.codec.encode(model_id, version).float()
        else:
            # IMPORTANT (attribution footgun):
            # If a manifest contains unlabeled positives and we always map them to a constant
            # default (0,0), any training path that accidentally supervises on these targets
            # can collapse into "presence-only + constant identity".
            #
            # Keep message shape fixed, but avoid a constant payload for unlabeled watermarked
            # items by sampling a balanced random ID pair.
            if has_watermark >= 0.5:
                rid = random.randrange(int(N_MODELS))
                rver = random.randrange(int(N_VERSIONS))
                message = self.codec.encode(rid, rver).float()
            else:
                # For clean negatives this value is unused (decoder sees clean audio), but we
                # still return a tensor for consistent collate.
                message = self.codec.encode(0, 0).float()
        
        return {
            "audio": audio,
            "has_watermark": torch.tensor(has_watermark, dtype=torch.float32),
            "model_id": torch.tensor(model_id, dtype=torch.long),
            "version": torch.tensor(version, dtype=torch.long),
            "has_labels": torch.tensor(has_labels, dtype=torch.bool),
            "message": message 
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack batch items."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
