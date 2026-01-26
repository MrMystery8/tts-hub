"""
Watermark Dataset Module

Handles loading audio and producing multiclass attribution labels.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.1.
"""
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

from watermark.config import SAMPLE_RATE, SEGMENT_SAMPLES
from watermark.config import CLASS_CLEAN, N_MODELS


from watermark.utils.io import load_audio

class WatermarkDataset(Dataset):
    """
    Dataset for watermarking training/eval.
    Now uses robust load_audio for canonical (1, T) format.
    """
    
    def __init__(self, manifest_path: str, *, training: bool = True, n_models: int = N_MODELS):
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        self.training = training
        self.n_models = int(n_models)
    
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
        

        # Metadata parsing (for logging/debug; class label drives training)
        raw_model_id = item.get("model_id", None)
        raw_version = item.get("version", None)
        model_id = int(raw_model_id) if raw_model_id is not None else -1
        version = int(raw_version) if raw_version is not None else -1
        has_watermark = float(item.get("has_watermark", 0))

        # Multiclass attribution label:
        # - class 0: clean / not watermarked
        # - class 1..K: model_id 0..K-1
        #
        # For simplicity and correctness, we require that any watermarked sample has a valid model_id.
        # Version is treated as external metadata (not embedded in the watermark in multiclass mode).
        if has_watermark >= 0.5:
            if model_id < 0:
                raise ValueError(f"manifest has watermarked sample without model_id: idx={idx} path={path}")
            if not (0 <= model_id < self.n_models):
                raise ValueError(f"model_id out of range: idx={idx} model_id={model_id} n_models={self.n_models}")
            y_class = int(model_id) + 1
        else:
            y_class = int(CLASS_CLEAN)

        return {
            "audio": audio,
            "has_watermark": torch.tensor(has_watermark, dtype=torch.float32),
            "model_id": torch.tensor(model_id, dtype=torch.long),
            "version": torch.tensor(version, dtype=torch.long),
            "y_class": torch.tensor(y_class, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack batch items."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
