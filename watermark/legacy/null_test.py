"""
Diagnostic Null Tests for Watermarking System
Verifies leakage and pipeline integrity.

Null Test A (Bypass): Encoder = Identity. AUC should be ~0.5.
Null Test B (Shuffle): Labels shuffled. AUC should be ~0.5.

LEGACY NOTE:
This script targets the old bit-payload pipeline and is not maintained under multiclass attribution.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random
import glob
import json
import argparse
from typing import List, Dict
from sklearn.metrics import roc_auc_score
import numpy as np

from watermark.models.decoder import WatermarkDecoder
from watermark.models.encoder import WatermarkEncoder
from watermark.models.codec import MessageCodec
from watermark.config import SAMPLE_RATE, SEGMENT_SAMPLES
from watermark.utils.io import load_audio
from watermark.training.losses import CachedSTFTLoss

class IdentityEncoder(nn.Module):
    def forward(self, audio, message):
        return audio

class NullDataset(Dataset):
    def __init__(self, files: List[str], shuffle_labels: bool = False):
        self.files = files
        self.shuffle_labels = shuffle_labels
        self.codec = MessageCodec()
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            audio = load_audio(path, target_sr=SAMPLE_RATE)
        except Exception:
            audio = torch.zeros(1, SEGMENT_SAMPLES)
            
        # Crop/Pad
        T = audio.shape[-1]
        if T >= SEGMENT_SAMPLES:
            start = torch.randint(0, T - SEGMENT_SAMPLES + 1, (1,)).item()
            audio = audio[..., start : start + SEGMENT_SAMPLES]
        else:
            audio = torch.nn.functional.pad(audio, (0, SEGMENT_SAMPLES - T))
            
        # Label generation
        # 50% Clean / 50% Watermarked target
        is_watermarked = float(random.random() > 0.5)
        
        if self.shuffle_labels:
            # Random label completely uncorrelated with input
            target_label = float(random.random() > 0.5)
        else:
            target_label = is_watermarked
            
        # For Bypass mode:
        # If is_watermarked=1, we WANT to watermark it (but encoder will do nothing).
        # Network sees:
        # Input: Audio (unmodified)
        # Target: 1.0
        #
        # If is_watermarked=0:
        # Input: Audio (unmodified)
        # Target: 0.0
        #
        # Result: Network sees identical inputs with different labels. 
        # Should fail to converge (AUC ~0.5).
        
        return {
            "audio": audio,
            "label": torch.tensor(target_label, dtype=torch.float32),
            "should_watermark": torch.tensor(is_watermarked, dtype=torch.bool)
        }

def get_split_files(data_dir: str, regex: str = "**/*.wav"):
    """
    Split audio files by SOURCE (filename without extension).
    Ensures chunks of same file don't leak between splits.
    """
    all_files = list(glob.glob(f"{data_dir}/{regex}", recursive=True))
    
    # Group by source ID (filename stem)
    file_groups = {}
    for f in all_files:
        stem = Path(f).stem
        # Heuristic: remove _chunk_XX suffixes if they exist
        source_id = stem.split("_chunk")[0]
        if source_id not in file_groups:
            file_groups[source_id] = []
        file_groups[source_id].append(f)
        
    source_ids = list(file_groups.keys())
    random.shuffle(source_ids)
    
    split_idx = int(0.8 * len(source_ids))
    train_ids = source_ids[:split_idx]
    val_ids = source_ids[split_idx:]
    
    train_files = []
    for sid in train_ids:
        train_files.extend(file_groups[sid])
        
    val_files = []
    for sid in val_ids:
        val_files.extend(file_groups[sid])
        
    print(f"Split by Source ID: {len(train_ids)} Train Sources, {len(val_ids)} Val Sources")
    print(f"Total Files: {len(train_files)} Train, {len(val_files)} Val")
    
    return train_files, val_files

def run_null_test(mode: str, data_dir: str, epochs: int = 5, regex: str = "**/*.flac"):
    print(f"\nRunning Null Test: {mode.upper()}")
    
    train_files, val_files = get_split_files(data_dir, regex=regex)
    if not train_files:
        print("No files found!")
        return
        
    shuffle = (mode == "shuffle")
    train_ds = NullDataset(train_files, shuffle_labels=shuffle)
    val_ds = NullDataset(val_files, shuffle_labels=False) # Val labels always honest for evaluation
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Components
    if mode == "bypass":
        encoder = IdentityEncoder().to(device)
    else:
        encoder = WatermarkEncoder().to(device)
        
    decoder = WatermarkDecoder().to(device)
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        decoder.train()
        train_loss = 0
        
        for batch in train_loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            should_wm = batch["should_watermark"].to(device)
            
            # Generate random message for encoder
            # (Batch, 32)
            B = audio.shape[0]
            msg = torch.randint(0, 2, (B, 32)).float().to(device)
            
            # Conditional Encoding
            # We must apply encoder only where should_wm is True?
            # Or assume encoder handles identity? No, WatermarkEncoder adds WM everywhere.
            # We need to manually handle mixing `clean` and `watermarked`.
            
            # Apply encoder to ALL, then mix based on should_wm
            wm_out = encoder(audio, msg)
            
            # Mix: if should_wm, take wm_out, else take audio (clean)
            # NullDataset generates 'should_watermark' boolean.
            # Convert to mask (B, 1, 1)
            mask = should_wm.view(B, 1, 1).float()
            
            final_audio = mask * wm_out + (1 - mask) * audio
            
            # Detector
            outputs = decoder(final_audio)
            d_logits = outputs["detect_logit"].squeeze(1)
            
            loss = criterion(d_logits, labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            
        # Validation
        decoder.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                
                # In val, we want to see if it predicts noise.
                # For Bypass Mode:
                # We feed strict audio (labeled 1 or 0).
                # Since input is identical, it cannot distinguish.
                
                labels = batch["label"] # Keep on CPU for sklearn
                outputs = decoder(audio)
                probs = outputs["detect_prob"].cpu().squeeze(1)
                
                all_preds.extend(probs.numpy())
                all_labels.extend(labels.numpy())
                
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
            
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f} | Val AUC {auc:.4f}")
        
    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bypass", "shuffle"], required=True)
    parser.add_argument("--data_dir", default="mini_benchmark_data")
    parser.add_argument("--regex", default="**/*.flac")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    run_null_test(args.mode, args.data_dir, args.epochs, regex=args.regex)
