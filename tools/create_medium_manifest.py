#!/usr/bin/env python3
"""
Script to create manifest files for the 'medium_benchmark_data' dataset.
"""
import json
import random
from pathlib import Path

DATA_DIR = Path("medium_benchmark_data")
TRAIN_MANIFEST = "medium_benchmark_train.json"
TEST_MANIFEST = "medium_benchmark_test.json"
N_MODELS = 8  # Standard for this project

def main():
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} not found.")
        return 1

    print("Scanning files...")
    files = list(DATA_DIR.rglob("*.flac"))
    files.sort() # Deterministic sort before shuffle
    random.seed(1337)
    random.shuffle(files)
    
    print(f"Found {len(files)} files.")
    
    # 80/20 Train/Test Split
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    print(f"Training: {len(train_files)} files")
    print(f"Testing:  {len(test_files)} files")
    
    # Create Train Manifest
    # 50% Watermarked / 50% Clean
    manifest = []
    
    # We want balanced classes for the positives (0..7)
    # Strategy: 
    #   Msg (Positive): id = (i % N_MODELS)
    #   Clean (Negative): id = -1
    
    # To keep it simple and balanced:
    # Iterate train files, alternate Pos/Neg
    
    for i, f in enumerate(train_files):
        is_pos = (i % 2 == 0)
        
        if is_pos:
            # Assign specific model ID (0..7)
            # We map the index to a model ID
            # Since we only do this for even indices (0, 2, 4...), 
            # we divide by 2 to get a continuous counter 0, 1, 2...
            counter = i // 2
            model_id = counter % N_MODELS
            has_wm = 1
        else:
            model_id = None # or -1, but logic often expects None in JSON for simplicity unless schema compliant
            has_wm = 0
            
        manifest.append({
            "path": str(f.absolute()),
            "has_watermark": has_wm,
            "model_id": model_id,
            "version": 1
        })
        
    with open(TRAIN_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        
    # Create Test Manifest (List of paths)
    test_manifest = [str(f.absolute()) for f in test_files]
    
    with open(TEST_MANIFEST, "w") as f:
        json.dump(test_manifest, f, indent=2)
        
    print(f"Created {TRAIN_MANIFEST} with {len(manifest)} items.")
    print(f"Created {TEST_MANIFEST} with {len(test_manifest)} items.")
    print("Done.")

if __name__ == "__main__":
    main()
