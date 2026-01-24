
import os
import shutil
import urllib.request
import tarfile
import random
import json
import soundfile as sf
from glob import glob
from pathlib import Path

DATA_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DATA_DIR = Path("mini_benchmark_data")
MANIFEST_PATH = "mini_benchmark_manifest.json"

def download_and_extract():
    if DATA_DIR.exists():
        print(f"{DATA_DIR} exists, skipping download.")
        # Check if we have wav/flac files
        if len(list(DATA_DIR.rglob("*.flac"))) > 100:
            return
        
    print(f"Downloading {DATA_URL}...")
    tar_path = "dev-clean.tar.gz"
    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(DATA_URL, tar_path)
    
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()
    
    # Move Librispeech/dev-clean to DATA_DIR
    src = Path("LibriSpeech/dev-clean")
    if src.exists():
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        shutil.move(str(src), str(DATA_DIR))
        shutil.rmtree("LibriSpeech")
    
    # Clean up tar
    # os.remove(tar_path) # Keep it for now just in case
    print("Data extraction complete.")

def create_manifest(num_clips=500):
    files = list(DATA_DIR.rglob("*.flac"))
    random.shuffle(files)
    
    selected = files[:num_clips]
    print(f"Selected {len(selected)} clips from {len(files)} total.")
    
    # Split Train/Test (80/20) - actually we just need a collection, split is done in logic
    # But let's create a single manifest for Training.
    # Evaluation will pick unseen files (the remaining ones from `files` list?).
    # User asked for strict train/test split.
    # Let's use the first 400 for Train Manifest, leave 100 for Test (managed by benchmark script).
    
    train_files = selected[:int(num_clips * 0.8)]
    test_files = selected[int(num_clips * 0.8):]
    
    # Create Train Manifest (50% watermarked, 50% clean)
    manifest = []
    for i, f in enumerate(train_files):
        is_pos = i % 2 == 0
        manifest.append({
            "path": str(f.absolute()),
            "has_watermark": 1 if is_pos else 0,
            "model_id": i % 8 if is_pos else None,
            "version": 1
        })
        
    with open("mini_benchmark_train.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    # Create Test Manifest (Just paths, we will watermark manually for evaluation)
    test_manifest = []
    for f in test_files:
         test_manifest.append(str(f.absolute()))
         
    with open("mini_benchmark_test.json", "w") as f:
        json.dump(test_manifest, f, indent=2)
        
    print(f"Created 'mini_benchmark_train.json' with {len(manifest)} items.")
    print(f"Created 'mini_benchmark_test.json' with {len(test_manifest)} items.")

if __name__ == "__main__":
    download_and_extract()
    create_manifest()
