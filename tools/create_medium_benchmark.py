#!/usr/bin/env python3
"""
Script to create a 'medium_benchmark_data' dataset with ~20,000 files 
from LibriSpeech train-clean-100.
"""
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
TARGET_DIR = Path("medium_benchmark_data")
TEMP_NAME = "train-clean-100"
ARCHIVE_NAME = "train-clean-100.tar.gz"
TARGET_COUNT = 20000

def main():
    if TARGET_DIR.exists():
        print(f"Error: {TARGET_DIR} already exists. Please remove it first.")
        return 1

    # 1. Download
    if not Path(ARCHIVE_NAME).exists():
        print(f"Downloading {URL}...")
        subprocess.run(["curl", "-O", URL], check=True)
    else:
        print(f"Found existing {ARCHIVE_NAME}, skipping download.")

    # 2. Extract
    print(f"Extracting {ARCHIVE_NAME}...")
    # LibriSpeech tarballs extract into a 'LibriSpeech' folder
    # We'll extract to a temporary subdir to keep things clean
    extract_root = Path("temp_extraction")
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir()
    
    subprocess.run(["tar", "-xzf", ARCHIVE_NAME, "-C", str(extract_root)], check=True)
    
    # Path is typically extracts/LibriSpeech/train-clean-100
    source_root = extract_root / "LibriSpeech" / "train-clean-100"
    if not source_root.exists():
        print(f"Error: Expected extracted path {source_root} not found.")
        return 1

    # 3. Select Speakers
    print("Scanning speakers...")
    speakers = [d for d in source_root.iterdir() if d.is_dir()]
    selected_speakers = []
    total_files = 0
    
    # Sort for deterministic selection
    speakers.sort(key=lambda p: int(p.name))

    for spk_dir in speakers:
        if total_files >= TARGET_COUNT:
            break
        
        # Count flac files recursively
        count = sum(1 for _ in spk_dir.rglob("*.flac"))
        if count > 0:
            selected_speakers.append(spk_dir)
            total_files += count
    
    print(f"Selected {len(selected_speakers)} speakers with {total_files} files.")

    # 4. Move
    print(f"Moving extracted subset to {TARGET_DIR}...")
    TARGET_DIR.mkdir()
    
    for spk_dir in selected_speakers:
        shutil.move(str(spk_dir), str(TARGET_DIR / spk_dir.name))

    # 5. Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(extract_root)
    os.unlink(ARCHIVE_NAME)
    
    print("Done!")
    print(f"Created {TARGET_DIR} with {total_files} files.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
