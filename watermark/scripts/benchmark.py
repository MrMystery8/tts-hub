"""
Benchmark Script for Watermark System

1. Generates synthetic dataset (simulating dataset creation cost).
2. Runs short training epochs (simulating training cost).
3. Extrapolates time for large scale datasets.
"""
import time
import torch
import numpy as np
import soundfile as sf
import json
import shutil
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from watermark.config import DEVICE, SAMPLE_RATE
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2

def generate_synthetic_data(num_files=200, duration_sec=3, output_dir="benchmark_data"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"Generating {num_files} files ({duration_sec}s each)...")
    start = time.time()
    
    manifest = []
    
    for i in range(num_files):
        # Generate random audio
        # Mix of sine waves and noise to simulate complexity
        t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec))
        freq = np.random.uniform(100, 1000)
        audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        audio = audio.astype(np.float32)
        
        path = output_dir / f"sample_{i}.wav"
        sf.write(str(path), audio, SAMPLE_RATE)
        
        is_pos = i % 2 == 0
        manifest.append({
            "path": str(path.absolute()),
            "has_watermark": 1 if is_pos else 0,
            "model_id": i % 8 if is_pos else None,
            "version": 1
        })
        
    duration = time.time() - start
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Dataset generation took {duration:.2f}s ({num_files/duration:.1f} files/sec)")
    return output_dir / "manifest.json", duration

def run_benchmark(manifest_path, epochs=1):
    print(f"\nRunning training benchmark on {DEVICE}...")
    
    codec = MessageCodec(key="test")
    
    # Models
    base_encoder = WatermarkEncoder(msg_bits=32)
    encoder = OverlapAddEncoder(base_encoder).to(DEVICE)
    
    base_decoder = WatermarkDecoder(msg_bits=32)
    decoder = SlidingWindowDecoder(base_decoder).to(DEVICE)
    
    # Dataset
    data_start = time.time()
    dataset = WatermarkDataset(manifest_path, codec=codec, training=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0) # MPS often dislikes multiprocessing workers
    # Just iterate once to verify load speed
    _ = next(iter(loader))
    load_time = time.time() - data_start
    print(f"Dataloader init + first batch: {load_time:.3f}s")

    # Time Stage 1
    s1_start = time.time()
    train_stage1(decoder, encoder, loader, DEVICE, epochs=epochs, log_interval=9999)
    s1_time = time.time() - s1_start
    print(f"Stage 1 ({epochs} ep): {s1_time:.2f}s")
    
    # Time Stage 2
    s2_start = time.time()
    train_stage2(encoder, decoder, loader, DEVICE, epochs=epochs, log_interval=9999)
    s2_time = time.time() - s2_start
    print(f"Stage 2 ({epochs} ep): {s2_time:.2f}s")
    
    return {
        "stage1_time": s1_time,
        "stage2_time": s2_time,
    }

def print_projection(num_files_bench, gen_time, bench_res, epochs_full=20):
    """
    Project times for typical datasets.
    LJSpeech ~ 13,100 clips (24h).
    LibriTTS-100h ~ 50,000 clips? (Actually much more, let's assume 100h approx 60,000 clips of 6s? Or 120,000 of 3s)
    Let's assume 100h = 120,000 clips (3s avg).
    """
    
    def fmt_time(seconds):
        if seconds < 60: return f"{seconds:.1f}s"
        if seconds < 3600: return f"{seconds/60:.1f}m"
        return f"{seconds/3600:.1f}h"

    # Metrics
    sec_per_file_gen = gen_time / num_files_bench
    
    # Training time per file per epoch
    # We ran X epochs.
    # time_per_epoch = total / epochs
    # time_per_file_per_epoch = time_per_epoch / num_files_bench
    
    epochs_bench = 5 # hardcoded in main call below
    
    s1_per_epoch = bench_res["stage1_time"] / epochs_bench
    s2_per_epoch = bench_res["stage2_time"] / epochs_bench
    
    print("\n" + "="*40)
    print("       TIME PROJECTIONS       ")
    print("="*40)
    
    datasets = [
        ("LJSpeech (24h)", 13100),
        ("LibriTTS (100h)", 120000), 
        ("Full Scale (1000h)", 1200000)
    ]
    
    print(f"{'Dataset':<20} | {'Gen Data':<10} | {'Train S1 (20ep)':<15} | {'Train S2 (20ep)':<15} | {'Total Est':<10}")
    print("-" * 85)
    
    for name, count in datasets:
        gen_est = count * sec_per_file_gen
        
        # Linear scaling assumption (valid if batch size constant and IO bottleneck similar)
        scale = count / num_files_bench
        
        s1_est = s1_per_epoch * 20 * scale # 20 epochs
        s2_est = s2_per_epoch * 20 * scale # 20 epochs
        
        total = gen_est + s1_est + s2_est
        
        print(f"{name:<20} | {fmt_time(gen_est):<10} | {fmt_time(s1_est):<15} | {fmt_time(s2_est):<15} | {fmt_time(total):<10}")
    
    print("-" * 85)
    print("Note: Estimates assume linear scaling and single-GPU/MPS throughput.")

def main():
    N = 200 # Benchmark size
    manifest_path, gen_time = generate_synthetic_data(num_files=N)
    
    # Run 5 epochs for stability
    res = run_benchmark(manifest_path, epochs=5)
    
    print_projection(N, gen_time, res)
    
    # Cleanup
    shutil.rmtree("benchmark_data")

if __name__ == "__main__":
    main()
