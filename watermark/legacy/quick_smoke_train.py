#!/usr/bin/env python3
"""
Quick smoke test for the full watermarking pipeline.

Creates a tiny synthetic dataset, runs Stage 1/1B/2 for a few epochs,
and writes listenable clean/watermarked WAVs for a hearing check.

LEGACY NOTE:
This script targets the old bit-payload pipeline. The current supported smoke run is:
  `python -m watermark.scripts.quick_voice_smoke_train ...`
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from watermark.config import DEVICE, SAMPLE_RATE, SEGMENT_SAMPLES
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2
from watermark.utils.io import load_audio, save_audio
from watermark.evaluation.attacks import ATTACKS, apply_attack_safe


def _make_sine_mix(duration_sec: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    freqs = rng.choice(np.arange(150, 1200), size=3, replace=False)
    signal = sum(0.2 * np.sin(2 * math.pi * f * t) for f in freqs)
    noise = 0.02 * rng.standard_normal(len(t))
    audio = (signal + noise).astype(np.float32)
    return audio


def generate_dataset(out_dir: Path, num_clips: int, duration_sec: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i in range(num_clips):
        audio = _make_sine_mix(duration_sec, seed=1337 + i)
        audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        clip_path = out_dir / f"clip_{i:03d}.wav"
        save_audio(str(clip_path), audio_t, SAMPLE_RATE)

        is_pos = (i % 2 == 0)
        manifest.append({
            "path": str(clip_path.resolve()),
            "has_watermark": 1 if is_pos else 0,
            "model_id": (i % 8) if is_pos else None,
            "version": 1,
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick smoke train for watermark pipeline")
    parser.add_argument("--out", type=str, default="outputs/quick_smoke", help="Output directory")
    parser.add_argument("--num_clips", type=int, default=12, help="Number of synthetic clips")
    parser.add_argument("--duration_sec", type=float, default=3.0, help="Clip duration in seconds")
    parser.add_argument("--epochs_s1", type=int, default=2, help="Stage 1 epochs")
    parser.add_argument("--epochs_s1b", type=int, default=2, help="Stage 1B epochs")
    parser.add_argument("--epochs_s2", type=int, default=2, help="Stage 2 epochs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    data_dir = out_dir / "data"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QuickSmoke] Using device: {DEVICE}")
    manifest_path = generate_dataset(data_dir, args.num_clips, args.duration_sec)
    print(f"[QuickSmoke] Manifest: {manifest_path}")

    codec = MessageCodec(key="quick_smoke")
    encoder = OverlapAddEncoder(WatermarkEncoder(msg_bits=32)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(msg_bits=32)).to(DEVICE)

    dataset = WatermarkDataset(str(manifest_path), codec=codec, training=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print("\n[QuickSmoke] Stage 1 (Detection)")
    train_stage1(decoder, encoder, loader, DEVICE, epochs=args.epochs_s1, log_interval=999)

    print("\n[QuickSmoke] Stage 1B (Payload)")
    train_stage1b(decoder, encoder, loader, DEVICE, codec.preamble, epochs=args.epochs_s1b, warmup=1, log_interval=999)

    print("\n[QuickSmoke] Stage 2 (Encoder)")
    train_stage2(encoder, decoder, loader, DEVICE, epochs=args.epochs_s2, log_interval=999)

    # Save a clean + watermarked sample for listening.
    sample_item = dataset[0]
    clean = sample_item["audio"].unsqueeze(0).to(DEVICE)  # (1, 1, T)
    message = sample_item["message"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        watermarked = encoder(clean, message)

    clean_path = audio_dir / "clean.wav"
    wm_path = audio_dir / "watermarked.wav"

    save_audio(str(clean_path), clean.squeeze(0), SAMPLE_RATE)
    save_audio(str(wm_path), watermarked.squeeze(0), SAMPLE_RATE)

    # Optional attacked sample for comparison
    if "reverb" in ATTACKS:
        attacked = apply_attack_safe(watermarked.squeeze(0).cpu(), ATTACKS["reverb"])
        attacked_path = audio_dir / "watermarked_reverb.wav"
        save_audio(str(attacked_path), attacked, SAMPLE_RATE)
    else:
        attacked_path = None

    print("\n[QuickSmoke] Saved audio for listening:")
    print(f"  - Clean: {clean_path}")
    print(f"  - Watermarked: {wm_path}")
    if attacked_path:
        print(f"  - Watermarked + Reverb: {attacked_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
