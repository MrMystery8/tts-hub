#!/usr/bin/env python3
"""
Complete training pipeline for audio watermarking.
Usage: python -m watermark.scripts.train_full --manifest /path/to/manifest.json --output ./checkpoints
"""
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from watermark.config import DEVICE, SAMPLE_RATE, SEGMENT_SAMPLES
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(description="Watermark Training Pipeline")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs_s1", type=int, default=20, help="Stage 1 epochs")
    parser.add_argument("--epochs_s1b", type=int, default=10, help="Stage 1B epochs")
    parser.add_argument("--warmup_s1b", type=int, default=3, help="Stage 1B warmup epochs")
    parser.add_argument("--epochs_s2", type=int, default=20, help="Stage 2 epochs")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    
    # Initialize codec
    codec = MessageCodec(key="fyp2026")
    
    # Initialize models
    base_encoder = WatermarkEncoder(msg_bits=32)
    encoder = OverlapAddEncoder(base_encoder, window=16000, hop_ratio=0.5)
    
    base_decoder = WatermarkDecoder(msg_bits=32)
    decoder = SlidingWindowDecoder(base_decoder, window=16000, hop_ratio=0.5, top_k=3)
    
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    
    # Create dataset
    dataset = WatermarkDataset(args.manifest, codec=codec, training=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # === STAGE 1: Detection ===
    print("\n=== Stage 1: Detection Training ===")
    train_stage1(decoder, loader, DEVICE, epochs=args.epochs_s1)
    torch.save(decoder.state_dict(), output_dir / "decoder_stage1.pt")
    
    # === STAGE 1B: Payload (with curriculum) ===
    print("\n=== Stage 1B: Payload Training ===")
    train_stage1b(
        decoder, 
        loader, 
        DEVICE, 
        preamble=codec.preamble, 
        epochs=args.epochs_s1b, 
        warmup=args.warmup_s1b
    )
    torch.save(decoder.state_dict(), output_dir / "decoder_stage1b.pt")
    
    # === STAGE 2: Encoder ===
    print("\n=== Stage 2: Encoder Training ===")
    train_stage2(encoder, decoder, loader, DEVICE, epochs=args.epochs_s2)
    torch.save(encoder.state_dict(), output_dir / "encoder_stage2.pt")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
