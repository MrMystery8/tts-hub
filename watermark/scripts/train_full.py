#!/usr/bin/env python3
"""
Complete training pipeline for audio watermarking (multiclass attribution).

Usage:
  python -m watermark.scripts.train_full --manifest /path/to/manifest.json --output ./checkpoints/run1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from watermark.config import DEVICE, N_CLASSES
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage2 import train_stage2
from watermark.utils.metrics_logger import JSONLMetricsLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Watermark Training Pipeline (Multiclass)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints")

    parser.add_argument("--epochs_s1", type=int, default=20, help="Stage 1 epochs (decoder pretrain)")
    parser.add_argument("--epochs_s1b", type=int, default=0, help="Legacy: folded into Stage 1 epochs")
    parser.add_argument("--epochs_s2", type=int, default=20, help="Stage 2 epochs (encoder)")
    parser.add_argument("--epochs_s1b_post", type=int, default=0, help="Stage 3 epochs (finetune encoder+decoder)")

    parser.add_argument("--neg_weight", type=float, default=0.4, help="Stage 3: weight for clean CE regularization")
    parser.add_argument("--reverb_prob", type=float, default=0.25, help="Stage 2/3 differentiable reverb probability")

    parser.add_argument("--log_metrics", type=str, default=None, help="Write JSONL metrics for live dashboard")
    parser.add_argument("--probe_every", type=int, default=1, help="Run probe every N epochs (S2 + S3)")
    parser.add_argument("--probe_clips", type=int, default=256, help="Number of probe clips (cached in RAM once)")
    parser.add_argument("--probe_reverb_every", type=int, default=1, help="Compute reverb probe every N probe runs")
    parser.add_argument("--log_steps_every", type=int, default=100, help="Log step events every N batches (0 disables)")

    # Legacy flags accepted for compatibility (ignored in multiclass mode)
    parser.add_argument("--warmup_s1b", type=int, default=0)
    parser.add_argument("--neg_preamble_target", type=float, default=0.5)
    parser.add_argument("--unknown_ce_weight", type=float, default=0.0)
    parser.add_argument("--s1b_heads_only", action="store_true")
    parser.add_argument("--model_ce_weight", type=float, default=1.0)
    parser.add_argument("--version_ce_weight", type=float, default=1.0)
    parser.add_argument("--pair_ce_weight", type=float, default=2.0)
    parser.add_argument("--msg_weight", type=float, default=1.0)
    parser.add_argument("--stage2_payload_on_all", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    print(f"Using device: {DEVICE}")
    print(f"N_CLASSES={N_CLASSES}")

    # Models
    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES)).to(DEVICE)

    # Data
    dataset = WatermarkDataset(str(manifest_path), training=True)
    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Probe cache (center crop, deterministic)
    probe_items: list[ProbeItem] = []
    probe_dataset = WatermarkDataset(str(manifest_path), training=False)
    probe_n = min(max(0, int(args.probe_clips)), len(probe_dataset))
    for i in range(probe_n):
        it = probe_dataset[i]
        probe_items.append(ProbeItem(audio=it["audio"].detach().cpu(), y_class=int(it["y_class"].item())))

    metrics_path = Path(args.log_metrics) if args.log_metrics else (output_dir / "metrics.jsonl")
    probe_every = max(1, int(args.probe_every))
    probe_reverb_every = max(1, int(args.probe_reverb_every))

    best_by_wm_acc: dict[str, dict[str, float]] = {}

    def maybe_probe(logger: JSONLMetricsLogger, *, stage: str, epoch: int) -> Optional[dict[str, Any]]:
        if epoch % probe_every != 0 and epoch != 1:
            return None
        do_reverb = (epoch % probe_reverb_every == 0)
        m = compute_probe_metrics(
            probe_items,
            encoder=encoder,
            decoder=decoder,
            device=DEVICE,
            compute_reverb=bool(do_reverb),
        )
        logger.log({"type": "probe", "stage": stage, "epoch": epoch, **m})
        return m

    def maybe_save_best(*, stage: str, epoch: int, metrics: Optional[dict[str, Any]]) -> None:
        if not metrics:
            return
        key = "wm_acc"
        v = metrics.get(key)
        if not isinstance(v, (int, float)):
            return
        cur = float(v)
        prev = float(best_by_wm_acc.get(stage, {}).get(key, -1.0))
        if cur <= prev:
            return
        best_by_wm_acc[stage] = {key: cur, "epoch": float(epoch)}
        if stage == "s2_encoder":
            torch.save(encoder.state_dict(), output_dir / "encoder_stage2_best.pt")
        elif stage == "s3_finetune":
            torch.save(decoder.state_dict(), output_dir / "decoder_stage3_best.pt")

    epochs_s1_total = int(args.epochs_s1) + max(0, int(args.epochs_s1b))

    with JSONLMetricsLogger(metrics_path) as mlog:
        mlog.log(
            {
                "type": "meta",
                "run_name": "train_full",
                "device": str(DEVICE),
                "metrics_path": str(metrics_path),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "n_classes": int(N_CLASSES),
                "config": {
                    "epochs_s1": int(args.epochs_s1),
                    "epochs_s1b": int(args.epochs_s1b),
                    "epochs_s2": int(args.epochs_s2),
                    "epochs_s1b_post": int(args.epochs_s1b_post),
                    "neg_weight": float(args.neg_weight),
                    "reverb_prob": float(args.reverb_prob),
                    "probe_n": int(probe_n),
                    "probe_every": int(probe_every),
                    "probe_reverb_every": int(probe_reverb_every),
                    "log_steps_every": int(args.log_steps_every),
                    "batch_size": int(batch_size),
                },
            }
        )

        # === STAGE 1: Decoder pretrain ===
        if epochs_s1_total > 0:
            print("\n=== Stage 1: Decoder pretraining ===")
            train_stage1(
                decoder,
                encoder,
                loader,
                DEVICE,
                epochs=epochs_s1_total,
                step_interval=int(args.log_steps_every),
                on_step=mlog.log,
                on_epoch_end=mlog.log,
            )
            torch.save(decoder.state_dict(), output_dir / "decoder_stage1.pt")

        # === STAGE 2: Encoder ===
        if int(args.epochs_s2) > 0:
            print("\n=== Stage 2: Encoder training ===")
            train_stage2(
                encoder,
                decoder,
                loader,
                DEVICE,
                epochs=int(args.epochs_s2),
                reverb_prob=float(args.reverb_prob),
                step_interval=int(args.log_steps_every),
                on_step=mlog.log,
                on_epoch_end=lambda e: (
                    mlog.log(e),
                    maybe_save_best(stage="s2_encoder", epoch=int(e["epoch"]), metrics=maybe_probe(mlog, stage="s2_encoder", epoch=int(e["epoch"]))),
                ),
                finetune_mode=False,
            )
            torch.save(encoder.state_dict(), output_dir / "encoder_stage2.pt")

        # === STAGE 3: Finetune ===
        if int(args.epochs_s1b_post) > 0:
            print("\n=== Stage 3: Finetuning (encoder + decoder) ===")
            train_stage2(
                encoder,
                decoder,
                loader,
                DEVICE,
                epochs=int(args.epochs_s1b_post),
                lr=1e-5,
                reverb_prob=float(args.reverb_prob),
                neg_weight=float(args.neg_weight),
                step_interval=int(args.log_steps_every),
                on_step=mlog.log,
                on_epoch_end=lambda e: (
                    mlog.log(e),
                    maybe_save_best(stage="s3_finetune", epoch=int(e["epoch"]), metrics=maybe_probe(mlog, stage="s3_finetune", epoch=int(e["epoch"]))),
                ),
                finetune_mode=True,
            )
            torch.save(decoder.state_dict(), output_dir / "decoder_stage3.pt")

    print(f"\nSaved results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

