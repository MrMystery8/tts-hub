#!/usr/bin/env python3
"""
Complete training pipeline for audio watermarking (multiclass attribution).

Usage:
  python -m watermark.scripts.train_full --manifest /path/to/manifest.json --output ./checkpoints/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from watermark.config import DEVICE, N_CLASSES, N_MODELS
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage2 import train_stage2
from watermark.utils.metrics_logger import JSONLMetricsLogger


def _infer_n_models_from_manifest(manifest_path: Path, *, default: int) -> int:
    """
    Infer K (number of attribution IDs) from a manifest by taking max(model_id)+1 over positives.
    Falls back to `default` if inference fails or there are no positive samples.
    """
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            return int(default)
        max_id = -1
        for e in entries:
            if not isinstance(e, dict):
                continue
            has_wm = float(e.get("has_watermark", 0.0) or 0.0)
            if has_wm < 0.5:
                continue
            mid = e.get("model_id", None)
            if mid is None:
                continue
            m = int(mid)
            if m >= 0:
                max_id = max(max_id, m)
        if max_id >= 0:
            return int(max_id + 1)
        return int(default)
    except Exception:
        return int(default)


def main() -> int:
    parser = argparse.ArgumentParser(description="Watermark Training Pipeline (Multiclass)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--load_encoder", type=str, default=None, help="Optional: path to encoder .pt state_dict")
    parser.add_argument("--load_decoder", type=str, default=None, help="Optional: path to decoder .pt state_dict")
    parser.add_argument(
        "--n_models",
        type=int,
        default=None,
        help="Optional K (number of attribution IDs). If omitted, inferred from the manifest.",
    )

    parser.add_argument("--epochs_s1", type=int, default=20, help="Stage 1 epochs (decoder pretrain)")
    parser.add_argument("--epochs_s1b", type=int, default=0, help="Legacy: folded into Stage 1 epochs")
    parser.add_argument("--epochs_s2", type=int, default=20, help="Stage 2 epochs (encoder)")
    parser.add_argument("--epochs_s1b_post", type=int, default=0, help="Stage 3 epochs (finetune encoder+decoder)")

    parser.add_argument("--neg_weight", type=float, default=0.4, help="Stage 3: weight for clean CE regularization")
    parser.add_argument("--reverb_prob", type=float, default=0.25, help="Stage 2/3 differentiable reverb probability")
    parser.add_argument("--detect_weight", type=float, default=1.0, help="Weight for detect loss")
    parser.add_argument("--id_weight", type=float, default=2.0, help="Weight for ID loss (positives only)")
    parser.add_argument(
        "--freeze_detect_head_in_s3",
        action="store_true",
        help="Freeze detect head during finetune to reduce detect/ID interference",
    )

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

    n_models = int(args.n_models) if args.n_models is not None else _infer_n_models_from_manifest(manifest_path, default=int(N_MODELS))
    if n_models <= 0:
        raise ValueError(f"n_models must be >= 1, got {n_models}")
    num_classes = int(n_models + 1)

    print(f"Using device: {DEVICE}")
    print(f"N_CLASSES(default)={N_CLASSES} N_MODELS(default)={N_MODELS}")
    print(f"n_models(K)={n_models} num_classes(K+1)={num_classes}")

    # Models
    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=num_classes)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=num_classes)).to(DEVICE)

    if args.load_encoder:
        p = Path(args.load_encoder).expanduser().resolve()
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        encoder.load_state_dict(state, strict=True)
        print(f"Loaded encoder weights: {p}")

    if args.load_decoder:
        p = Path(args.load_decoder).expanduser().resolve()
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        decoder.load_state_dict(state, strict=True)
        print(f"Loaded decoder weights: {p}")

    # Data
    dataset = WatermarkDataset(str(manifest_path), training=True, n_models=n_models)
    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Probe cache (center crop, deterministic)
    probe_items: list[ProbeItem] = []
    probe_dataset = WatermarkDataset(str(manifest_path), training=False, n_models=n_models)
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
            num_classes=num_classes,
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
                "n_models": int(n_models),
                "num_classes": int(num_classes),
                "n_classes": int(num_classes),  # backward compat
                "config": {
                    "n_models": int(n_models),
                    "num_classes": int(num_classes),
                    "n_classes": int(num_classes),
                    "epochs_s1": int(args.epochs_s1),
                    "epochs_s1b": int(args.epochs_s1b),
                    "epochs_s2": int(args.epochs_s2),
                    "epochs_s1b_post": int(args.epochs_s1b_post),
                    "neg_weight": float(args.neg_weight),
                    "reverb_prob": float(args.reverb_prob),
                    "detect_weight": float(args.detect_weight),
                    "id_weight": float(args.id_weight),
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
                detect_weight=float(args.detect_weight),
                id_weight=float(args.id_weight),
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
                detect_weight=float(args.detect_weight),
                id_weight=float(args.id_weight),
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
                detect_weight=float(args.detect_weight),
                id_weight=float(args.id_weight),
                freeze_detect_head=bool(args.freeze_detect_head_in_s3),
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
