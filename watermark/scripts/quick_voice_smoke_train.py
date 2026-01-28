#!/usr/bin/env python3
"""
Quick smoke train using REAL voice clips from a local dataset (multiclass attribution).

Multiclass attribution:
  - class 0: clean / not watermarked
  - class 1..K: model attribution classes (K = N_MODELS)

This script is dashboard-compatible: it writes `metrics.jsonl` via `JSONLMetricsLogger`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from watermark.config import DEVICE, N_CLASSES, N_MODELS
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage2 import train_stage2
from watermark.utils.checkpointing import CheckpointManager
from watermark.utils.metrics_logger import JSONLMetricsLogger


def collect_audio_files(source_dir: Path) -> list[Path]:
    exts = [".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg"]
    files = [p for p in source_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def build_manifest(paths: list[Path], out_dir: Path) -> Path:
    """
    Manifest format (compat):
      - has_watermark: 0/1
      - model_id: 0..N_MODELS-1 for positives, -1 for clean
      - version: optional metadata (not watermarked in multiclass mode)
    """
    manifest: list[dict[str, object]] = []
    pos_idx = 0
    for i, p in enumerate(paths):
        is_pos = (i % 2 == 0)
        if is_pos:
            pair = pos_idx % (N_MODELS * 16)
            model_id = pair % N_MODELS
            version = (pair // N_MODELS) % 16
            pos_idx += 1
        else:
            model_id = -1
            version = -1
        manifest.append(
            {
                "path": str(p.resolve()),
                "has_watermark": 1 if is_pos else 0,
                "model_id": int(model_id),
                "version": int(version),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick voice smoke train (Dashboard Compatible)")
    parser.add_argument("--source_dir", type=str, default="mini_benchmark_data")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional: use an existing manifest JSON instead of sampling from --source_dir",
    )
    parser.add_argument("--out", type=str, default="outputs/quick_voice_smoke")
    parser.add_argument("--num_clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_metrics", type=str, default=None)
    parser.add_argument("--reverb_prob", type=float, default=0.0)
    parser.add_argument("--load_encoder", type=str, default=None, help="Optional: path to encoder .pt state_dict")
    parser.add_argument("--load_decoder", type=str, default=None, help="Optional: path to decoder .pt state_dict")

    # Schedule
    parser.add_argument("--epochs_s1", type=int, default=3, help="Stage 1: decoder pretrain")
    parser.add_argument("--epochs_s1b", type=int, default=0, help="Legacy: folded into Stage 1 epochs")
    parser.add_argument("--epochs_s2", type=int, default=3, help="Stage 2: encoder training (decoder frozen)")
    parser.add_argument("--epochs_s1b_post", type=int, default=2, help="Stage 3: finetune encoder+decoder")

    # Legacy weight flags (accepted for dashboard compatibility; not used in multiclass mode)
    parser.add_argument("--msg_weight", type=float, default=1.0)
    parser.add_argument("--model_ce_weight", type=float, default=1.0)
    parser.add_argument("--version_ce_weight", type=float, default=1.0)
    parser.add_argument("--pair_ce_weight", type=float, default=1.0)
    parser.add_argument("--unknown_ce_weight", type=float, default=1.0)
    parser.add_argument("--neg_weight", type=float, default=1.0)
    parser.add_argument("--neg_preamble_target", type=float, default=0.5)
    parser.add_argument("--stage2_payload_on_all", action="store_true")

    # Logging / probe
    parser.add_argument("--log_steps_every", type=int, default=25)
    parser.add_argument("--probe_every", type=int, default=1)
    parser.add_argument("--probe_reverb_every", type=int, default=999)
    parser.add_argument("--probe_clips", type=int, default=128)
    parser.add_argument("--detect_weight", type=float, default=1.0, help="Weight for detect loss")
    parser.add_argument("--id_weight", type=float, default=2.0, help="Weight for ID loss (positives only)")
    parser.add_argument(
        "--freeze_detect_head_in_s3",
        action="store_true",
        help="Freeze detect head during finetune to reduce detect/ID interference",
    )

    # Checkpointing options
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save checkpoints (default: <out>/checkpoints)")
    parser.add_argument("--save_last", action="store_true", default=True, help="Save last checkpoint (default: True)")
    parser.add_argument("--no_save_last", dest="save_last", action="store_false", help="Disable saving last checkpoint")
    parser.add_argument("--save_best", action="store_true", default=True, help="Save best checkpoint (default: True)")
    parser.add_argument("--no_save_best", dest="save_best", action="store_false", help="Disable saving best checkpoint")
    parser.add_argument("--best_metric", type=str, default=None, help="Metric to use for best checkpoint (default: auto-select)")
    parser.add_argument("--best_mode", type=str, choices=["min", "max"], default="max", help="Minimize or maximize best metric (default: max)")
    parser.add_argument("--save_every", type=int, default=1, help="Save last checkpoint every N epochs (0 to save every epoch)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    print(f"[QuickVoice] Using device: {DEVICE}")
    print(f"[QuickVoice] N_MODELS={N_MODELS} N_CLASSES={N_CLASSES}")

    rng = random.Random(int(args.seed))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.exists():
            print(f"[QuickVoice] Manifest not found: {manifest_path}")
            return 1
    else:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"[QuickVoice] Source dir not found: {source_dir}")
            return 1
        audio_files = collect_audio_files(source_dir)
        if not audio_files:
            print(f"No audio files in {source_dir}")
            return 1

        num_clips = int(args.num_clips) if args.num_clips is not None else 500
        if num_clips <= 0:
            print("--num_clips must be > 0")
            return 1

        if num_clips < len(audio_files):
            selected = rng.sample(audio_files, num_clips)
        else:
            selected = list(audio_files)
            while len(selected) < num_clips:
                selected.append(rng.choice(audio_files))
        manifest_path = build_manifest(selected, out_dir)

    # Compute schedule EARLY (needed for default best_metric)
    epochs_s1_total = int(args.epochs_s1) + max(0, int(args.epochs_s1b))
    epochs_s2 = int(args.epochs_s2)
    epochs_s3 = int(args.epochs_s1b_post)
    print(f"[QuickVoice] Schedule: s1={epochs_s1_total}, s2_encoder={epochs_s2}, s3_finetune={epochs_s3}")

    # Models
    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES)).to(DEVICE)

    if args.load_encoder:
        p = Path(args.load_encoder).expanduser().resolve()
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        encoder.load_state_dict(state, strict=True)
        print(f"[QuickVoice] Loaded encoder weights: {p}")

    if args.load_decoder:
        p = Path(args.load_decoder).expanduser().resolve()
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        decoder.load_state_dict(state, strict=True)
        print(f"[QuickVoice] Loaded decoder weights: {p}")

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        run_dir=out_dir,
        save_last=args.save_last,
        save_best=args.save_best,
        best_metric=args.best_metric or CheckpointManager.get_default_best_metric(
            epochs_s1=epochs_s1_total, epochs_s2=epochs_s2, epochs_s1b_post=epochs_s3
        ),
        best_mode=args.best_mode,
        save_every=args.save_every,
        ckpt_dir=args.ckpt_dir,
    )

    # Save config.json and copy manifest for self-contained run directory
    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Copy manifest to run directory if it exists and is not already in the right place
    if manifest_path.exists():
        import shutil
        dest_manifest = out_dir / "manifest.json"
        # Only copy if source and destination are different files
        if manifest_path.resolve() != dest_manifest.resolve():
            shutil.copy2(manifest_path, dest_manifest)

    # Data
    dataset = WatermarkDataset(str(manifest_path), training=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Probe cache (deterministic center crop)
    probe_items: list[ProbeItem] = []
    probe_dataset = WatermarkDataset(str(manifest_path), training=False)
    probe_n = min(max(0, int(args.probe_clips)), len(probe_dataset))
    for i in range(probe_n):
        it = probe_dataset[i]
        probe_items.append(ProbeItem(audio=it["audio"].detach().cpu(), y_class=int(it["y_class"].item())))

    metrics_path = Path(args.log_metrics) if args.log_metrics else (out_dir / "metrics.jsonl")

    def log_event(logger: JSONLMetricsLogger, event: dict) -> None:
        logger.log(event)

    def maybe_probe(logger: JSONLMetricsLogger, stage: str, epoch: int) -> dict | None:
        # Fix: Handle probe_every=0 to disable probing
        pe = int(args.probe_every)
        if pe <= 0:
            return None
        if epoch != 1 and (epoch % pe) != 0:
            return None

        # Fix: Handle probe_reverb_every=0 to avoid ZeroDivisionError
        pr = int(args.probe_reverb_every)
        reverb = (pr > 0 and (epoch % pr) == 0)

        metrics = compute_probe_metrics(
            probe_items,
            encoder=encoder,
            decoder=decoder,
            device=DEVICE,
            compute_reverb=bool(reverb),
        )
        log_event(logger, {"type": "probe", "stage": stage, "epoch": int(epoch), **metrics})
        return metrics

    def handle_epoch_end_s1(e):
        log_event(mlog, e)
        metrics = maybe_probe(mlog, "s1", int(e["epoch"]))
        ckpt_manager.save_last(
            encoder=encoder,
            decoder=decoder,
            stage="s1",
            epoch=int(e["epoch"]),
            args=vars(args)
        )
        if metrics is not None:
            ckpt_manager.maybe_save_best(
                encoder=encoder,
                decoder=decoder,
                stage="s1",
                epoch=int(e["epoch"]),
                probe_metrics=metrics,
                args=vars(args)
            )

    def handle_epoch_end_s2(e):
        log_event(mlog, e)
        metrics = maybe_probe(mlog, "s2_encoder", int(e["epoch"]))
        ckpt_manager.save_last(
            encoder=encoder,
            decoder=decoder,
            stage="s2_encoder",
            epoch=int(e["epoch"]),
            args=vars(args)
        )
        if metrics is not None:
            ckpt_manager.maybe_save_best(
                encoder=encoder,
                decoder=decoder,
                stage="s2_encoder",
                epoch=int(e["epoch"]),
                probe_metrics=metrics,
                args=vars(args)
            )

    def handle_epoch_end_s3(e):
        log_event(mlog, e)
        metrics = maybe_probe(mlog, "s3_finetune", int(e["epoch"]))
        ckpt_manager.save_last(
            encoder=encoder,
            decoder=decoder,
            stage="s3_finetune",
            epoch=int(e["epoch"]),
            args=vars(args)
        )
        if metrics is not None:
            ckpt_manager.maybe_save_best(
                encoder=encoder,
                decoder=decoder,
                stage="s3_finetune",
                epoch=int(e["epoch"]),
                probe_metrics=metrics,
                args=vars(args)
            )

    # Resume from checkpoint if specified
    start_epoch_s1 = 0
    start_epoch_s2 = 0
    start_epoch_s3 = 0

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        print(f"[QuickVoice] Resuming from checkpoint: {resume_path}")
        ckpt_data = ckpt_manager.resume_from_checkpoint(resume_path, encoder, decoder)

        # Stage-aware resume: determine which stage to resume from
        resumed_stage = ckpt_data.get("stage", "s1")
        resumed_epoch = ckpt_data.get("epoch", 0)

        if resumed_stage == "s1":
            start_epoch_s1 = resumed_epoch + 1
            print(f"[QuickVoice] Resumed from Stage 1, epoch {start_epoch_s1}")
        elif resumed_stage == "s2_encoder":
            start_epoch_s1 = epochs_s1_total  # skip stage 1 entirely
            start_epoch_s2 = resumed_epoch + 1
            print(f"[QuickVoice] Resumed from Stage 2, epoch {start_epoch_s2}")
        elif resumed_stage == "s3_finetune":
            start_epoch_s1 = epochs_s1_total  # skip stage 1
            start_epoch_s2 = epochs_s2        # skip stage 2
            start_epoch_s3 = resumed_epoch + 1
            print(f"[QuickVoice] Resumed from Stage 3, epoch {start_epoch_s3}")
        else:
            # Default to s1 if stage is unknown
            start_epoch_s1 = resumed_epoch + 1
            print(f"[QuickVoice] Resumed from epoch {start_epoch_s1} (unknown stage, defaulting to s1)")

    with JSONLMetricsLogger(metrics_path) as mlog:
        mlog.log(
            {
                "type": "meta",
                "run_name": out_dir.name,
                "config": vars(args),
                "device": str(DEVICE),
                "metrics_path": str(metrics_path),
                "manifest": str(manifest_path),
                "n_classes": int(N_CLASSES),
            }
        )

        if epochs_s1_total > 0:
            print(f"\n[Stage 1] Decoder pretraining (multiclass), starting from epoch {start_epoch_s1}")
            train_stage1(
                decoder,
                encoder,
                loader,
                DEVICE,
                epochs=epochs_s1_total,
                detect_weight=float(args.detect_weight),
                id_weight=float(args.id_weight),
                on_step=lambda e: log_event(mlog, e) if int(args.log_steps_every) > 0 and int(e.get("batch", 0)) % int(args.log_steps_every) == 0 else None,
                on_epoch_end=handle_epoch_end_s1,
                start_epoch=start_epoch_s1,
            )

        if epochs_s2 > 0:
            print(f"\n[Stage 2] Encoder training (decoder frozen), starting from epoch {start_epoch_s2}")
            train_stage2(
                encoder,
                decoder,
                loader,
                DEVICE,
                epochs=epochs_s2,
                reverb_prob=float(args.reverb_prob),
                detect_weight=float(args.detect_weight),
                id_weight=float(args.id_weight),
                on_step=lambda e: log_event(mlog, e) if int(args.log_steps_every) > 0 and int(e.get("batch", 0)) % int(args.log_steps_every) == 0 else None,
                on_epoch_end=handle_epoch_end_s2,
                finetune_mode=False,
                start_epoch=start_epoch_s2,
            )

        if epochs_s3 > 0:
            print(f"\n[Stage 3] Finetuning (encoder + decoder), starting from epoch {start_epoch_s3}")
            train_stage2(
                encoder,
                decoder,
                loader,
                DEVICE,
                epochs=epochs_s3,
                lr=1e-5,
                reverb_prob=float(args.reverb_prob),
                neg_weight=float(args.neg_weight),
                detect_weight=float(args.detect_weight),
                id_weight=float(args.id_weight),
                freeze_detect_head=bool(args.freeze_detect_head_in_s3),
                on_step=lambda e: log_event(mlog, e) if int(args.log_steps_every) > 0 and int(e.get("batch", 0)) % int(args.log_steps_every) == 0 else None,
                on_epoch_end=handle_epoch_end_s3,
                finetune_mode=True,
                start_epoch=start_epoch_s3,
            )

    # Save final models to run directory for compatibility (CPU-safe, atomic)
    print(f"\nSaved results to {out_dir}")
    encoder_state_cpu = {k: v.detach().to("cpu") for k, v in encoder.state_dict().items()}
    decoder_state_cpu = {k: v.detach().to("cpu") for k, v in decoder.state_dict().items()}

    # Atomic save to prevent corruption on crash
    temp_encoder_path = str(out_dir / "encoder.pt") + ".tmp"
    temp_decoder_path = str(out_dir / "decoder.pt") + ".tmp"

    torch.save(encoder_state_cpu, temp_encoder_path)
    os.replace(temp_encoder_path, str(out_dir / "encoder.pt"))

    torch.save(decoder_state_cpu, temp_decoder_path)
    os.replace(temp_decoder_path, str(out_dir / "decoder.pt"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
