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
import numpy as np
import signal
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from watermark.config import DEVICE, N_CLASSES, N_MODELS
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1_adaptive import train_stage1_adaptive
from watermark.training.stage2 import train_stage2
from watermark.utils.checkpointing import CheckpointManager
from watermark.utils.loss_balancing import UncertaintyBalancer
from watermark.utils.metrics_logger import JSONLMetricsLogger


def _looks_like_state_dict(obj: object) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    # Heuristic: state_dict is usually str->Tensor (or Parameter). We accept tensors only here.
    for v in obj.values():
        if not torch.is_tensor(v):
            return False
    return True


def _extract_state_dict(ckpt: object, *, which: str) -> dict:
    """
    Accept either:
    - plain state_dict (str->Tensor)
    - a dict containing {"state_dict": ...}
    - a CheckpointManager checkpoint containing {"encoder": ..., "decoder": ...}
    """
    if _looks_like_state_dict(ckpt):
        return ckpt  # type: ignore[return-value]
    if isinstance(ckpt, dict):
        if which in ckpt and isinstance(ckpt.get(which), dict):
            return ckpt[which]  # type: ignore[return-value]
        sd = ckpt.get("state_dict")
        if isinstance(sd, dict) and _looks_like_state_dict(sd):
            return sd
    raise TypeError(f"Unsupported checkpoint format for {which}: {type(ckpt)}")


def collect_audio_files(source_dir: Path) -> list[Path]:
    exts = [".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg"]
    files = [p for p in source_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def build_manifest(
    paths: list[Path],
    out_dir: Path,
    *,
    filename: str = "manifest.json",
    n_models: int = N_MODELS,
) -> Path:
    """
    Manifest format (compat):
      - has_watermark: 0/1
      - model_id: 0..N_MODELS-1 for positives, -1 for clean
      - version: optional metadata (not watermarked in multiclass mode)
    """
    manifest: list[dict[str, object]] = []
    pos_idx = 0
    n_models = int(n_models)
    if n_models <= 0:
        raise ValueError(f"n_models must be >= 1, got {n_models}")
    for i, p in enumerate(paths):
        is_pos = (i % 2 == 0)
        if is_pos:
            pair = pos_idx % (n_models * 16)
            model_id = pair % n_models
            version = (pair // n_models) % 16
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
    manifest_path = out_dir / str(filename)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_manifest_entries(entries: list[dict[str, object]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return out_path


def _split_by_path(
    entries: list[dict[str, object]],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """
    Split manifest entries into train/val/test by unique `path`, keeping duplicates together.

    This prevents leakage where the same carrier clip appears in multiple splits.
    """
    train_frac = float(train_frac)
    val_frac = float(val_frac)
    test_frac = float(test_frac)
    if train_frac < 0 or val_frac < 0 or test_frac < 0:
        raise ValueError("split fracs must be >= 0")

    s = train_frac + val_frac + test_frac
    if s <= 0:
        raise ValueError("At least one split fraction must be > 0")
    if abs(s - 1.0) > 1e-6:
        train_frac /= s
        val_frac /= s
        test_frac /= s

    # Group indices by path so duplicates stay in the same split.
    buckets: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        p = str(e.get("path") or "")
        buckets.setdefault(p, []).append(i)

    paths = list(buckets.keys())
    rng = random.Random(int(seed))
    rng.shuffle(paths)

    n_paths = len(paths)
    n_train = int(round(train_frac * n_paths))
    n_val = int(round(val_frac * n_paths))
    n_train = max(0, min(n_paths, n_train))
    n_val = max(0, min(n_paths - n_train, n_val))
    n_test = max(0, n_paths - n_train - n_val)

    train_paths = set(paths[:n_train])
    val_paths = set(paths[n_train : n_train + n_val])
    test_paths = set(paths[n_train + n_val : n_train + n_val + n_test])

    train_entries: list[dict[str, object]] = []
    val_entries: list[dict[str, object]] = []
    test_entries: list[dict[str, object]] = []

    for p in paths:
        idxs = buckets[p]
        dst = train_entries if p in train_paths else val_entries if p in val_paths else test_entries
        for i in idxs:
            dst.append(entries[i])

    return train_entries, val_entries, test_entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick voice smoke train (Dashboard Compatible)")
    parser.add_argument("--source_dir", type=str, default="mini_benchmark_data")
    parser.add_argument(
        "--n_models",
        type=int,
        default=int(N_MODELS),
        help="Number of attribution IDs (K). Class 0 is clean; classes 1..K are IDs.",
    )
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
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train split fraction (by unique file path)")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation split fraction (by unique file path)")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test split fraction (by unique file path)")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for train/val/test splitting (default: use --seed)",
    )

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
    parser.add_argument(
        "--test_attacks",
        type=str,
        nargs="?",
        const="",
        default="resample_8k,noise_white_20db",
        help="Comma-separated extra attacks to evaluate in the final held-out test_probe (in addition to clean + reverb).",
    )
    parser.add_argument("--detect_weight", type=float, default=1.0, help="Weight for detect loss")
    parser.add_argument("--id_weight", type=float, default=2.0, help="Weight for ID loss (positives only)")
    parser.add_argument(
        "--freeze_detect_head_in_s3",
        action="store_true",
        help="Freeze detect head during finetune to reduce detect/ID interference",
    )
    
    # Adaptive Architecture
    parser.add_argument(
        "--s1_arch", 
        type=str, 
        choices=["static", "adaptive_uncertainty"], 
        default="static", 
        help="S1 architecture: 'static' (legacy) or 'adaptive_uncertainty' (learnable weights)"
    )
    parser.add_argument(
        "--balancer_init_detect", 
        type=float, 
        default=None, 
        help="Initial effective weight for detection (adaptive mode). Defaults to --detect_weight if unset."
    )
    parser.add_argument(
        "--balancer_init_id", 
        type=float, 
        default=None, 
        help="Initial effective weight for ID (adaptive mode). Defaults to --id_weight if unset."
    )
    parser.add_argument(
        "--force_arch_mismatch", 
        action="store_true", 
        help="Allow resuming even if checkpoint s1_arch differs from current (dangerous)"
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
    parser.add_argument(
        "--extend_to_epochs_s1",
        type=int,
        default=None,
        help="If resuming Stage 1, extend training UNTIL this total epoch count is reached.",
    )

    args = parser.parse_args()

    stop_requested = {"flag": False, "sig": None}

    def _handle_stop(sig: int, _frame) -> None:
        stop_requested["flag"] = True
        stop_requested["sig"] = int(sig)
        try:
            name = signal.Signals(sig).name
        except Exception:
            name = str(sig)
        print(f"[QuickVoice] Stop requested via {name}. Will stop gracefully soon (finishes partial epoch + test_probe).")

    try:
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
    except Exception:
        pass

    def should_stop() -> bool:
        return bool(stop_requested["flag"])

    n_models = int(args.n_models)
    if n_models <= 0:
        raise ValueError(f"--n_models must be >= 1, got {n_models}")
    num_classes = int(n_models + 1)

    print(f"[QuickVoice] Using device: {DEVICE}")
    print(f"[QuickVoice] N_MODELS(default)={N_MODELS} N_CLASSES(default)={N_CLASSES}")
    print(f"[QuickVoice] n_models(K)={n_models} num_classes(K+1)={num_classes}")

    # Reproducibility
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS reproducibility if needed, but manual_seed usually covers it.
    
    rng = random.Random(seed)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Proper train/val/test split (by unique carrier path, to avoid leakage).
    split_seed = int(args.split_seed) if args.split_seed is not None else seed
    manifest_train_path = out_dir / "manifest_train.json"
    manifest_val_path = out_dir / "manifest_val.json"
    manifest_test_path = out_dir / "manifest_test.json"

    if manifest_train_path.exists() and manifest_val_path.exists() and manifest_test_path.exists():
        print("[QuickVoice] Using existing split manifests from run directory.")
    else:
        entries: list[dict[str, object]]
        if args.manifest:
            src_manifest_path = Path(args.manifest).expanduser().resolve()
            if not src_manifest_path.exists():
                print(f"[QuickVoice] Manifest not found: {src_manifest_path}")
                return 1
            src_dir = src_manifest_path.parent
            src_train = src_dir / "manifest_train.json"
            src_val = src_dir / "manifest_val.json"
            src_test = src_dir / "manifest_test.json"
            src_full = src_dir / "manifest_full.json"

            # If the source manifest belongs to a prior run that already has split manifests,
            # reuse those splits instead of resplitting (avoids accidental data leakage/shift).
            if src_train.exists() and src_val.exists() and src_test.exists():
                print(f"[QuickVoice] Reusing pre-split manifests from: {src_dir}")
                manifest_train_path.write_text(src_train.read_text(encoding="utf-8"), encoding="utf-8")
                manifest_val_path.write_text(src_val.read_text(encoding="utf-8"), encoding="utf-8")
                manifest_test_path.write_text(src_test.read_text(encoding="utf-8"), encoding="utf-8")
                if src_full.exists():
                    (out_dir / "manifest_full.json").write_text(src_full.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                entries = json.loads(src_manifest_path.read_text(encoding="utf-8"))
                _write_manifest_entries(entries, out_dir / "manifest_full.json")
                train_entries, val_entries, test_entries = _split_by_path(
                    entries,
                    train_frac=float(args.train_frac),
                    val_frac=float(args.val_frac),
                    test_frac=float(args.test_frac),
                    seed=split_seed,
                )
                _write_manifest_entries(train_entries, manifest_train_path)
                _write_manifest_entries(val_entries, manifest_val_path)
                _write_manifest_entries(test_entries, manifest_test_path)
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

            full_manifest_path = build_manifest(selected, out_dir, filename="manifest_full.json", n_models=n_models)
            entries = json.loads(full_manifest_path.read_text(encoding="utf-8"))

            train_entries, val_entries, test_entries = _split_by_path(
                entries,
                train_frac=float(args.train_frac),
                val_frac=float(args.val_frac),
                test_frac=float(args.test_frac),
                seed=split_seed,
            )
            _write_manifest_entries(train_entries, manifest_train_path)
            _write_manifest_entries(val_entries, manifest_val_path)
            _write_manifest_entries(test_entries, manifest_test_path)

    manifest_path = manifest_train_path

    # Compute schedule EARLY (needed for default best_metric)
    epochs_s1_total = int(args.epochs_s1) + max(0, int(args.epochs_s1b))
    epochs_s2 = int(args.epochs_s2)
    epochs_s3 = int(args.epochs_s1b_post)
    print(f"[QuickVoice] Schedule: s1={epochs_s1_total}, s2_encoder={epochs_s2}, s3_finetune={epochs_s3}")

    # Models
    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=num_classes)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=num_classes)).to(DEVICE)

    if args.load_encoder:
        p = Path(args.load_encoder).expanduser().resolve()
        ckpt = torch.load(p, map_location="cpu")
        state = _extract_state_dict(ckpt, which="encoder")
        encoder.load_state_dict(state, strict=True)
        print(f"[QuickVoice] Loaded encoder weights: {p}")

    if args.load_decoder:
        p = Path(args.load_decoder).expanduser().resolve()
        ckpt = torch.load(p, map_location="cpu")
        state = _extract_state_dict(ckpt, which="decoder")
        decoder.load_state_dict(state, strict=True)
        print(f"[QuickVoice] Loaded decoder weights: {p}")

    # Build Balancer if needed
    balancer = None
    if args.s1_arch == "adaptive_uncertainty":
        # Default to existing weights if init args not provided
        init_det = args.balancer_init_detect if args.balancer_init_detect is not None else args.detect_weight
        init_id = args.balancer_init_id if args.balancer_init_id is not None else args.id_weight
        
        balancer = UncertaintyBalancer(
            init_weight_detect=float(init_det),
            init_weight_id=float(init_id)
        ).to(DEVICE)
        print(f"[QuickVoice] Initialized UncertaintyBalancer (init_det={init_det}, init_id={init_id})")

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
        cfg = dict(vars(args))
        cfg["n_models"] = int(n_models)
        cfg["num_classes"] = int(num_classes)
        cfg["n_classes"] = int(num_classes)  # backward compat
        json.dump(cfg, f, indent=2)

    # Save train/val/test manifests into the run directory (and also write manifest.json for compatibility).
    try:
        (out_dir / "manifest.json").write_text(manifest_train_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Data
    train_dataset = WatermarkDataset(str(manifest_train_path), training=True, n_models=n_models)
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Probe cache (validation; deterministic center crop)
    probe_items: list[ProbeItem] = []
    probe_dataset = WatermarkDataset(str(manifest_val_path), training=False, n_models=n_models)
    probe_n = min(max(0, int(args.probe_clips)), len(probe_dataset))
    for i in range(probe_n):
        it = probe_dataset[i]
        probe_items.append(ProbeItem(audio=it["audio"].detach().cpu(), y_class=int(it["y_class"].item())))

    # Split summary (by entries)
    try:
        n_train = len(json.loads(manifest_train_path.read_text(encoding="utf-8")))
        n_val = len(json.loads(manifest_val_path.read_text(encoding="utf-8")))
        n_test = len(json.loads(manifest_test_path.read_text(encoding="utf-8")))
        print(f"[QuickVoice] Split entries: train={n_train} val={n_val} test={n_test} (total={n_train + n_val + n_test})")
        print(f"[QuickVoice] Probe clips (val): {probe_n}")
    except Exception:
        pass

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
            num_classes=num_classes,
        )
        # Calculate composite score if available
        if "tpr_at_fpr_1pct" in metrics and "id_acc_pos" in metrics:
            metrics["composite_score"] = float(metrics["tpr_at_fpr_1pct"]) * float(metrics["id_acc_pos"])

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
            balancer=balancer,
            args=vars(args)
        )
        if metrics is not None:
            ckpt_manager.maybe_save_best(
                encoder=encoder,
                decoder=decoder,
                stage="s1",
                epoch=int(e["epoch"]),
                probe_metrics=metrics,
                balancer=balancer,
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
        # Note: We pass balancer to resume so it can load state if present
        ckpt_data = ckpt_manager.resume_from_checkpoint(resume_path, encoder, decoder, balancer=balancer)
        
        # Verify Architecture Consistency
        ckpt_arch = ckpt_data.get("s1_arch", "static")
        if ckpt_arch != args.s1_arch:
            msg = f"Architecture mismatch! Checkpoint is '{ckpt_arch}', but current run is '{args.s1_arch}'."
            if args.force_arch_mismatch:
                print(f"WARNING: {msg} Proceeding due to --force_arch_mismatch.")
            else:
                raise ValueError(f"{msg} Use --force_arch_mismatch to override if you really mean it.")

        # Stage-aware resume: determine which stage to resume from
        resumed_stage = ckpt_data.get("stage", "s1")
        resumed_epoch = ckpt_data.get("epoch", 0)

        if resumed_stage == "s1":
            # Handle extend logic
            if args.extend_to_epochs_s1 is not None:
                target = int(args.extend_to_epochs_s1)
                remaining = max(0, target - resumed_epoch)
                epochs_s1_total = remaining  # Override total epochs to run
                start_epoch_s1 = resumed_epoch
                
                # If we're already past the target, this will result in 0 epochs, which is correct
                print(f"[QuickVoice] Extending Stage 1: resumed at {resumed_epoch}, target {target}, running {remaining} more epochs")
            else:
                start_epoch_s1 = resumed_epoch
                print(f"[QuickVoice] Resumed from Stage 1, epoch {start_epoch_s1+1}") # Print +1 for user friendliness matching log
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
    
    # Handle non-resume extend case (just override total)
    elif args.extend_to_epochs_s1 is not None:
         epochs_s1_total = int(args.extend_to_epochs_s1)

    with JSONLMetricsLogger(metrics_path) as mlog:
        mlog.log(
            {
                "type": "meta",
                "run_name": out_dir.name,
                "config": {
                    **vars(args),
                    "n_models": int(n_models),
                    "num_classes": int(num_classes),
                    "n_classes": int(num_classes),
                },
                "device": str(DEVICE),
                "metrics_path": str(metrics_path),
                "manifest": str(manifest_train_path),
                "manifest_train": str(manifest_train_path),
                "manifest_val": str(manifest_val_path),
                "manifest_test": str(manifest_test_path),
                "n_models": int(n_models),
                "num_classes": int(num_classes),
                "n_classes": int(num_classes),  # backward compat
            }
        )

        if epochs_s1_total > 0:
            if args.s1_arch == "static":
                print(f"\n[Stage 1] Decoder pretraining (multiclass, static), starting from epoch {start_epoch_s1}")
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
                    on_epoch_end=handle_epoch_end_s1,
                    start_epoch=start_epoch_s1,
                    should_stop=should_stop,
                )
            elif args.s1_arch == "adaptive_uncertainty":
                print(f"\n[Stage 1] Decoder pretraining (multiclass, adaptive), starting from epoch {start_epoch_s1}")
                if balancer is None: 
                     raise RuntimeError("Balancer is None in adaptive mode!")
                
                train_stage1_adaptive(
                    decoder,
                    encoder,
                    balancer,
                    loader,
                    DEVICE,
                    epochs=epochs_s1_total,
                    class_weights=None, # explicit
                    # adaptive trainer handles balancing, but consistency weight is still relevant
                    loc_consistency_weight=1.0, # default in stage1.py def, using config value implicitly in function def, but let's leave it to default
                    step_interval=int(args.log_steps_every),
                    on_step=mlog.log,
                    on_epoch_end=handle_epoch_end_s1,
                    start_epoch=start_epoch_s1,
                    should_stop=should_stop,
                )
            else:
                raise ValueError(f"Unknown architecture: {args.s1_arch}")

        if should_stop():
            log_event(
                mlog,
                {
                    "type": "summary",
                    "stage": "control",
                    "epoch": -1,
                    "stopped_early": True,
                    "signal": stop_requested.get("sig"),
                },
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
                step_interval=int(args.log_steps_every),
                on_step=mlog.log,
                on_epoch_end=handle_epoch_end_s2,
                finetune_mode=False,
                start_epoch=start_epoch_s2,
                should_stop=should_stop,
            )

        if should_stop():
            log_event(
                mlog,
                {
                    "type": "summary",
                    "stage": "control",
                    "epoch": -1,
                    "stopped_early": True,
                    "signal": stop_requested.get("sig"),
                },
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
                step_interval=int(args.log_steps_every),
                on_step=mlog.log,
                on_epoch_end=handle_epoch_end_s3,
                finetune_mode=True,
                start_epoch=start_epoch_s3,
                should_stop=should_stop,
            )

        # Final held-out test probe (optional; uses test split, no reverb attack by default).
        try:
            test_dataset = WatermarkDataset(str(manifest_test_path), training=False, n_models=n_models)
            test_n = min(max(0, int(args.probe_clips)), len(test_dataset))
            if test_n > 0:
                test_items: list[ProbeItem] = []
                for i in range(test_n):
                    it = test_dataset[i]
                    test_items.append(ProbeItem(audio=it["audio"].detach().cpu(), y_class=int(it["y_class"].item())))
                test_metrics = compute_probe_metrics(
                    test_items,
                    encoder=encoder,
                    decoder=decoder,
                    device=DEVICE,
                    compute_reverb=True,
                    extra_attacks=[a.strip() for a in str(args.test_attacks).split(",") if a.strip()],
                    num_classes=num_classes,
                )
                log_event(mlog, {"type": "test_probe", "stage": "test", "epoch": -1, **test_metrics})
                print(
                    f"[QuickVoice] Test probe (n={test_n}): "
                    f"wm_acc={test_metrics.get('wm_acc')} "
                    f"tpr_at_fpr_1pct={test_metrics.get('tpr_at_fpr_1pct')} "
                    f"id_acc_pos={test_metrics.get('id_acc_pos')} "
                    f"tpr_at_fpr_1pct_reverb={test_metrics.get('tpr_at_fpr_1pct_reverb')} "
                    f"id_acc_pos_reverb={test_metrics.get('id_acc_pos_reverb')}"
                )
        except Exception as e:
            print(f"[QuickVoice] Test probe skipped: {e}")

    # Export weights for hub consumption.
    # IMPORTANT: Prefer exporting the checkpoint manager's BEST weights so `encoder.pt` / `decoder.pt`
    # reflect the selected metric (and aren't overwritten by the final epoch).
    best_ckpt_path = (Path(ckpt_manager.checkpoints_dir) / "best.pt").resolve()
    if bool(args.save_best) and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        encoder_state_cpu = ckpt.get("encoder")
        decoder_state_cpu = ckpt.get("decoder")
        balancer_state_cpu = ckpt.get("balancer_state")
        print(f"[QuickVoice] Exporting BEST weights from: {best_ckpt_path}")
    else:
        encoder_state_cpu = {k: v.detach().to("cpu") for k, v in encoder.state_dict().items()}
        decoder_state_cpu = {k: v.detach().to("cpu") for k, v in decoder.state_dict().items()}
        balancer_state_cpu = {k: v.detach().to("cpu") for k, v in balancer.state_dict().items()} if balancer else None
        print("[QuickVoice] Exporting LAST weights (no best checkpoint found)")

    # Atomic save to prevent corruption on crash
    temp_encoder_path = str(out_dir / "encoder.pt") + ".tmp"
    temp_decoder_path = str(out_dir / "decoder.pt") + ".tmp"

    torch.save(encoder_state_cpu, temp_encoder_path)
    os.replace(temp_encoder_path, str(out_dir / "encoder.pt"))

    torch.save(decoder_state_cpu, temp_decoder_path)
    os.replace(temp_decoder_path, str(out_dir / "decoder.pt"))

    if balancer_state_cpu:
        temp_bal_path = str(out_dir / "balancer.pt") + ".tmp"
        torch.save(balancer_state_cpu, temp_bal_path)
        os.replace(temp_bal_path, str(out_dir / "balancer.pt"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
