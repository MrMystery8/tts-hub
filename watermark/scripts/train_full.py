#!/usr/bin/env python3
"""
Complete training pipeline for audio watermarking.
Usage: python -m watermark.scripts.train_full --manifest /path/to/manifest.json --output ./checkpoints
"""
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Optional

from watermark.config import DEVICE, SAMPLE_RATE, SEGMENT_SAMPLES
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2
from watermark.utils.metrics_logger import JSONLMetricsLogger
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics


def main():
    parser = argparse.ArgumentParser(description="Watermark Training Pipeline")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    parser.add_argument("--output", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs_s1", type=int, default=20, help="Stage 1 epochs")
    parser.add_argument("--epochs_s1b", type=int, default=10, help="Stage 1B epochs")
    parser.add_argument("--warmup_s1b", type=int, default=3, help="Stage 1B warmup epochs")
    parser.add_argument("--epochs_s2", type=int, default=20, help="Stage 2 epochs")
    parser.add_argument("--epochs_s1b_post", type=int, default=0, help="Optional Stage 1B fine-tune AFTER Stage 2")
    parser.add_argument("--neg_weight", type=float, default=0.4, help="Stage 1B negative preamble weight")
    parser.add_argument("--neg_preamble_target", type=float, default=0.5, help="Stage 1B negative preamble target prob")
    parser.add_argument("--unknown_ce_weight", type=float, default=0.0, help="Stage 1B unknown-class CE weight (negatives)")
    parser.add_argument("--s1b_heads_only", action="store_true", help="Stage 1B: train only message/id heads (freeze backbone/det head)")
    parser.add_argument("--model_ce_weight", type=float, default=1.0, help="Stage 1B/2 model_id CE weight")
    parser.add_argument("--version_ce_weight", type=float, default=1.0, help="Stage 1B/2 version CE weight")
    parser.add_argument("--pair_ce_weight", type=float, default=2.0, help="Stage 1B/2 (model_id,version) joint CE weight")
    parser.add_argument("--msg_weight", type=float, default=1.0, help="Stage 2 message loss weight")
    parser.add_argument(
        "--stage2_payload_on_all",
        action="store_true",
        help="Stage 2: apply payload/ID losses on all carriers (ignores has_watermark gating)",
    )
    parser.add_argument("--reverb_prob", type=float, default=0.25, help="Stage 2 differentiable reverb probability")
    parser.add_argument("--log_metrics", type=str, default=None, help="Write JSONL metrics for live dashboard")
    parser.add_argument("--probe_every", type=int, default=1, help="Run probe every N epochs (Stage 2 + post-Stage1B)")
    parser.add_argument("--probe_clips", type=int, default=256, help="Number of probe clips (cached in RAM once)")
    parser.add_argument("--probe_reverb_every", type=int, default=1, help="Compute reverb probe every N probe runs")
    parser.add_argument("--log_steps_every", type=int, default=100, help="Log step events every N batches (0 disables)")
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
    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Probe cache (center crop, deterministic)
    probe_items: list[ProbeItem] = []
    probe_dataset = WatermarkDataset(args.manifest, codec=codec, training=False)
    probe_n = min(max(0, int(args.probe_clips)), len(probe_dataset))
    for i in range(probe_n):
        it = probe_dataset[i]
        probe_items.append(
            ProbeItem(
                audio=it["audio"].detach().cpu(),
                has_watermark=float(it["has_watermark"].item()),
                message=it["message"].detach().cpu(),
                model_id=int(it["model_id"].item()),
                version=int(it["version"].item()),
            )
        )

    metrics_path = Path(args.log_metrics) if args.log_metrics else (output_dir / "metrics.jsonl")
    targets = {
        "mini_auc": 0.95,
        "mini_auc_reverb": 0.85,
        "tpr_at_fpr_1pct": 0.90,
        "tpr_at_fpr_1pct_reverb": 0.70,
        "preamble_pos_avg": 0.95,
        "preamble_neg_avg": 0.60,
        "model_id_acc_cls": 0.60,
        "version_acc_cls": 0.40,
        "pair_acc_cls": 0.30,
        "payload_exact_acc_cls": 0.30,
        "model_unknown_rate": 0.10,
        "version_unknown_rate": 0.10,
    }

    probe_every = max(1, int(args.probe_every))
    probe_reverb_every = max(1, int(args.probe_reverb_every))
    probe_counter = {"n": 0}

    best_by_payload: dict[str, dict[str, float]] = {
        "s2": {"payload_exact_acc_cls": -1.0, "epoch": 0.0},
        "s1b_post": {"payload_exact_acc_cls": -1.0, "epoch": 0.0},
    }

    def maybe_probe(logger: JSONLMetricsLogger, *, stage: str, epoch: int) -> Optional[dict[str, Any]]:
        if probe_n == 0:
            return None
        if (epoch % probe_every) != 0:
            return None
        probe_counter["n"] += 1
        do_reverb = (probe_counter["n"] % probe_reverb_every) == 0
        m = compute_probe_metrics(
            probe_items,
            encoder=encoder,
            decoder=decoder,
            codec=codec,
            device=DEVICE,
            compute_reverb=do_reverb,
        )
        logger.log({"type": "probe", "stage": stage, "epoch": epoch, **m})
        return m

    def maybe_save_best(*, stage: str, epoch: int, metrics: Optional[dict[str, Any]]) -> None:
        if not metrics:
            return
        # Prefer real-use conditional attribution (detect -> then decode).
        # Fall back to unconditional metrics if cond metrics are missing.
        key_candidates = [
            "pair_acc_cls_cond_1pct",
            "payload_exact_acc_cls_cond_1pct",
            "pair_acc_cls",
            "payload_exact_acc_cls",
        ]
        key = None
        cur = None
        for k in key_candidates:
            v = metrics.get(k, None)
            if isinstance(v, (int, float)):
                key = k
                cur = float(v)
                break
        if key is None or cur is None:
            return

        prev = float(best_by_payload.get(stage, {}).get(key, -1.0))
        if cur <= prev:
            return
        best_by_payload[stage] = {key: cur, "epoch": float(epoch)}
        if stage == "s2":
            torch.save(encoder.state_dict(), output_dir / "encoder_stage2_best.pt")
        elif stage == "s1b_post":
            torch.save(decoder.state_dict(), output_dir / "decoder_stage1b_post_best.pt")

    with JSONLMetricsLogger(metrics_path) as mlog:
        mlog.log(
            {
                "type": "meta",
                "run_name": "train_full",
                "device": str(DEVICE),
                "metrics_path": str(metrics_path),
                "output_dir": str(output_dir),
                "manifest": str(args.manifest),
                "targets": targets,
                "config": {
                    "epochs_s1": args.epochs_s1,
                    "epochs_s1b": args.epochs_s1b,
                    "warmup_s1b": args.warmup_s1b,
                    "epochs_s2": args.epochs_s2,
                    "epochs_s1b_post": args.epochs_s1b_post,
                    "neg_weight": args.neg_weight,
                    "neg_preamble_target": args.neg_preamble_target,
                    "unknown_ce_weight": args.unknown_ce_weight,
                    "s1b_heads_only": bool(args.s1b_heads_only),
                    "model_ce_weight": args.model_ce_weight,
                    "version_ce_weight": args.version_ce_weight,
                    "pair_ce_weight": args.pair_ce_weight,
                    "msg_weight": args.msg_weight,
                    "stage2_payload_on_all": bool(args.stage2_payload_on_all),
                    "reverb_prob": args.reverb_prob,
                    "probe_n": probe_n,
                    "probe_every": probe_every,
                    "probe_reverb_every": probe_reverb_every,
                    "log_steps_every": int(args.log_steps_every),
                    "batch_size": int(batch_size),
                },
            }
        )

        # === STAGE 1: Detection ===
        print("\n=== Stage 1: Detection Training ===")
        train_stage1(
            decoder,
            encoder,
            loader,
            DEVICE,
            epochs=args.epochs_s1,
            step_interval=int(args.log_steps_every),
            on_step=mlog.log,
            on_epoch_end=mlog.log,
        )
        torch.save(decoder.state_dict(), output_dir / "decoder_stage1.pt")
        
        # === STAGE 1B: Payload (with curriculum) ===
        print("\n=== Stage 1B: Payload Training ===")
        train_stage1b(
            decoder,
            encoder,
            loader,
            DEVICE,
            preamble=codec.preamble,
            stage="s1b",
            heads_only=bool(args.s1b_heads_only),
            epochs=args.epochs_s1b,
            warmup=args.warmup_s1b,
            neg_weight=args.neg_weight,
            neg_preamble_target=args.neg_preamble_target,
            unknown_ce_weight=args.unknown_ce_weight,
            model_ce_weight=args.model_ce_weight,
            version_ce_weight=args.version_ce_weight,
            pair_ce_weight=args.pair_ce_weight,
            step_interval=int(args.log_steps_every),
            on_step=mlog.log,
            on_epoch_end=mlog.log,
        )
        torch.save(decoder.state_dict(), output_dir / "decoder_stage1b.pt")
        
        # === STAGE 2: Encoder ===
        print("\n=== Stage 2: Encoder Training ===")
        train_stage2(
            encoder,
            decoder,
            loader,
            DEVICE,
            epochs=args.epochs_s2,
            msg_weight=args.msg_weight,
            model_ce_weight=args.model_ce_weight,
            version_ce_weight=args.version_ce_weight,
            pair_ce_weight=args.pair_ce_weight,
            reverb_prob=args.reverb_prob,
            payload_pos_only=(not bool(args.stage2_payload_on_all)),
            step_interval=int(args.log_steps_every),
            on_step=mlog.log,
            on_epoch_end=lambda e: (
                mlog.log(e),
                maybe_save_best(
                    stage="s2",
                    epoch=e["epoch"],
                    metrics=maybe_probe(mlog, stage="s2", epoch=e["epoch"]),
                ),
            ),
        )
        torch.save(encoder.state_dict(), output_dir / "encoder_stage2.pt")

        if args.epochs_s1b_post and args.epochs_s1b_post > 0:
            print("\n=== Stage 1B: Post Stage 2 Fine-tune ===")
            # Keep this as a heads-only fit by running Stage1B entirely in warmup mode.
            # This preserves Stage 1 detection behavior while adapting the attribution heads.
            train_stage1b(
                decoder,
                encoder,
                loader,
                DEVICE,
                preamble=codec.preamble,
                stage="s1b_post",
                epochs=args.epochs_s1b_post,
                warmup=args.epochs_s1b_post,
                neg_weight=args.neg_weight,
                neg_preamble_target=args.neg_preamble_target,
                unknown_ce_weight=args.unknown_ce_weight,
                model_ce_weight=args.model_ce_weight,
                version_ce_weight=args.version_ce_weight,
                pair_ce_weight=args.pair_ce_weight,
                step_interval=int(args.log_steps_every),
                on_step=mlog.log,
                on_epoch_end=lambda e: (
                    mlog.log(e),
                    maybe_save_best(
                        stage="s1b_post",
                        epoch=e["epoch"],
                        metrics=maybe_probe(mlog, stage="s1b_post", epoch=e["epoch"]),
                    ),
                ),
            )
            torch.save(decoder.state_dict(), output_dir / "decoder_stage1b_post.pt")

        if probe_n > 0:
            m = compute_probe_metrics(
                probe_items,
                encoder=encoder,
                decoder=decoder,
                codec=codec,
                device=DEVICE,
                compute_reverb=True,
            )
            mlog.log({"type": "probe", "stage": "final", "epoch": 0, **m})
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
