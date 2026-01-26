# Issue: "Identity Starvation" in Watermark Encoder Training

## Context
We are training a robust audio watermarking system (Encoder-Decoder) on a small dataset (0.5k - 2k clips).
- **Encoder**: Overlap-Add, embeds 32-bit (or 11-bit/23-bit) message.
- **Decoder**: Sliding window, predicts detection (binary) and payload (bits + classification).
- **Protocol**:
    - **Stage 1 (Detection)**: Train Decoder on Preamble-only.
    - **Stage 1B (Payload)**: Train Decoder on full payload using a *Random/Frozen Encoder*.
    - **Stage 2 (Encoder)**: Train Encoder against the *Frozen Decoder*.

## Symptoms
1.  **Identity Accuracy Flatline**: `model_id_acc` stays at random chance (~12.5%) throughout training.
2.  **Detection Dominance**: Detection AUC is perfect (>99%), but payload learning is zero.
3.  **Partial Success (Sanity 1)**: When we ramped detection loss (0->1) in Stage 2, identity accuracy jumped to **25%** (exactly 2x chance, meaning 1 bit learnt).
4.  **Sync Failure (Sanity 2)**: Reducing Preamble (16->4 bits) caused sync jitter and failed detection.
5.  **Current Status (Sanity 3)**: Restored 16-bit Preamble + Reduced Payload (7 bits). Detection is good, but Identity is stuck at chance.

## Diagnosis
The root cause is likely **"Incompetent Judge"**:
- **Stage 1B** (Decoder Pre-training) was running for only **1 Epoch**.
- The Decoder never learned to read the random encoder's noise reliably.
- In **Stage 2**, the Encoder is training against a Frozen Decoder that outputs garbage gradients for the payload.
- Result: Encoder learns detection (easy) but can't minimize payload loss because the target is moving/random.

## Code Fixes Applied (in the dumped files)
1.  **Stage 1B**: Extended to 20 Epochs (in Sanity 4 plan) to "cook" the judge.
2.  **Stage 2**: Added `det_ramp` (Detection Loss Warmup) to prevent gradient drowning.
3.  **Payload Diet**:
    - **Original**: 32 bits (16 Preamble + 16 Identity/Copies).
    - **Current (Balanced Diet)**: 23 bits (16 Preamble + 7 Identity).
    - Removes 9 redundant bits to free up encoder capacity.

## Files Included
1.  `quick_voice_smoke_train.py`: Main script with arguments and loop.
2.  `stage1b.py`: Decoder payload training logic (Modified for 23-bit layout).
3.  `stage2.py`: Encoder training logic (Modified for 23-bit layout + `det_ramp`).

---
START OF FILES


=== FILE: watermark/scripts/quick_voice_smoke_train.py ===
#!/usr/bin/env python3
"""
Quick smoke test using REAL voice clips from a local dataset (e.g. mini_benchmark_data).

Builds a tiny manifest from real audio, runs Stage 1/1B/2 for a few epochs,
and saves clean/watermarked WAVs for listening.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from watermark.config import DEVICE, SAMPLE_RATE
from watermark.models.codec import MessageCodec
from watermark.models.encoder import WatermarkEncoder, OverlapAddEncoder
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder
from watermark.training.dataset import WatermarkDataset, collate_fn
from watermark.training.stage1 import train_stage1
from watermark.training.stage1b import train_stage1b
from watermark.training.stage2 import train_stage2
from watermark.utils.io import load_audio, save_audio
from watermark.evaluation.attacks import ATTACKS, apply_attack_safe
from watermark.utils.metrics_logger import JSONLMetricsLogger
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics


def collect_audio_files(source_dir: Path) -> list[Path]:
    exts = [".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg"]
    files = [p for p in source_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def build_manifest(paths: list[Path], out_dir: Path) -> Path:
    manifest = []
    pos_idx = 0
    for i, p in enumerate(paths):
        is_pos = (i % 2 == 0)
        if is_pos:
            # IMPORTANT:
            # - Use a counter over POSITIVES so we cover all IDs evenly.
            # - Cover the full (model_id, version) grid (8 * 16 = 128 combos).
            #   A common pitfall is `model_id = pos_idx % 8` and `version = pos_idx % 16`,
            #   which correlates them (only 16 unique pairs) and makes `exact` accuracy
            #   look artificially high.
            pair = pos_idx % (8 * 16)
            model_id = pair % 8
            version = (pair // 8) % 16
            pos_idx += 1
        else:
            model_id = None
            version = None
        manifest.append({
            "path": str(p.resolve()),
            "has_watermark": 1 if is_pos else 0,
            "model_id": model_id,
            "version": version,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def summarize_decode(tag: str, audio: torch.Tensor, decoder: SlidingWindowDecoder, codec: MessageCodec) -> str:
    """Run decoder and return a one-line summary."""
    # Ensure audio is on same device as decoder buffers/params
    device = next(decoder.parameters()).device
    audio = audio.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = decoder(audio)
        detect_prob = outputs["clip_detect_prob"]
        if hasattr(detect_prob, "item"):
            detect_prob = detect_prob.item()
        msg_probs = outputs.get("avg_message_probs")
        model_pred = None
        model_conf = None
        if "avg_model_logits" in outputs:
            m = outputs["avg_model_logits"]
            if m.dim() == 2:
                m = m.squeeze(0)
            model_pred = int(torch.argmax(m).item())
            model_conf = float(torch.softmax(m, dim=0)[model_pred].item())

        ver_pred = None
        ver_conf = None
        if "avg_version_logits" in outputs:
            v = outputs["avg_version_logits"]
            if v.dim() == 2:
                v = v.squeeze(0)
            ver_pred = int(torch.argmax(v).item())
            ver_conf = float(torch.softmax(v, dim=0)[ver_pred].item())
        if msg_probs is not None:
            msg_probs = msg_probs.squeeze(0)
            decoded = codec.decode(msg_probs)
            preamble = decoded["preamble_score"]
            model_id_bits = decoded["model_id"]
            version_bits = decoded["version"]
            conf_bits = decoded["confidence"]
            cls_part = ""
            if model_pred is not None and ver_pred is not None:
                cls_part = f", cls_model={model_pred}@{model_conf:.2f}, cls_ver={ver_pred}@{ver_conf:.2f}"
            return (
                f"{tag}: detect_prob={detect_prob:.4f}, preamble={preamble:.2f}, "
                f"bits_model={model_id_bits}, bits_ver={version_bits}, bits_conf={conf_bits:.3f}"
                f"{cls_part}"
            )
        return f"{tag}: detect_prob={detect_prob:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick smoke train on real voice samples")
    parser.add_argument("--source_dir", type=str, default="mini_benchmark_data", help="Root folder with audio")
    parser.add_argument("--out", type=str, default="outputs/quick_voice_smoke", help="Output directory")
    parser.add_argument("--profile", type=str, default="medium", choices=["quick", "medium", "large"], help="Run size preset")
    parser.add_argument("--num_clips", type=int, default=None, help="Number of clips to use")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    parser.add_argument("--log_metrics", type=str, default=None, help="Write JSONL metrics for live dashboard")
    parser.add_argument("--probe_every", type=int, default=1, help="Run probe every N epochs (Stage 2 + post-Stage1B)")
    parser.add_argument("--probe_clips", type=int, default=256, help="Number of probe clips (cached in RAM once)")
    parser.add_argument("--probe_reverb_every", type=int, default=1, help="Compute reverb probe every N probe runs")
    parser.add_argument("--det_ramp", type=int, default=0, help="Ramp up detection loss over N epochs in Stage 2")
    parser.add_argument("--qual_weight", type=float, default=1.0, help="Weight for quality loss in Stage 2")
    parser.add_argument("--log_steps_every", type=int, default=25, help="Log step events every N batches (0 disables)")
    parser.add_argument("--epochs_s1", type=int, default=None, help="Stage 1 epochs")
    parser.add_argument("--epochs_s1b", type=int, default=None, help="Stage 1B epochs")
    parser.add_argument("--epochs_s2", type=int, default=None, help="Stage 2 epochs")
    parser.add_argument("--epochs_s1b_post", type=int, default=0, help="Optional Stage 1B fine-tune AFTER Stage 2")
    parser.add_argument("--neg_weight", type=float, default=None, help="Stage 1B negative preamble weight")
    parser.add_argument("--neg_preamble_target", type=float, default=None, help="Stage 1B negative preamble target prob (default 0.5)")
    parser.add_argument("--unknown_ce_weight", type=float, default=None, help="Stage 1B unknown-class CE weight (negatives)")
    parser.add_argument("--model_ce_weight", type=float, default=None, help="Stage 1B/2 model_id CE weight")
    parser.add_argument("--version_ce_weight", type=float, default=None, help="Stage 1B/2 version CE weight")
    parser.add_argument("--pair_ce_weight", type=float, default=None, help="Stage 1B/2 (model_id,version) joint CE weight")
    parser.add_argument("--msg_weight", type=float, default=None, help="Stage 2 message loss weight")
    parser.add_argument(
        "--stage2_payload_on_all",
        action="store_true",
        help="Stage 2: apply payload/ID losses on all carriers (ignores has_watermark gating)",
    )
    parser.add_argument("--reverb_prob", type=float, default=None, help="Stage 2 differentiable reverb probability")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone/det in Stage 1B")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"[QuickVoice] Source dir not found: {source_dir}")
        return 1

    profiles = {
        "quick": {"num_clips": 32, "epochs_s1": 3, "epochs_s1b": 2, "epochs_s2": 3},
        "medium": {"num_clips": 96, "epochs_s1": 8, "epochs_s1b": 5, "epochs_s2": 8},
        "large": {"num_clips": 160, "epochs_s1": 10, "epochs_s1b": 6, "epochs_s2": 10},
    }
    preset = profiles[args.profile]
    num_clips = args.num_clips if args.num_clips is not None else preset["num_clips"]
    epochs_s1 = args.epochs_s1 if args.epochs_s1 is not None else preset["epochs_s1"]
    epochs_s1b = args.epochs_s1b if args.epochs_s1b is not None else preset["epochs_s1b"]
    epochs_s2 = args.epochs_s2 if args.epochs_s2 is not None else preset["epochs_s2"]
    epochs_s1b_post = args.epochs_s1b_post

    neg_weight = 0.4 if args.neg_weight is None else args.neg_weight
    neg_preamble_target = 0.5 if args.neg_preamble_target is None else args.neg_preamble_target
    unknown_ce_weight = 0.2 if args.unknown_ce_weight is None else args.unknown_ce_weight
    model_ce_weight = 1.0 if args.model_ce_weight is None else args.model_ce_weight
    version_ce_weight = 1.0 if args.version_ce_weight is None else args.version_ce_weight
    pair_ce_weight = 2.0 if args.pair_ce_weight is None else args.pair_ce_weight
    msg_weight = 1.0 if args.msg_weight is None else args.msg_weight
    stage2_payload_on_all = bool(args.stage2_payload_on_all)
    reverb_prob = 0.25 if args.reverb_prob is None else args.reverb_prob

    print(f"[QuickVoice] Using device: {DEVICE}")
    print(
        f"[QuickVoice] Profile: {args.profile} | clips={num_clips} | "
        f"s1={epochs_s1} s1b={epochs_s1b} s2={epochs_s2} s1b_post={epochs_s1b_post} | "
        f"neg_w={neg_weight} neg_pre_tgt={neg_preamble_target} unk_ce_w={unknown_ce_weight} "
        f"model_ce_w={model_ce_weight} ver_ce_w={version_ce_weight} pair_ce_w={pair_ce_weight} | "
        f"msg_w={msg_weight} reverb_p={reverb_prob}"
    )
    audio_files = collect_audio_files(source_dir)
    if not audio_files:
        print(f"[QuickVoice] No audio files found in: {source_dir}")
        return 1

    rng = random.Random(args.seed)
    if num_clips < len(audio_files):
        selected = rng.sample(audio_files, num_clips)
        selected.sort()
    else:
        selected = audio_files

    out_dir = Path(args.out)
    data_dir = out_dir / "data"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = build_manifest(selected, data_dir)
    print(f"[QuickVoice] Manifest: {manifest_path}")
    print(f"[QuickVoice] Clips used: {len(selected)}")

    codec = MessageCodec(key="quick_voice")
    encoder = OverlapAddEncoder(WatermarkEncoder(msg_bits=32)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(msg_bits=32)).to(DEVICE)

    dataset = WatermarkDataset(str(manifest_path), codec=codec, training=True)
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ---- Live metrics + probe cache (optional but default-on via out/metrics.jsonl) ----
    metrics_path = Path(args.log_metrics) if args.log_metrics else (out_dir / "metrics.jsonl")
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

    # Cache a small deterministic probe set (center crop, no random crops).
    probe_items: list[ProbeItem] = []
    probe_dataset = WatermarkDataset(str(manifest_path), codec=codec, training=False)
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

    probe_every = max(1, int(args.probe_every))
    probe_reverb_every = max(1, int(args.probe_reverb_every))
    probe_counter = {"n": 0}

    # State for best checkpoints
    best_det_auc = 0.0
    best_attr_acc = 0.0
    
    def check_and_save_best(metrics, decoder, encoder, out_dir):
        nonlocal best_det_auc, best_attr_acc
        
        # Save Best Detection
        if metrics.get("mini_auc", 0) > best_det_auc:
            best_det_auc = metrics["mini_auc"]
            torch.save(decoder.state_dict(), out_dir / "best_det_decoder.pt")
            torch.save(encoder.state_dict(), out_dir / "best_det_encoder.pt")
            # print(f"    >> New Best Detection AUC: {best_det_auc:.4f} (Saved)")
            
        # Save Best Attribution
        if metrics.get("pair_acc_cls_cond_1pct", 0) > best_attr_acc:
            best_attr_acc = metrics["pair_acc_cls_cond_1pct"]
            torch.save(decoder.state_dict(), out_dir / "best_attr_decoder.pt")
            torch.save(encoder.state_dict(), out_dir / "best_attr_encoder.pt")
            # print(f"    >> New Best Attribution Acc: {best_attr_acc:.4f} (Saved)")

    def log_event(logger: JSONLMetricsLogger, event: dict) -> None:
        logger.log(event)

    def maybe_probe(logger: JSONLMetricsLogger, *, stage: str, epoch: int) -> None:
        if probe_n == 0:
            return
        if (epoch % probe_every) != 0:
            return
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
        log_event(
            logger,
            {
                "type": "probe",
                "stage": stage,
                "epoch": epoch,
                **m,
            },
        )
        check_and_save_best(m, decoder, encoder, out_dir)

    with JSONLMetricsLogger(metrics_path) as mlog:
        log_event(
            mlog,
            {
                "type": "meta",
                "run_name": "quick_voice_smoke_train",
                "ts": time.time(),
                "device": str(DEVICE),
                "metrics_path": str(metrics_path),
                "out_dir": str(out_dir),
                "manifest": str(manifest_path),
                "targets": targets,
                "config": {
                    "profile": args.profile,
                    "num_clips": len(selected),
                    "epochs_s1": epochs_s1,
                    "epochs_s1b": epochs_s1b,
                    "epochs_s2": epochs_s2,
                    "epochs_s1b_post": epochs_s1b_post,
                    "neg_weight": neg_weight,
                    "neg_preamble_target": neg_preamble_target,
                    "unknown_ce_weight": unknown_ce_weight,
                    "model_ce_weight": model_ce_weight,
                    "version_ce_weight": version_ce_weight,
                    "pair_ce_weight": pair_ce_weight,
                    "msg_weight": msg_weight,
                    "reverb_prob": reverb_prob,
                    "stage2_payload_on_all": stage2_payload_on_all,
                    "probe_n": probe_n,
                    "probe_every": probe_every,
                    "probe_reverb_every": probe_reverb_every,
                    "log_steps_every": int(args.log_steps_every),
                    "batch_size": int(batch_size),
                    "steps_per_epoch": int(math.ceil(len(selected) / float(batch_size))),
                },
            },
        )

        print("\n[QuickVoice] Stage 1 (Detection)")
        train_stage1(
            decoder,
            encoder,
            loader,
            DEVICE,
            epochs=epochs_s1,
            log_interval=999,
            step_interval=int(args.log_steps_every),
            on_step=lambda e: log_event(mlog, e),
            on_epoch_end=lambda e: log_event(mlog, e),
        )

        print("\n[QuickVoice] Stage 1B (Payload)")
        train_stage1b(
            decoder,
            encoder,
            loader,
            device=DEVICE,
            preamble=codec.preamble,
            stage="s1b",
            epochs=int(epochs_s1b),
            warmup=1,
            neg_weight=float(neg_weight),
            neg_preamble_target=float(neg_preamble_target),
            unknown_ce_weight=float(unknown_ce_weight),
            model_ce_weight=float(model_ce_weight),
            version_ce_weight=float(version_ce_weight),
            pair_ce_weight=float(pair_ce_weight),
            freeze_backbone=bool(args.freeze_backbone),
            log_interval=int(args.log_steps_every),
            step_interval=1,
            on_step=lambda e: log_event(mlog, e),
            on_epoch_end=lambda e: log_event(mlog, e),
        )

        print("\n[QuickVoice] Stage 2 (Encoder)")
        train_stage2(
            encoder,
            decoder,
            loader,
            DEVICE,
            epochs=epochs_s2,
            msg_weight=msg_weight,
            model_ce_weight=model_ce_weight,
            version_ce_weight=version_ce_weight,
            pair_ce_weight=pair_ce_weight,
            qual_weight=float(args.qual_weight),
            reverb_prob=reverb_prob,
            payload_pos_only=(not stage2_payload_on_all),
            log_interval=999,
            step_interval=int(args.log_steps_every),
            on_step=lambda e: log_event(mlog, e),
            on_epoch_end=lambda e: (log_event(mlog, e), maybe_probe(mlog, stage="s2", epoch=e["epoch"])),
            det_ramp=int(args.det_ramp),
        )

        if epochs_s1b_post and epochs_s1b_post > 0:
            print("\n[QuickVoice] Stage 1B (Payload) - Post Stage 2 fine-tune")
            # ALLOW BACKBONE TO UNFREEZE:
            # We want the heads to align first (warmup), then let the backbone
            # adapt to the encoder's new embedding style.
            post_warmup = min(3, epochs_s1b_post)
            train_stage1b(
                decoder,
                encoder,
                loader,
                DEVICE,
                codec.preamble,
                stage="s1b_post",
                epochs=epochs_s1b_post,
                warmup=post_warmup,
                neg_weight=neg_weight,
                neg_preamble_target=neg_preamble_target,
                unknown_ce_weight=unknown_ce_weight,
                model_ce_weight=model_ce_weight,
                version_ce_weight=version_ce_weight,
                pair_ce_weight=pair_ce_weight,
                log_interval=999,
                step_interval=int(args.log_steps_every),
                on_step=lambda e: log_event(mlog, e),
                on_epoch_end=lambda e: (log_event(mlog, e), maybe_probe(mlog, stage="s1b_post", epoch=e["epoch"])),
            )

        # Final probe snapshot (always with reverb) for the dashboard
        if probe_n > 0:
            m = compute_probe_metrics(
                probe_items,
                encoder=encoder,
                decoder=decoder,
                codec=codec,
                device=DEVICE,
                compute_reverb=True,
            )
            log_event(mlog, {"type": "probe", "stage": "final", "epoch": 0, **m})

    # Save a clean + watermarked sample for listening.
    listen_src = selected[0]
    clean = load_audio(str(listen_src), target_sr=SAMPLE_RATE).unsqueeze(0).to(DEVICE)
    # Prefer a message that is in the training distribution (first POS sample in manifest).
    # If none found, fall back to a fixed message.
    msg_model_id = 3
    msg_version = 1
    for p in probe_items:
        if p.has_watermark == 1.0:
            msg_model_id = p.model_id
            msg_version = p.version
            break
    message = codec.encode(model_id=msg_model_id, version=msg_version).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        watermarked = encoder(clean, message)

    clean_path = audio_dir / "clean.wav"
    wm_path = audio_dir / "watermarked.wav"

    save_audio(str(clean_path), clean.squeeze(0), SAMPLE_RATE)
    save_audio(str(wm_path), watermarked.squeeze(0), SAMPLE_RATE)

    attacked_path = None
    attacked = None
    if "reverb" in ATTACKS:
        attacked = apply_attack_safe(watermarked.squeeze(0).cpu(), ATTACKS["reverb"])
        attacked_path = audio_dir / "watermarked_reverb.wav"
        save_audio(str(attacked_path), attacked, SAMPLE_RATE)

    # Log source used for listening
    source_note = audio_dir / "source.txt"
    source_note.write_text(str(listen_src.resolve()) + "\n", encoding="utf-8")

    # Decode summary for sanity
    decoder.eval()
    report_lines = []
    report_lines.append(f"Expected message: model_id={msg_model_id}, version={msg_version}")
    report_lines.append(summarize_decode("clean", clean, decoder, codec))
    report_lines.append(summarize_decode("watermarked", watermarked, decoder, codec))
    if attacked is not None:
        report_lines.append(
            summarize_decode("watermarked_reverb", attacked.unsqueeze(0), decoder, codec)
        )

    # Mini AUC diagnostic on the small dataset
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        # Run probe on all items
        results_pos = [] # list of (detect_prob, is_pair_correct, is_model_correct, is_ver_correct)
        results_neg = [] # list of detect_prob

        for item in probe_items:
            clip = item.audio.unsqueeze(0).to(DEVICE)  # (1, 1, T)
            msg = item.message.unsqueeze(0).to(DEVICE)
            label = float(item.has_watermark)

            with torch.no_grad():
                # For positives, watermark it. For negatives, use as-is.
                if label > 0.5:
                    wm = encoder(clip, msg)
                    inp = wm
                else:
                    inp = clip
                
                out = decoder(inp)
                det_prob = out["clip_detect_prob"].item()
                
                if label > 0.5:
                    # Check attribution 
                    target_model_id = int(item.model_id)
                    target_version = int(item.version)
                    
                    pair_correct = False
                    model_correct = False
                    ver_correct = False
                    
                    # Use CLS heads if available (preferred)
                    if "avg_pair_logits" in out:
                        p = out["avg_pair_logits"].squeeze(0)
                        pred_pair = int(torch.argmax(p).item())
                        pair_id = target_version * 8 + target_model_id
                        if pred_pair == pair_id:
                            pair_correct = True
                        
                        pred_model = pred_pair % 8
                        pred_ver = pred_pair // 8
                        if pred_model == target_model_id:
                            model_correct = True
                        if pred_ver == target_version:
                            ver_correct = True
                    
                    results_pos.append({
                        "score": det_prob,
                        "pair_correct": pair_correct,
                        "model_correct": model_correct,
                        "ver_correct": ver_correct
                    })
                else:
                    results_neg.append(det_prob)

        # Metrics Calculation
        y_true = [1.0]*len(results_pos) + [0.0]*len(results_neg)
        y_score = [r["score"] for r in results_pos] + results_neg
        
        if len(results_neg) > 0 and len(results_pos) > 0:
            auc = roc_auc_score(y_true, y_score)
            report_lines.append(f"mini_auc={auc:.4f} (on {len(y_true)} clips)")
            
            # Find Threshold @ 1% FPR
            # 1% FPR = 99th percentile of negatives
            sorted_neg = sorted(results_neg)
            # Index for 1% FPR: we want threshold where only 1% of negs are >= threshold
            # i.e., 99% are < threshold.
            idx_1pct = int(len(sorted_neg) * 0.99)
            idx_1pct = min(idx_1pct, len(sorted_neg) - 1)
            thr_1pct = sorted_neg[idx_1pct]
            
            # Report Threshold
            report_lines.append(f"thr_at_fpr_1pct={thr_1pct:.4f}")
            
            # TPR @ 1% FPR
            n_pos_accepted = sum(1 for r in results_pos if r["score"] > thr_1pct)
            tpr_1pct = n_pos_accepted / len(results_pos)
            report_lines.append(f"tpr_at_fpr_1pct={tpr_1pct:.4f}")
            
            # Acceptance Rate (Positives > Thr)
            # This is technically the same as TPR, but explicit labeled for clarity
            report_lines.append(f"accept_rate_pos_at_fpr_1pct={n_pos_accepted/len(results_pos):.4f}")
            
            # Conditional Attribution Accuracy (on positives accepted by detector)
            # "pair_acc_cls_cond_1pct"
            if n_pos_accepted > 0:
                n_pair_correct_cond = sum(1 for r in results_pos if r["score"] > thr_1pct and r["pair_correct"])
                n_model_correct_cond = sum(1 for r in results_pos if r["score"] > thr_1pct and r["model_correct"])
                n_ver_correct_cond = sum(1 for r in results_pos if r["score"] > thr_1pct and r["ver_correct"])
                
                report_lines.append(f"pair_acc_cls_cond_1pct={n_pair_correct_cond/n_pos_accepted:.4f}")
                report_lines.append(f"model_id_acc_cond_1pct={n_model_correct_cond/n_pos_accepted:.4f}")
                report_lines.append(f"version_acc_cond_1pct={n_ver_correct_cond/n_pos_accepted:.4f}")
            else:
                report_lines.append("pair_acc_cls_cond_1pct=0.0000 (0 accepted)")

            # Conditional @ 5% FPR (Less strict)
            idx_5pct = int(len(sorted_neg) * 0.95)
            idx_5pct = min(idx_5pct, len(sorted_neg) - 1)
            thr_5pct = sorted_neg[idx_5pct]
            n_pos_accepted_5 = sum(1 for r in results_pos if r["score"] > thr_5pct)
            
            if n_pos_accepted_5 > 0:
                n_pair_correct_cond_5 = sum(1 for r in results_pos if r["score"] > thr_5pct and r["pair_correct"])
                report_lines.append(f"pair_acc_cls_cond_5pct={n_pair_correct_cond_5/n_pos_accepted_5:.4f}")
            else:
                report_lines.append("pair_acc_cls_cond_5pct=0.0000")

            # Unconditional stats (for backward comp)
            n_pair_correct_all = sum(1 for r in results_pos if r["pair_correct"])
            report_lines.append(f"pair_acc_cls={n_pair_correct_all/len(results_pos):.4f}")

        # Robustness check: AUC under reverb (CPU attack -> decode on model device)
        if "reverb" in ATTACKS:
            y_score_reverb = []
            for item in probe_items:
                clip = item.audio.unsqueeze(0).to(DEVICE)  # (1, 1, T)
                msg = item.message.unsqueeze(0).to(DEVICE)
                label = float(item.has_watermark)

                with torch.no_grad():
                    if label > 0.5:
                        wm = encoder(clip, msg)
                        inp = wm 
                    else:
                        inp = clip

                attacked_ct = apply_attack_safe(inp.squeeze(0).cpu(), ATTACKS["reverb"])  # (1, T) on CPU
                attacked_bt = attacked_ct.unsqueeze(0).to(DEVICE)  # (1, 1, T) on model device

                with torch.no_grad():
                    out = decoder(attacked_bt)
                    y_score_reverb.append(out["clip_detect_prob"].item())

            auc_reverb = roc_auc_score([1.0 if i.has_watermark > 0.5 else 0.0 for i in probe_items], y_score_reverb)
            report_lines.append(f"mini_auc_reverb={auc_reverb:.4f}")
            
            # Simple mean scores
            pos_scores = [s for s, item in zip(y_score_reverb, probe_items) if item.has_watermark > 0.5]
            neg_scores = [s for s, item in zip(y_score_reverb, probe_items) if item.has_watermark < 0.5]
            if pos_scores:
                report_lines.append(f"reverb_pos_mean={sum(pos_scores)/len(pos_scores):.4f}")
            if neg_scores:
                report_lines.append(f"reverb_neg_mean={sum(neg_scores)/len(neg_scores):.4f}")
    except Exception as e:
        report_lines.append(f"mini_auc=error ({e})")

    report_path = audio_dir / "decode_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n[QuickVoice] Saved audio for listening:")
    print(f"  - Clean: {clean_path}")
    print(f"  - Watermarked: {wm_path}")
    if attacked_path:
        print(f"  - Watermarked + Reverb: {attacked_path}")
    print(f"  - Decode report: {report_path}")
    if auc is not None:
        print(f"  - Mini AUC: {auc:.4f}")
    print(f"  - Source file: {listen_src}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


=== FILE: watermark/training/stage1b.py ===
"""
Stage 1B: Payload Training (Curriculum)

Trains the decoder to recover payload bits.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.3.

CORRECTION (On-the-fly Embedding):
Dataset yields CLEAN audio. We apply watermark on-the-fly to samples.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional

from watermark.config import N_MODELS, N_VERSIONS


def _sample_pair_ids(
    batch_size: int,
    *,
    n_models: int,
    n_versions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_pairs = int(n_models) * int(n_versions)
    pair_cpu = torch.randint(0, n_pairs, (int(batch_size),), device="cpu", dtype=torch.long)
    model_id = (pair_cpu % int(n_models)).to(device=device)
    version = (pair_cpu // int(n_models)).to(device=device)
    return model_id, version


def _encode_message_batch(
    *,
    preamble_16: torch.Tensor,
    model_id: torch.Tensor,
    version: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized MessageCodec.encode() for a batch (32-bit format).
    """
    if preamble_16.dim() == 2:
        pre = preamble_16.to(dtype=torch.float32)
        if pre.shape != (1, 16):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = pre.expand(int(model_id.shape[0]), -1)
    elif preamble_16.dim() == 1:
        if preamble_16.shape != (16,):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = preamble_16.to(dtype=torch.float32).view(1, 16).expand(int(model_id.shape[0]), -1)
    else:
        raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")

    msg = torch.zeros((int(model_id.shape[0]), 32), device=model_id.device, dtype=torch.float32)
    # Balanced Diet: 16-bit preamble (Robust Sync)
    msg[:, 0:16] = pre.to(device=model_id.device)

    # Shifted Identity (start at bit 16)
    for i in range(3):
        msg[:, 16 + i] = ((model_id >> i) & 1).to(torch.float32)
    for i in range(4):
        msg[:, 19 + i] = ((version >> i) & 1).to(torch.float32)

    # No copies
    return msg


def compute_preamble_log_likelihood(msg_probs: torch.Tensor, preamble: torch.Tensor) -> torch.Tensor:
    """
    Log-likelihood for preamble selection (not hard match!).
    
    Args:
        msg_probs: (B, n_win, 32) probabilities
        preamble: (16,) preamble tensor
        
    Returns:
        (B, n_win) log likelihood scores
    """
    B, n_win, _ = msg_probs.shape
    preamble_probs = msg_probs[:, :, :16]
    preamble_exp = preamble.view(1, 1, 16).expand(B, n_win, -1)
    
    eps = 1e-7
    p = torch.clamp(preamble_probs, eps, 1 - eps)
    
    # If preamble bit is 1, add log(p), else log(1-p)
    ll = torch.where(preamble_exp == 1, torch.log(p), torch.log(1 - p))
    
    return ll.sum(dim=2)


def train_stage1b(
    decoder: torch.nn.Module,
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    preamble: torch.Tensor,
    *,
    stage: str = "s1b",
    heads_only: bool = False,
    freeze_backbone: bool = False,
    epochs: int = 10,
    warmup: int = 3,
    top_k: int = 3,
    lr: float = 1e-4,
    neg_weight: float = 0.4,
    neg_preamble_target: float = 0.5,
    unknown_ce_weight: float = 0.2,
    model_ce_weight: float = 1.0,
    version_ce_weight: float = 1.0,
    pair_ce_weight: float = 2.0,
    n_models: int = N_MODELS,
    n_versions: int = N_VERSIONS,
    preamble_weight: float = 0.2,
    model_id_weight: float = 2.0,
    version_weight: float = 0.5,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
):
    """
    Train payload with curriculum:
    - Warmup: use preamble correlation (detector not trusted yet)
    - After: use detect prob
    
    Applies encoder only to watermaked samples (which is all samples in Stage 1B context 
    if we filter correctly, but here we filter has_wm from mixed batch).
    """
    print(f"Starting Stage 1B: Payload Training for {epochs} epochs (Warmup: {warmup})")
    
    decoder.train()
    encoder.eval() # Encoder frozen
    preamble = preamble.to(device)

    # Bitwise weights (avoid constant bits dominating payload learning).
    # Layout (32 bits):
    # - 0:16 preamble
    # - 16:19 model_id
    # - 19:23 version
    # - 23:26 model_id copy
    # - 26:30 version copy
    # - 30:32 reserved (ignored)
    msg_weights = torch.zeros(32, device=device)
    # Balanced Diet Layout:
    # 0-16: Preamble
    # 16-19: Model ID
    # 19-23: Version ID
    msg_weights[0:16] = preamble_weight
    msg_weights[16:19] = model_id_weight
    msg_weights[19:23] = version_weight
    # Rest are 0.0 (ignored)

    weight_sum = msg_weights.sum().clamp(min=1e-8)
    
    for epoch in range(epochs):
        in_warmup = epoch < warmup
        
        # --- Optimizer & Freeze Logic ---
        if epoch == 0 or epoch == warmup:
            # logic:
            # 1. if heads_only=True, we freeze backbone/det regardless of epoch
            # 2. if freeze_backbone=True, we freeze backbone/det regardless of epoch (often used with heads_only)
            # 3. if in_warmup=True, we default to freezing unless overruled? 
            # Actually, standard flow: warmup -> full.
            # But the new "Snap" strategy might request permanent freezing for the entire call.
            
            should_freeze = (in_warmup or heads_only or freeze_backbone)
            
            if should_freeze:
                if in_warmup:
                    print(">> WARMUP PHASE: Training message head only")
                elif heads_only or freeze_backbone:
                    print(">> LOCKED BACKBONE PHASE: Training attribution heads only")
                
                # Freeze everything EXCEPT message/id heads
                for n, p in decoder.named_parameters():
                    is_attr_head = any(h in n for h in ("head_message", "head_model", "head_version", "head_pair"))
                    if is_attr_head:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                
                # CRITICAL: Force frozen modules to eval mode to stop BN updates!
                if hasattr(decoder, "backbone"):
                    decoder.backbone.eval()
                if hasattr(decoder, "head_detect"):
                    decoder.head_detect.eval()
                # Ensure attribution heads are in train mode
                if hasattr(decoder, "head_message"):
                    decoder.head_message.train()
                if hasattr(decoder, "head_model"):
                    decoder.head_model.train()
                if hasattr(decoder, "head_version"):
                    decoder.head_version.train()
                if hasattr(decoder, "head_pair"):
                    decoder.head_pair.train()
            else:
                print(">> NORMAL PHASE: Unfreezing all parameters")
                decoder.train() # Set all to train
                for p in decoder.parameters():
                    p.requires_grad = True
            
            # Recreate optimizer for currently trainable params
            trainable = [p for p in decoder.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=lr)
        
        total_loss = 0.0
        total_loss_msg_bits = 0.0
        total_loss_model_ce = 0.0
        total_loss_version_ce = 0.0
        total_loss_pair_ce = 0.0
        total_loss_neg_preamble = 0.0
        total_loss_unknown_model_ce = 0.0
        total_loss_unknown_version_ce = 0.0
        batches = 0
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None
        
        for i, batch in enumerate(loader):
            has_wm = batch["has_watermark"].bool()
            pos_mask = has_wm
            neg_mask = ~has_wm
            loss = None
            loss_msg_bits = None
            loss_model_ce = None
            loss_version_ce = None
            loss_pair_ce = None
            loss_neg_preamble = None
            loss_unk_model_ce = None
            loss_unk_version_ce = None

            # --- Positive samples (watermarked) ---
            if pos_mask.any():
                audio = batch["audio"][pos_mask].to(device)
                batch_model = batch["model_id"][pos_mask].to(device)
                batch_ver = batch["version"][pos_mask].to(device)
                ids_valid = (batch_model >= 0) & (batch_ver >= 0)
                if (~ids_valid).any():
                    rand_model, rand_ver = _sample_pair_ids(
                        int(batch_model.shape[0]),
                        n_models=n_models,
                        n_versions=n_versions,
                        device=device,
                    )
                    batch_model = torch.where(ids_valid, batch_model, rand_model)
                    batch_ver = torch.where(ids_valid, batch_ver, rand_ver)

                # Keep message encoding on CPU for MPS safety, then move to device.
                message = _encode_message_batch(
                    preamble_16=preamble[:16].to(device="cpu", dtype=torch.float32),
                    model_id=batch_model.to(device="cpu", dtype=torch.long),
                    version=batch_ver.to(device="cpu", dtype=torch.long),
                ).to(device=device, dtype=torch.float32)
            
                # === On-the-fly Embedding ===
                with torch.no_grad():
                    watermarked_audio = encoder(audio, message).detach()

                outputs = decoder(watermarked_audio)

                if in_warmup:
                    ll = compute_preamble_log_likelihood(
                        outputs["all_message_probs"], preamble
                    )
                    _, top_idx = torch.topk(ll, min(top_k, ll.shape[1]), dim=1)
                else:
                    _, top_idx = torch.topk(
                        outputs["all_window_probs"],
                        min(top_k, outputs["all_window_probs"].shape[1]),
                        dim=1
                    )

                msg_logits = outputs["all_message_logits"]  # (B, n_win, 32)
                B, n_win, bits = msg_logits.shape

                idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, bits)
                selected = torch.gather(msg_logits, 1, idx_exp)
                avg_logits = selected.mean(dim=1)

                per_bit = F.binary_cross_entropy_with_logits(avg_logits, message, reduction="none")
                loss_msg_bits = ((per_bit * msg_weights).sum(dim=1) / weight_sum).mean()
                loss = loss_msg_bits

                # Classification heads (preferable for attribution vs bitwise decode)
                model_logits = outputs.get("all_model_logits")
                if model_logits is not None:
                    n_classes = model_logits.shape[-1]
                    idx_model = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                    top_model_logits = torch.gather(model_logits, 1, idx_model).mean(dim=1)
                    loss_model_ce = F.cross_entropy(top_model_logits, batch_model)
                    loss = loss + model_ce_weight * loss_model_ce

                version_logits = outputs.get("all_version_logits")
                if version_logits is not None:
                    n_classes = version_logits.shape[-1]
                    idx_ver = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                    top_ver_logits = torch.gather(version_logits, 1, idx_ver).mean(dim=1)
                    loss_version_ce = F.cross_entropy(top_ver_logits, batch_ver)
                    loss = loss + version_ce_weight * loss_version_ce

                pair_logits = outputs.get("all_pair_logits")
                if pair_logits is not None and pair_ce_weight > 0:
                    n_classes = pair_logits.shape[-1]
                    idx_pair = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                    top_pair_logits = torch.gather(pair_logits, 1, idx_pair).mean(dim=1)
                    pair_id = (batch_ver * int(n_models) + batch_model).to(dtype=torch.long)
                    loss_pair_ce = F.cross_entropy(top_pair_logits, pair_id)
                    loss = loss + pair_ce_weight * loss_pair_ce

            # --- Negative samples (clean) ---
            if neg_mask.any() and neg_weight > 0:
                neg_audio = batch["audio"][neg_mask].to(device)
                outputs_neg = decoder(neg_audio)

                # Penalize preamble confidence on negatives (reduce false preamble hits)
                msg_logits_neg = outputs_neg["all_message_logits"]  # (B, n_win, 32)
                Bn, n_win, bits = msg_logits_neg.shape
                ll_neg = compute_preamble_log_likelihood(
                    outputs_neg["all_message_probs"], preamble
                )
                _, top_idx_neg = torch.topk(
                    ll_neg,
                    min(top_k, ll_neg.shape[1]),
                    dim=1
                )
                idx_exp_neg = top_idx_neg.unsqueeze(-1).expand(-1, -1, bits)
                selected_neg = torch.gather(msg_logits_neg, 1, idx_exp_neg)
                avg_logits_neg = selected_neg.mean(dim=1)

                neg_preamble_logits = avg_logits_neg[:, :16]
                neg_target = torch.full_like(neg_preamble_logits, float(neg_preamble_target))
                loss_neg_preamble = F.binary_cross_entropy_with_logits(neg_preamble_logits, neg_target)

                if loss is None:
                    loss = neg_weight * loss_neg_preamble
                else:
                    loss = loss + neg_weight * loss_neg_preamble

                # Encourage "unknown" attribution on clean audio (optional).
                if unknown_ce_weight > 0:
                    model_logits_neg = outputs_neg.get("all_model_logits")
                    if model_logits_neg is not None:
                        n_classes = model_logits_neg.shape[-1]
                        idx_model_neg = top_idx_neg.unsqueeze(-1).expand(-1, -1, n_classes)
                        top_model_logits_neg = torch.gather(model_logits_neg, 1, idx_model_neg).mean(dim=1)
                        unk = torch.full((top_model_logits_neg.shape[0],), n_classes - 1, device=device, dtype=torch.long)
                        loss_unk_model_ce = F.cross_entropy(top_model_logits_neg, unk)
                        loss = loss + unknown_ce_weight * loss_unk_model_ce

                    version_logits_neg = outputs_neg.get("all_version_logits")
                    if version_logits_neg is not None:
                        n_classes = version_logits_neg.shape[-1]
                        idx_ver_neg = top_idx_neg.unsqueeze(-1).expand(-1, -1, n_classes)
                        top_ver_logits_neg = torch.gather(version_logits_neg, 1, idx_ver_neg).mean(dim=1)
                        unk = torch.full((top_ver_logits_neg.shape[0],), n_classes - 1, device=device, dtype=torch.long)
                        loss_unk_version_ce = F.cross_entropy(top_ver_logits_neg, unk)
                        loss = loss + unknown_ce_weight * loss_unk_version_ce

            if loss is None:
                continue
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            if loss_msg_bits is not None:
                total_loss_msg_bits += loss_msg_bits.item()
            if loss_model_ce is not None:
                total_loss_model_ce += loss_model_ce.item()
            if loss_version_ce is not None:
                total_loss_version_ce += loss_version_ce.item()
            if loss_pair_ce is not None:
                total_loss_pair_ce += loss_pair_ce.item()
            if loss_neg_preamble is not None:
                total_loss_neg_preamble += loss_neg_preamble.item()
            if loss_unk_model_ce is not None:
                total_loss_unknown_model_ce += loss_unk_model_ce.item()
            if loss_unk_version_ce is not None:
                total_loss_unknown_version_ce += loss_unk_version_ce.item()
            batches += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                on_step(
                    {
                        "type": "step",
                        "stage": str(stage),
                        "epoch": epoch + 1,
                        "batch": int(i),
                        "n_batches": n_batches,
                        "in_warmup": bool(in_warmup),
                        "loss": float(loss.item()) if loss is not None else None,
                        "loss_msg_bits": float(loss_msg_bits.item()) if loss_msg_bits is not None else None,
                        "loss_model_ce": float(loss_model_ce.item()) if loss_model_ce is not None else None,
                        "loss_version_ce": float(loss_version_ce.item()) if loss_version_ce is not None else None,
                        "loss_pair_ce": float(loss_pair_ce.item()) if loss_pair_ce is not None else None,
                        "loss_neg_preamble": float(loss_neg_preamble.item()) if loss_neg_preamble is not None else None,
                        "loss_unknown_model_ce": float(loss_unk_model_ce.item()) if loss_unk_model_ce is not None else None,
                        "loss_unknown_version_ce": float(loss_unk_version_ce.item()) if loss_unk_version_ce is not None else None,
                        "n_pos": int(pos_mask.sum().item()),
                        "n_neg": int(neg_mask.sum().item()),
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        if on_epoch_end is not None:
            denom = max(1, batches)
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": str(stage),
                    "epoch": epoch + 1,
                    "in_warmup": bool(in_warmup),
                    "loss": avg_loss,
                    "loss_msg_bits": total_loss_msg_bits / denom,
                    "loss_model_ce": total_loss_model_ce / denom,
                    "loss_version_ce": total_loss_version_ce / denom,
                    "loss_pair_ce": total_loss_pair_ce / denom,
                    "loss_neg_preamble": total_loss_neg_preamble / denom,
                    "loss_unknown_model_ce": total_loss_unknown_model_ce / denom,
                    "loss_unknown_version_ce": total_loss_unknown_version_ce / denom,
                    "neg_weight": float(neg_weight),
                    "neg_preamble_target": float(neg_preamble_target),
                    "unknown_ce_weight": float(unknown_ce_weight),
                    "model_ce_weight": float(model_ce_weight),
                    "version_ce_weight": float(version_ce_weight),
                    "pair_ce_weight": float(pair_ce_weight),
                    "lr": opt.param_groups[0].get("lr"),
                }
            )

    # Ensure everything is unfrozen at end
    for p in decoder.parameters():
        p.requires_grad = True


=== FILE: watermark/training/stage2.py ===
"""
Stage 2: Encoder Training

Trains the encoder to embed watermarks that obey the decoder,
while minimizing audio degradation.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.4.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from typing import Callable, Optional

from watermark.training.losses import CachedSTFTLoss
from watermark.config import SAMPLE_RATE


class DifferentiableAugmenter:
    """Only transforms that preserve gradient flow."""
    def __init__(self, device: torch.device, sample_rate: int = SAMPLE_RATE, reverb_prob: float = 0.25):
        self.device = device
        self.sample_rate = sample_rate
        self.reverb_prob = reverb_prob

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.reverb_prob > 0 and random.random() < self.reverb_prob:
            return self.apply_reverb(audio)
        transform = random.choice([self.identity, self.add_noise, self.apply_eq, self.volume_change])
        return transform(audio)
    
    def identity(self, x): return x
    
    def add_noise(self, x, snr=25):
        power = x.pow(2).mean()
        noise = torch.randn_like(x) * (power / 10**(snr/10)).sqrt()
        return x + noise
    
    def apply_eq(self, x):
        k = random.choice([3, 5, 7])
        kernel = torch.ones(1, 1, k, device=self.device) / k
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Apply conv and same pad
        out = F.conv1d(x, kernel, padding=k//2)
        if out.shape[-1] != x.shape[-1]:
             out = out[..., :x.shape[-1]]
        return out.squeeze(1)
    
    def volume_change(self, x):
        db = random.uniform(-6, 6)
        return x * 10**(db/20)

    def apply_reverb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable reverb approximation using time-domain convolution.
        Expects x as (B, T). Preserves length.
        """
        if x.dim() != 2:
            raise ValueError(f"apply_reverb expects (B, T), got {x.shape}")

        B, T = x.shape
        rir_seconds = random.uniform(0.05, 0.25)
        rir_len = max(64, int(self.sample_rate * rir_seconds))

        t = torch.linspace(0, 1, rir_len, device=self.device)
        decay_rate = random.uniform(4.0, 10.0)
        decay = torch.exp(-decay_rate * t)

        rir = torch.randn(rir_len, device=self.device) * decay
        rir[0] = rir[0] + 1.0  # direct path
        rir = rir / (rir.abs().sum() + 1e-6)

        y = F.conv1d(x.unsqueeze(1), rir.view(1, 1, -1), padding=rir_len - 1)
        y = y[..., :T].squeeze(1)

        wet = random.uniform(0.2, 0.6)
        return (1 - wet) * x + wet * y


def sample_pair_ids(
    batch_size: int,
    *,
    n_models: int,
    n_versions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Uniform sampling over (model_id, version) pairs.

    Returns:
        model_id: (B,) int64 in [0, n_models-1]
        version: (B,) int64 in [0, n_versions-1]
        pair_id: (B,) int64 in [0, n_models*n_versions-1]
    """
    # Keep sampling on CPU for MPS safety; move to target device after.
    n_pairs = int(n_models) * int(n_versions)
    pair_cpu = torch.randint(0, n_pairs, (int(batch_size),), device="cpu", dtype=torch.long)
    model_id = (pair_cpu % int(n_models)).to(device=device)
    version = (pair_cpu // int(n_models)).to(device=device)
    return model_id, version, pair_cpu.to(device=device)


def _encode_message_batch(
    *,
    preamble_16: torch.Tensor,
    model_id: torch.Tensor,
    version: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized MessageCodec.encode() for a batch.

    Args:
        preamble_16: (16,) or (1, 16) float tensor in {0,1}
        model_id: (B,) int64 tensor in [0, 7]
        version: (B,) int64 tensor in [0, 15]

    Returns:
        (B, 32) float tensor in {0,1}
    """
    if model_id.dim() != 1 or version.dim() != 1:
        raise ValueError(f"model_id/version must be 1D, got {model_id.shape} / {version.shape}")
    if model_id.shape[0] != version.shape[0]:
        raise ValueError(f"model_id/version batch mismatch: {model_id.shape} vs {version.shape}")

    B = model_id.shape[0]
    device = model_id.device

    if preamble_16.dim() == 2:
        if preamble_16.shape != (1, 16):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = preamble_16.to(device=device, dtype=torch.float32).expand(B, -1)
    elif preamble_16.dim() == 1:
        if preamble_16.shape != (16,):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        # View as (1, 16) for consistency if needed, or expand directly
        preamble_16 = preamble_16.view(1, 16)

    msg = torch.zeros((int(model_id.shape[0]), 32), device=model_id.device, dtype=torch.float32)
    # Balanced Diet: 16-bit preamble
    msg[:, 0:16] = preamble_16.to(device=model_id.device)

    # Shifted Identity (start at bit 16)
    for i in range(3):
        msg[:, 16 + i] = ((model_id >> i) & 1).to(torch.float32)
    for i in range(4):
        msg[:, 19 + i] = ((version >> i) & 1).to(torch.float32)
    
    # No copies
    return msg


def train_stage2(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    top_k: int = 3,
    lr: float = 1e-3,
    msg_weight: float = 1.0,
    model_ce_weight: float = 1.0,
    version_ce_weight: float = 1.0,
    pair_ce_weight: float = 2.0,
    aux_weight: float = 0.5,
    qual_weight: float = 1.0,
    reverb_prob: float = 0.25,
    preamble_weight: float = 0.2,
    model_id_weight: float = 2.0,
    version_weight: float = 0.5,
    payload_on_clean: bool = True,
    payload_pos_only: bool = True,
    message_mode: str = "random_if_mixed",
    n_models: int = 8,
    n_versions: int = 16,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
    det_ramp: int = 0,
):
    """
    Train encoder with differentiable augments only.
    
    FIX from project plan: Use TOP-K windows for loss (matches inference objective!)
    """
    print(f"Starting Stage 2: Encoder Training for {epochs} epochs")
    
    aug = DifferentiableAugmenter(device, reverb_prob=reverb_prob)
    
    # Quality loss
    stft_loss = CachedSTFTLoss().to(device)
    
    # Freeze decoder
    for p in decoder.parameters():
        p.requires_grad = False
    
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    # Bitwise weights for message loss (same rationale as Stage 1B).
    # Bitwise weights for message loss (Balanced Diet).
    msg_weights = torch.zeros(32, device=device)
    msg_weights[0:16] = preamble_weight
    msg_weights[16:19] = model_id_weight
    msg_weights[19:23] = version_weight
    # Rest are 0.0

    weight_sum = msg_weights.sum().clamp(min=1e-8)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_loss_det = 0.0
        total_loss_aux = 0.0
        total_loss_msg = 0.0
        total_loss_model = 0.0
        total_loss_version = 0.0
        total_loss_pair = 0.0
        total_loss_qual = 0.0
        batches = 0
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None
        
        for i, batch in enumerate(loader):
            # Train only on samples meant to be watermarked according to manifest?
            # Or force all samples to be watermarked for encoder training?
            # Usually we take any audio and force watermark it for training.
            
            audio = batch["audio"].to(device)  # (B, 1, T)
            B = int(audio.shape[0])

            # Stage 2 always watermarks its carriers on-the-fly. When training on a mixed
            # manifest (watermarked positives + clean negatives for Stage 1), you can:
            # - `payload_pos_only=True`  : apply payload/ID losses only on labeled positives.
            # - `payload_pos_only=False` : apply payload/ID losses on all carriers (recommended
            #   when you want the encoder to strongly learn identity, since all carriers are
            #   watermarked in Stage 2 anyway).
            if payload_pos_only and ("has_watermark" in batch):
                pos_mask = (batch["has_watermark"].to(device) > 0.5)
            else:
                pos_mask = torch.ones((B,), device=device, dtype=torch.bool)

            if message_mode not in {"batch", "random", "random_if_mixed"}:
                raise ValueError(
                    f"message_mode must be one of {{'batch','random','random_if_mixed'}}, got {message_mode!r}"
                )

            # Build per-sample targets:
            # - If IDs are missing (e.g., model_id/version == -1), sample a balanced random pair.
            # - Do not allow constant default IDs to dominate supervision.
            batch_model = batch.get("model_id", torch.full((B,), -1, dtype=torch.long)).to(device)
            batch_ver = batch.get("version", torch.full((B,), -1, dtype=torch.long)).to(device)
            ids_valid = (batch_model >= 0) & (batch_ver >= 0)

            rand_model, rand_ver, _rand_pair = sample_pair_ids(B, n_models=n_models, n_versions=n_versions, device=device)

            if message_mode == "random":
                target_model = rand_model
                target_ver = rand_ver
            elif message_mode == "batch":
                # Use provided IDs for labeled positives; fall back to random for unlabeled.
                # (Allows mixed manifests without silently collapsing to a constant default.)
                target_model = torch.where(ids_valid, batch_model, rand_model)
                target_ver = torch.where(ids_valid, batch_ver, rand_ver)
            else:  # random_if_mixed
                # Overwrite only unlabeled samples (and always avoid invalid IDs).
                target_model = torch.where(ids_valid, batch_model, rand_model)
                target_ver = torch.where(ids_valid, batch_ver, rand_ver)

            preamble_16_cpu = batch["message"][:1, :16].to(device="cpu", dtype=torch.float32)
            message = _encode_message_batch(
                preamble_16=preamble_16_cpu,
                model_id=target_model.to(device="cpu", dtype=torch.long),
                version=target_ver.to(device="cpu", dtype=torch.long),
            ).to(device=device, dtype=torch.float32)
            
            # Embed (Only watermark positives to avoid polluting negatives!)
            # Initialize with original audio
            wm = audio.clone()
            
            if pos_mask.any():
                # Extract positives
                audio_pos = audio[pos_mask]
                msg_pos = message[pos_mask]
                
                # Encode positives
                wm_pos = encoder(audio_pos, msg_pos)
                
                # Place back into batch
                wm[pos_mask] = wm_pos
            
            # Augment (gradient-safe)
            # DifferentiableAugmenter needs to handle (B, 1, T) or we adapt
            # Current augmenter expects (B, T).
            # We can update augmenter or squeeze/unsqueeze. 
            # Let's fix augmenter to be shape agnostic or handle (B, 1, T).
            # For now, squeeze to (B, T) then unsqueeze after?
            # Augmenter returns (B, T) implies loss of channel info.
            # Best: Update augmenter to handle (B, 1, T).
            # For immediate fix: squeeze, augment, unsqueeze.
            augmented = aug(wm.squeeze(1)).unsqueeze(1)
            
            # Decode
            # - Detection should be trained against (potentially) attacked audio to build robustness.
            # - Payload/attribution is much harder; training it directly under heavy augments early
            #   often collapses into “presence-only”. By default we supervise payload on the clean
            #   watermarked audio while still supervising detection on augmented audio.
            outputs_det = decoder(augmented)
            outputs_payload = decoder(wm) if payload_on_clean else outputs_det
            
            # 1. Quality Loss
            # STFT loss expects (B, T) usually? 
            # CachedSTFTLoss usually takes (B, T).
            loss_qual = stft_loss(audio.squeeze(1), wm.squeeze(1))
            
            # 2. Detection & Message Loss (Top-K AND All-Window Aux)
            detect = outputs_det["all_window_probs"]  # (B, n_win)
            detect_logits = outputs_det["all_window_logits"]
            msg_logits = outputs_payload["all_message_logits"]
            model_logits = outputs_payload.get("all_model_logits")  # (B, n_win, n_models+1)
            version_logits = outputs_payload.get("all_version_logits")  # (B, n_win, n_versions+1)
            pair_logits = outputs_payload.get("all_pair_logits")  # (B, n_win, n_pairs+1)
            
            B, n_win = detect.shape
            k = min(top_k, n_win)
            
            # Top-k by detection probability
            _, top_idx = torch.topk(detect, k, dim=1)
            
            # Gather top-k detect logits
            top_det_logits = torch.gather(detect_logits, 1, top_idx)
            
            # Detection loss (Target = 1) - PRIMARY
            loss_det = F.binary_cross_entropy_with_logits(
                top_det_logits.mean(dim=1),
                torch.ones(B, device=device)
            )
            
            # AUXILIARY: All-window detection loss
            # Ensures every window contributes gradients, preventing "sparse gradient" plateau
            loss_aux = F.binary_cross_entropy_with_logits(
                detect_logits,
                torch.ones_like(detect_logits)
            )
            
            # Gather top-k message logits
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_logits.shape[-1])
            top_msg_logits = torch.gather(msg_logits, 1, top_idx_exp)
            
            # Message loss
            avg_msg_logits = top_msg_logits.mean(dim=1)
            per_bit = F.binary_cross_entropy_with_logits(avg_msg_logits, message, reduction="none")
            per_sample = (per_bit * msg_weights).sum(dim=1) / weight_sum
            if pos_mask.any():
                loss_msg = per_sample[pos_mask].mean()
            else:
                loss_msg = torch.tensor(0.0, device=device)

            # Attribution losses (classification heads).
            loss_model = torch.tensor(0.0, device=device)
            if model_logits is not None and model_ce_weight > 0 and pos_mask.any():
                n_classes = model_logits.shape[-1]
                top_model_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                avg_model_logits = torch.gather(model_logits, 1, top_model_idx).mean(dim=1)
                loss_model = F.cross_entropy(avg_model_logits[pos_mask], target_model[pos_mask])

            loss_version = torch.tensor(0.0, device=device)
            if version_logits is not None and version_ce_weight > 0 and pos_mask.any():
                n_classes = version_logits.shape[-1]
                top_ver_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                avg_ver_logits = torch.gather(version_logits, 1, top_ver_idx).mean(dim=1)
                loss_version = F.cross_entropy(avg_ver_logits[pos_mask], target_ver[pos_mask])

            loss_pair = torch.tensor(0.0, device=device)
            if pair_logits is not None and pair_ce_weight > 0 and pos_mask.any():
                n_classes = pair_logits.shape[-1]
                top_pair_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                avg_pair_logits = torch.gather(pair_logits, 1, top_pair_idx).mean(dim=1)
                pair_id = (target_ver * int(n_models) + target_model).to(dtype=torch.long)
                loss_pair = F.cross_entropy(avg_pair_logits[pos_mask], pair_id[pos_mask])
            
            # Combined Loss
            # Weighted: 1.0 Det + aux_weight * Aux + msg_weight * Msg + CE(model/version) + qual_weight * Qual
            
            # Apply detection ramp if requested
            w_det = 1.0
            if det_ramp > 0:
                # epoch is 0-indexed.
                # ramp: 0 -> 1 over det_ramp epochs.
                # e=0 -> 0.0 (or small epsilon?) Let's do 0.0 to 1.0 linearly.
                # e=det_ramp -> 1.0
                progress = min(1.0, float(epoch) / float(det_ramp))
                w_det = progress
            
            loss = (
                w_det * loss_det
                + w_det * aux_weight * loss_aux
                + msg_weight * loss_msg
                + model_ce_weight * loss_model
                + version_ce_weight * loss_version
                + pair_ce_weight * loss_pair
                + qual_weight * loss_qual
            )
            
            if loss.grad_fn is None:
                # No positives in batch -> encoder was not used -> no gradients.
                # Skip step.
                batches += 1 # Count batch even if skipped to avoid div by zero issues
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            total_loss_det += loss_det.item()
            total_loss_aux += loss_aux.item()
            total_loss_msg += loss_msg.item()
            total_loss_model += loss_model.item()
            total_loss_version += loss_version.item()
            total_loss_pair += loss_pair.item()
            total_loss_qual += loss_qual.item()
            batches += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                on_step(
                    {
                        "type": "step",
                        "stage": "s2",
                        "epoch": epoch + 1,
                        "batch": int(i),
                        "n_batches": n_batches,
                        "loss": float(loss.item()),
                        "loss_det": float(loss_det.item()),
                        "loss_aux": float(loss_aux.item()),
                        "loss_msg": float(loss_msg.item()),
                        "loss_model_ce": float(loss_model.item()),
                        "loss_version_ce": float(loss_version.item()),
                        "loss_pair_ce": float(loss_pair.item()),
                        "loss_qual": float(loss_qual.item()),
                        "msg_weight": float(msg_weight),
                        "model_ce_weight": float(model_ce_weight),
                        "version_ce_weight": float(version_ce_weight),
                        "pair_ce_weight": float(pair_ce_weight),
                        "payload_on_clean": bool(payload_on_clean),
                        "payload_pos_only": bool(payload_pos_only),
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f} (Qual: {loss_qual.item():.4f}, Det: {loss_det.item():.4f})")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        if on_epoch_end is not None:
            denom = max(1, batches)
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": "s2",
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "loss_det": total_loss_det / denom,
                    "loss_aux": total_loss_aux / denom,
                    "loss_msg": total_loss_msg / denom,
                    "loss_model_ce": total_loss_model / denom,
                    "loss_version_ce": total_loss_version / denom,
                    "loss_pair_ce": total_loss_pair / denom,
                    "loss_qual": total_loss_qual / denom,
                    "msg_weight": float(msg_weight),
                    "model_ce_weight": float(model_ce_weight),
                    "version_ce_weight": float(version_ce_weight),
                    "pair_ce_weight": float(pair_ce_weight),
                    "aux_weight": float(aux_weight),
                    "qual_weight": float(qual_weight),
                    "reverb_prob": float(reverb_prob),
                    "payload_on_clean": bool(payload_on_clean),
                    "payload_pos_only": bool(payload_pos_only),
                    "lr": opt.param_groups[0].get("lr"),
                }
            )


=== ANALYSIS: final_analysis_report.md ===
# Final Analysis: Identity Starvation Root Cause & Fix

**Status**: 🟡 Sanity Test 3 (Balanced Diet) is Running.
**Verdict**: We have successfully isolated the two factors killing Identity Accuracy.

## 1. The Diagnosis
Our tests confirm a "Double Bottleneck":

### Factor A: Gradient Drowning (Confirmed)
- **Evidence**: In **Experiment E**, detection gradients (`neg_weight=5.0`) completely drowned out identity gradients, keeping accuracy at random chance (14.2%).
- **Proof**: In **Sanity Test 1**, strictly by introducing a `det_ramp` (suppressing detection gradients early), identity accuracy jumped to **25%** (2x chance).
- **Conclusion**: The encoder *cannot* learn identity if the detection supervisor is screaming too loud. The `det_ramp` is mandatory.

### Factor B: Bandwidth Exhaustion (Confirmed)
- **Evidence**: **Sanity Test 1** plateaued effectively at 25% (exactly 1 bit of information).
- **Proof**: The model exhausted its capacity learning the 16-bit Preamble and could only fit 1 bit of Identity into the residual bottleneck.
- **Fail Case**: **Sanity Test 2** (4-bit Preamble) failed because the sync became too loose (AUC 99% but margin <0.2), causing "Synchronization Jitter".
- **Conclusion**: We need a "Balanced Diet": A robust 16-bit Preamble for sync, but a slimmed-down Identity payload (7 bits vs 16 bits) to unclog the bottleneck.

---

## 2. The Solution: "Balanced Diet" Strategy
We are currently running **Sanity Test 3** with the optimal configuration derived from these findings.

| Component | Exp E (Fail) | Sanity 3 (Optimized) | Why? |
| :--- | :--- | :--- | :--- |
| **Detection Weight** | Constant (Strong) | **Ramp 0→1** | Gives encoder "free time" to learn bits first. |
| **Preamble** | 16 bits | **16 bits** | Required for robust window synchronization. |
| **Payload** | 16 bits (w/ copies) | **7 bits** (No copies) | Frees up 30% of bandwidth for Identity. |
| **Total Bits** | 32 bits | **23 bits** | Fits within the encoder's observed capacity. |

## 3. Next Steps
Once Sanity 3 confirms >50% accuracy (breaking the 1-bit ceiling), we should scale this configuration to the full 2k dataset:
1.  **Modify** `train_full` config to use `det_ramp=10`.
2.  **Keep** the 23-bit payload (Code patch is already applied).
3.  **Launch** `exp_f_final`.
