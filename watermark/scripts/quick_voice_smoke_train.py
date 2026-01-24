#!/usr/bin/env python3
"""
Quick smoke test using REAL voice clips from a local dataset (e.g. mini_benchmark_data).

Builds a tiny manifest from real audio, runs Stage 1/1B/2 for a few epochs,
and saves clean/watermarked WAVs for listening.
"""
import argparse
import json
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
    parser.add_argument("--reverb_prob", type=float, default=None, help="Stage 2 differentiable reverb probability")
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
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

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
                    "probe_n": probe_n,
                    "probe_every": probe_every,
                    "probe_reverb_every": probe_reverb_every,
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
            on_epoch_end=lambda e: log_event(mlog, e),
        )

        print("\n[QuickVoice] Stage 1B (Payload)")
        train_stage1b(
            decoder,
            encoder,
            loader,
            DEVICE,
            codec.preamble,
            epochs=epochs_s1b,
            warmup=1,
            neg_weight=neg_weight,
            neg_preamble_target=neg_preamble_target,
            unknown_ce_weight=unknown_ce_weight,
            model_ce_weight=model_ce_weight,
            version_ce_weight=version_ce_weight,
            pair_ce_weight=pair_ce_weight,
            log_interval=999,
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
            reverb_prob=reverb_prob,
            log_interval=999,
            on_epoch_end=lambda e: (log_event(mlog, e), maybe_probe(mlog, stage="s2", epoch=e["epoch"])),
        )

        if epochs_s1b_post and epochs_s1b_post > 0:
            print("\n[QuickVoice] Stage 1B (Payload) - Post Stage 2 fine-tune")
            # IMPORTANT: Post-Stage2 fine-tune should not wreck detection/preamble separation.
            # Keep this as a heads-only fit by running Stage1B entirely in "warmup" mode.
            train_stage1b(
                decoder,
                encoder,
                loader,
                DEVICE,
                codec.preamble,
                epochs=epochs_s1b_post,
                warmup=epochs_s1b_post,
                neg_weight=neg_weight,
                neg_preamble_target=neg_preamble_target,
                unknown_ce_weight=unknown_ce_weight,
                model_ce_weight=model_ce_weight,
                version_ce_weight=version_ce_weight,
                pair_ce_weight=pair_ce_weight,
                log_interval=999,
                on_epoch_end=lambda e: (
                    log_event(mlog, {**e, "stage": "s1b_post"}),
                    maybe_probe(mlog, stage="s1b_post", epoch=e["epoch"]),
                ),
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
        y_true = []
        y_score = []
        preamble_scores = []
        preamble_scores_pos = []
        preamble_scores_neg = []
        pos_total = 0
        pos_exact = 0
        pos_model_correct = 0
        pos_version_correct = 0
        pos_exact_cls = 0
        pos_model_cls_correct = 0
        pos_version_cls_correct = 0
        pos_pair_cls_correct = 0
        saw_cls_heads = False
        pos_bits_correct = 0.0
        pos_bits_total = 0.0
        pos_model_bits_correct = 0.0
        pos_model_bits_total = 0.0
        pos_version_bits_correct = 0.0
        pos_version_bits_total = 0.0
        for item in dataset:
            clip = item["audio"].unsqueeze(0).to(DEVICE)  # (1, 1, T)
            msg = item["message"].unsqueeze(0).to(DEVICE)
            label = float(item["has_watermark"].item())

            with torch.no_grad():
                wm = encoder(clip, msg)
                if label == 1.0:
                    inp = wm
                else:
                    inp = clip
                out = decoder(inp)
                y_score.append(out["clip_detect_prob"].item())
                y_true.append(label)
                msg_probs = out["avg_message_probs"].squeeze(0)
                decoded = codec.decode(msg_probs)
                preamble_scores.append(decoded["preamble_score"])
                if label == 1.0:
                    preamble_scores_pos.append(decoded["preamble_score"])
                else:
                    preamble_scores_neg.append(decoded["preamble_score"])

                if label == 1.0:
                    pos_total += 1
                    target_model_id = int(item["model_id"].item())
                    target_version = int(item["version"].item())
                    if decoded["model_id"] == target_model_id:
                        pos_model_correct += 1
                    if decoded["version"] == target_version:
                        pos_version_correct += 1
                    if decoded["model_id"] == target_model_id and decoded["version"] == target_version:
                        pos_exact += 1

                    # Classification head attribution (preferred)
                    if "avg_pair_logits" in out:
                        saw_cls_heads = True
                        p = out["avg_pair_logits"].squeeze(0)
                        pred_pair = int(torch.argmax(p).item())
                        pair_id = target_version * 8 + target_model_id
                        if pred_pair == pair_id:
                            pos_pair_cls_correct += 1
                            pos_exact_cls += 1
                        pred_model = pred_pair % 8
                        pred_ver = pred_pair // 8
                        if pred_model == target_model_id:
                            pos_model_cls_correct += 1
                        if pred_ver == target_version:
                            pos_version_cls_correct += 1
                    elif "avg_model_logits" in out and "avg_version_logits" in out:
                        saw_cls_heads = True
                        m = out["avg_model_logits"].squeeze(0)
                        v = out["avg_version_logits"].squeeze(0)
                        pred_model = int(torch.argmax(m).item())
                        pred_ver = int(torch.argmax(v).item())
                        if pred_model == target_model_id:
                            pos_model_cls_correct += 1
                        if pred_ver == target_version:
                            pos_version_cls_correct += 1
                        if pred_model == target_model_id and pred_ver == target_version:
                            pos_exact_cls += 1

                    # Bits 16-29 = payload + copies. Split model/version to avoid misleading constants.
                    model_idx = torch.tensor([16, 17, 18, 23, 24, 25], device=msg_probs.device)
                    ver_idx = torch.tensor([19, 20, 21, 22, 26, 27, 28, 29], device=msg_probs.device)

                    target_bits_all = item["message"][16:30].to(msg_probs.device)
                    pred_bits_all = (msg_probs[16:30] > 0.5).float()
                    pos_bits_correct += (pred_bits_all == target_bits_all).float().sum().item()
                    pos_bits_total += target_bits_all.numel()

                    target_model_bits = item["message"].to(msg_probs.device).index_select(0, model_idx)
                    pred_model_bits = (msg_probs.index_select(0, model_idx) > 0.5).float()
                    pos_model_bits_correct += (pred_model_bits == target_model_bits).float().sum().item()
                    pos_model_bits_total += target_model_bits.numel()

                    target_ver_bits = item["message"].to(msg_probs.device).index_select(0, ver_idx)
                    pred_ver_bits = (msg_probs.index_select(0, ver_idx) > 0.5).float()
                    pos_version_bits_correct += (pred_ver_bits == target_ver_bits).float().sum().item()
                    pos_version_bits_total += target_ver_bits.numel()

        auc = roc_auc_score(y_true, y_score)
        report_lines.append(f"mini_auc={auc:.4f} (on {len(y_true)} clips)")
        if preamble_scores:
            report_lines.append(f"preamble_avg={sum(preamble_scores)/len(preamble_scores):.2f}")
        if preamble_scores_pos:
            report_lines.append(f"preamble_pos_avg={sum(preamble_scores_pos)/len(preamble_scores_pos):.2f}")
        if preamble_scores_neg:
            report_lines.append(f"preamble_neg_avg={sum(preamble_scores_neg)/len(preamble_scores_neg):.2f}")
        if pos_total > 0:
            report_lines.append(f"payload_exact_acc={pos_exact/pos_total:.3f} (model_id+version)")
            report_lines.append(f"model_id_acc={pos_model_correct/pos_total:.3f}")
            report_lines.append(f"version_acc={pos_version_correct/pos_total:.3f}")
            if saw_cls_heads:
                report_lines.append(f"payload_exact_acc_cls={pos_exact_cls/pos_total:.3f} (model_id+version)")
                report_lines.append(f"model_id_acc_cls={pos_model_cls_correct/pos_total:.3f}")
                report_lines.append(f"version_acc_cls={pos_version_cls_correct/pos_total:.3f}")
                if pos_pair_cls_correct > 0:
                    report_lines.append(f"pair_acc_cls={pos_pair_cls_correct/pos_total:.3f}")
            if pos_bits_total > 0:
                report_lines.append(f"payload_bit_acc={pos_bits_correct/pos_bits_total:.3f} (bits 16-29)")
            if pos_model_bits_total > 0:
                report_lines.append(f"model_id_bit_acc={pos_model_bits_correct/pos_model_bits_total:.3f} (bits 16-18,23-25)")
            if pos_version_bits_total > 0:
                report_lines.append(f"version_bit_acc={pos_version_bits_correct/pos_version_bits_total:.3f} (bits 19-22,26-29)")

        # Robustness check: AUC under reverb (CPU attack -> decode on model device)
        if "reverb" in ATTACKS:
            y_score_reverb = []
            for item in dataset:
                clip = item["audio"].unsqueeze(0).to(DEVICE)  # (1, 1, T)
                msg = item["message"].unsqueeze(0).to(DEVICE)
                label = float(item["has_watermark"].item())

                with torch.no_grad():
                    wm = encoder(clip, msg)
                    inp = wm if label == 1.0 else clip

                attacked_ct = apply_attack_safe(inp.squeeze(0).cpu(), ATTACKS["reverb"])  # (1, T) on CPU
                attacked_bt = attacked_ct.unsqueeze(0).to(DEVICE)  # (1, 1, T) on model device

                with torch.no_grad():
                    out = decoder(attacked_bt)
                    y_score_reverb.append(out["clip_detect_prob"].item())

            auc_reverb = roc_auc_score(y_true, y_score_reverb)
            report_lines.append(f"mini_auc_reverb={auc_reverb:.4f}")
            pos_scores = [s for s, y in zip(y_score_reverb, y_true) if y == 1.0]
            neg_scores = [s for s, y in zip(y_score_reverb, y_true) if y == 0.0]
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
