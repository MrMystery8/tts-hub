#!/usr/bin/env python3
"""
Evaluate a trained watermark run (encoder.pt + decoder.pt) on a manifest split using the probe metric suite.

Examples:
  ./.venv/bin/python watermark/scripts/eval_run_suite.py --run outputs/dashboard_runs/1769835316_1c6833 --n 256
  ./.venv/bin/python watermark/scripts/eval_run_suite.py --run outputs/dashboard_runs/1769794446_d26a56 --manifest outputs/dashboard_runs/1769794446_d26a56/_tmp_eval_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from watermark.config import DEVICE, N_CLASSES
from watermark.evaluation.probe import ProbeItem, compute_probe_metrics
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset


def _load_state_dict(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _pick_manifest(run_dir: Path) -> Path:
    for name in ["manifest_test.json", "_tmp_eval_manifest.json", "manifest_val.json", "manifest_train.json", "manifest.json"]:
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No manifest found in {run_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a watermark run using the probe metric suite.")
    ap.add_argument("--run", type=str, required=True, help="Run directory containing encoder.pt + decoder.pt")
    ap.add_argument("--manifest", type=str, default=None, help="Optional manifest to evaluate (default: pick from run dir)")
    ap.add_argument("--n", type=int, default=256, help="Max number of clips to evaluate")
    ap.add_argument(
        "--attacks",
        type=str,
        default="resample_8k,noise_white_20db",
        help="Comma-separated extra attacks to evaluate (in addition to clean + reverb).",
    )
    ap.add_argument("--no_reverb", action="store_true", help="Disable reverb attack evaluation")
    ap.add_argument("--json_out", type=str, default=None, help="Write full metrics JSON to this path")

    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    enc_path = run_dir / "encoder.pt"
    dec_path = run_dir / "decoder.pt"
    if not enc_path.exists():
        raise FileNotFoundError(f"Missing encoder.pt: {enc_path}")
    if not dec_path.exists():
        raise FileNotFoundError(f"Missing decoder.pt: {dec_path}")

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else _pick_manifest(run_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    extra_attacks = [a.strip() for a in str(args.attacks).split(",") if a.strip()]

    print(f"Run: {run_dir.name}")
    print(f"Manifest: {manifest_path}")
    print(f"Device: {DEVICE}")

    encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES)).to(DEVICE)
    decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES)).to(DEVICE)
    encoder.load_state_dict(_load_state_dict(enc_path), strict=True)
    decoder.load_state_dict(_load_state_dict(dec_path), strict=True)

    ds = WatermarkDataset(str(manifest_path), training=False)
    n = min(max(0, int(args.n)), len(ds))
    if n <= 0:
        raise RuntimeError("No clips available for evaluation (n<=0).")

    items: list[ProbeItem] = []
    for i in range(n):
        it = ds[i]
        items.append(ProbeItem(audio=it["audio"].detach().cpu(), y_class=int(it["y_class"].item())))

    metrics = compute_probe_metrics(
        items,
        encoder=encoder,
        decoder=decoder,
        device=DEVICE,
        compute_reverb=not bool(args.no_reverb),
        extra_attacks=extra_attacks,
        include_confusion=False,
    )

    # Print a compact summary first (keys most useful for stage comparisons).
    keys = [
        "mini_auc",
        "tpr_at_fpr_1pct",
        "id_acc_pos",
        "wm_acc",
        "wm_snr_db_mean",
        "wm_budget_ok_frac",
        "mini_auc_reverb",
        "tpr_at_fpr_1pct_reverb",
        "id_acc_pos_reverb",
    ]
    for a in extra_attacks:
        keys.extend([f"tpr_at_fpr_1pct_{a}", f"id_acc_pos_{a}"])

    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
