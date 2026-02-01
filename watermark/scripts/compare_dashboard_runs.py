#!/usr/bin/env python3
"""
Compare multiple dashboard runs under a shared evaluation suite.

Default behavior:
- Scans outputs/dashboard_runs
- Evaluates each run on its own manifest_test.json if present, else manifest_val/train/manifest.json
- Computes clean + reverb + extra attacks metrics

Example:
  ./.venv/bin/python watermark/scripts/compare_dashboard_runs.py --limit 8 --n 256
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


def _read_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare watermark dashboard runs.")
    ap.add_argument("--runs_dir", type=str, default="outputs/dashboard_runs")
    ap.add_argument("--limit", type=int, default=10, help="Evaluate the most recent N runs")
    ap.add_argument("--n", type=int, default=256, help="Max clips per run")
    ap.add_argument(
        "--attacks",
        type=str,
        default="resample_8k,noise_white_20db",
        help="Comma-separated extra attacks to evaluate (in addition to clean + reverb).",
    )
    ap.add_argument("--no_reverb", action="store_true")
    ap.add_argument("--sort_by", type=str, default="composite", help="Key to sort by (or 'composite')")

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    extra_attacks = [a.strip() for a in str(args.attacks).split(",") if a.strip()]

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name)
    if int(args.limit) > 0:
        run_dirs = run_dirs[-int(args.limit) :]

    rows: list[dict[str, Any]] = []

    for rd in run_dirs:
        enc_path = rd / "encoder.pt"
        dec_path = rd / "decoder.pt"
        if not (enc_path.exists() and dec_path.exists()):
            continue

        try:
            manifest_path = _pick_manifest(rd)
        except Exception:
            continue

        encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES)).to(DEVICE)
        decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES)).to(DEVICE)
        encoder.load_state_dict(_load_state_dict(enc_path), strict=True)
        decoder.load_state_dict(_load_state_dict(dec_path), strict=True)

        ds = WatermarkDataset(str(manifest_path), training=False)
        n = min(max(0, int(args.n)), len(ds))
        if n <= 0:
            continue

        items: list[ProbeItem] = []
        for i in range(n):
            it = ds[i]
            items.append(ProbeItem(audio=it["audio"].detach().cpu(), y_class=int(it["y_class"].item())))

        m = compute_probe_metrics(
            items,
            encoder=encoder,
            decoder=decoder,
            device=DEVICE,
            compute_reverb=not bool(args.no_reverb),
            extra_attacks=extra_attacks,
            include_confusion=False,
        )

        cfg = _read_config(rd)
        row: dict[str, Any] = {
            "run": rd.name,
            "manifest": str(manifest_path.name),
            "epochs_s1": cfg.get("epochs_s1"),
            "epochs_s2": cfg.get("epochs_s2"),
            "epochs_s1b_post": cfg.get("epochs_s1b_post"),
            "id_weight": cfg.get("id_weight"),
            "detect_weight": cfg.get("detect_weight"),
            "mini_auc": m.get("mini_auc"),
            "tpr_at_fpr_1pct": m.get("tpr_at_fpr_1pct"),
            "id_acc_pos": m.get("id_acc_pos"),
            "wm_acc": m.get("wm_acc"),
            "wm_snr_db_mean": m.get("wm_snr_db_mean"),
            "wm_budget_ok_frac": m.get("wm_budget_ok_frac"),
            "tpr_at_fpr_1pct_reverb": m.get("tpr_at_fpr_1pct_reverb"),
            "id_acc_pos_reverb": m.get("id_acc_pos_reverb"),
        }
        for a in extra_attacks:
            row[f"tpr_at_fpr_1pct_{a}"] = m.get(f"tpr_at_fpr_1pct_{a}")
            row[f"id_acc_pos_{a}"] = m.get(f"id_acc_pos_{a}")

        # Simple composite for sorting: clean detection strict point * clean ID accuracy.
        try:
            row["composite"] = float(row.get("tpr_at_fpr_1pct") or 0.0) * float(row.get("id_acc_pos") or 0.0)
        except Exception:
            row["composite"] = 0.0

        rows.append(row)

    sort_key = str(args.sort_by)
    if sort_key == "composite":
        rows.sort(key=lambda r: float(r.get("composite") or 0.0), reverse=True)
    else:
        rows.sort(key=lambda r: float(r.get(sort_key) or 0.0), reverse=True)

    # Print as TSV for quick copy/paste.
    cols = [
        "run",
        "manifest",
        "epochs_s1",
        "epochs_s2",
        "epochs_s1b_post",
        "detect_weight",
        "id_weight",
        "tpr_at_fpr_1pct",
        "id_acc_pos",
        "tpr_at_fpr_1pct_reverb",
        "id_acc_pos_reverb",
        "wm_snr_db_mean",
        "wm_budget_ok_frac",
        "composite",
    ]
    for a in extra_attacks:
        cols.extend([f"tpr_at_fpr_1pct_{a}", f"id_acc_pos_{a}"])

    print("\t".join(cols))
    for r in rows:
        out: list[str] = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                out.append(f"{v:.4f}")
            elif v is None:
                out.append("")
            else:
                out.append(str(v))
        print("\t".join(out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
