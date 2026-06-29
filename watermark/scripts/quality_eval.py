#!/usr/bin/env python3
"""
Perceptual quality eval for a trained watermark run.

Embeds the watermark in clean carrier clips and measures clean-vs-watermarked:
  - PESQ (wideband, 16 kHz): perceptual speech quality, ~1.0 (bad) .. 4.5 (clean)
  - STOI: short-time objective intelligibility, 0 .. 1
  - SNR (dB): raw energy ratio, for reference (NOT a perceptual measure)

Rule of thumb for "transparent": PESQ >= ~4.0 and STOI >= ~0.95.
These complement, not replace, informal listening.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi

from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset
from watermark.config import SAMPLE_RATE


def _extract(ckpt, which):
    if isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    if isinstance(ckpt, dict):
        if which in ckpt and isinstance(ckpt[which], dict):
            return ckpt[which]
        if isinstance(ckpt.get("state_dict"), dict):
            return ckpt["state_dict"]
    raise TypeError(f"bad ckpt for {which}")


def _pct(a, q):
    return float(np.percentile(np.array(a, dtype=np.float64), q)) if a else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n", type=int, default=200, help="clips to score")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        run_dir = Path("outputs/dashboard_runs") / args.run_dir
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    num_classes = int(cfg.get("num_classes") or cfg.get("n_classes") or (int(cfg.get("n_models", 8)) + 1))
    K = num_classes - 1

    enc = OverlapAddEncoder(WatermarkEncoder(num_classes=num_classes))
    enc.load_state_dict(_extract(torch.load(run_dir / "encoder.pt", map_location="cpu"), "encoder"), strict=True)
    enc.eval()

    man = next((run_dir / m for m in ("manifest_test.json", "manifest_val.json", "manifest.json") if (run_dir / m).exists()))
    ds = WatermarkDataset(str(man), training=False, n_models=K)
    n = min(max(1, args.n), len(ds))

    pesqs, stois, snrs = [], [], []
    for i in range(n):
        clean = ds[i]["audio"].unsqueeze(0)              # (1,1,T)
        y = torch.tensor([1 + (i % K)], dtype=torch.long)  # round-robin classes 1..K
        with torch.no_grad():
            wm = enc(clean, y)
        ref = clean.squeeze().detach().cpu().numpy().astype(np.float64)
        deg = wm.squeeze().detach().cpu().numpy().astype(np.float64)
        delta = deg - ref
        snrs.append(10.0 * float(np.log10((np.mean(ref**2) + 1e-12) / (np.mean(delta**2) + 1e-12))))
        try:
            pesqs.append(float(pesq(SAMPLE_RATE, ref, deg, "wb")))
        except Exception:
            pass
        try:
            stois.append(float(stoi(ref, deg, SAMPLE_RATE, extended=False)))
        except Exception:
            pass

    def line(name, a, hi_good=True):
        if not a:
            return f"  {name:6s}: (none)"
        return (f"  {name:6s}: mean={np.mean(a):.3f}  p10={_pct(a,10):.3f}  "
                f"p50={_pct(a,50):.3f}  min={min(a):.3f}  max={max(a):.3f}")

    print(f"run={run_dir.name}  scored={n} clips  classes={K}")
    print(line("PESQ", pesqs))
    print(line("STOI", stois))
    print(line("SNRdB", snrs))
    verdict = "TRANSPARENT-ish" if (pesqs and np.mean(pesqs) >= 4.0 and stois and np.mean(stois) >= 0.95) else "check"
    print(f"  heuristic: {verdict}  (transparent if PESQ>=4.0 and STOI>=0.95)")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "quality_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "run": run_dir.name, "n": n, "classes": K,
        "pesq": {"mean": float(np.mean(pesqs)) if pesqs else None, "p10": _pct(pesqs, 10), "p50": _pct(pesqs, 50)},
        "stoi": {"mean": float(np.mean(stois)) if stois else None, "p10": _pct(stois, 10), "p50": _pct(stois, 50)},
        "snr_db": {"mean": float(np.mean(snrs)) if snrs else None, "p10": _pct(snrs, 10), "p50": _pct(snrs, 50)},
    }
    (out_dir / "quality_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}/quality_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
