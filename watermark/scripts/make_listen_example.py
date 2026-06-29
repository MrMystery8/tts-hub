#!/usr/bin/env python3
"""
Generate A/B listening examples for a trained watermark run.

For each carrier clip it writes:
  - <i>_clean.wav      : original (no watermark)
  - <i>_watermarked.wav: same clip with the model's watermark embedded
  - <i>_delta_x8.wav   : the watermark signal ALONE, amplified 8x so you can
                         hear *what* is being added (this is NOT how loud it is
                         in the mix — it's the isolated mark, boosted).

Also prints the measured SNR per clip (higher dB = quieter/more imperceptible
watermark; the project target is >= 30 dB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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


def _save(path: Path, x: torch.Tensor):
    a = x.squeeze().detach().cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(a))) or 1.0
    if peak > 1.0:  # avoid clipping on save only
        a = a / peak
    sf.write(str(path), a, SAMPLE_RATE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--class_id", type=int, default=1, help="watermark class (1..K)")
    ap.add_argument("--n", type=int, default=3, help="how many clips")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        run_dir = Path("outputs/dashboard_runs") / args.run_dir
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    num_classes = int(cfg.get("num_classes") or cfg.get("n_classes") or (int(cfg.get("n_models", 8)) + 1))

    enc = OverlapAddEncoder(WatermarkEncoder(num_classes=num_classes))
    enc.load_state_dict(_extract(torch.load(run_dir / "encoder.pt", map_location="cpu"), "encoder"), strict=True)
    enc.eval()

    man = next((run_dir / m for m in ("manifest_test.json", "manifest_val.json", "manifest.json") if (run_dir / m).exists()))
    ds = WatermarkDataset(str(man), training=False, n_models=num_classes - 1)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "listen")
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    i = 0
    print(f"run={run_dir.name} class_id={args.class_id} -> {out_dir}")
    while made < args.n and i < len(ds):
        item = ds[i]; i += 1
        clean = item["audio"].unsqueeze(0)  # (1,1,T)
        y = torch.tensor([int(args.class_id)], dtype=torch.long)
        with torch.no_grad():
            wm = enc(clean, y)
        delta = (wm - clean)
        po = float(clean.pow(2).mean()); pd = float(delta.pow(2).mean())
        snr = 10.0 * float(np.log10((po + 1e-12) / (pd + 1e-12)))
        _save(out_dir / f"{made}_clean.wav", clean)
        _save(out_dir / f"{made}_watermarked.wav", wm)
        _save(out_dir / f"{made}_delta_x8.wav", (delta * 8.0).clamp(-1, 1))
        print(f"  clip {made}: SNR={snr:.1f} dB  (target >=30 dB)")
        made += 1

    print(f"done: {made} examples in {out_dir}")
    print("Listen: compare *_clean.wav vs *_watermarked.wav. *_delta_x8.wav = the mark alone, 8x louder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
