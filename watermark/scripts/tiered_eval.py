#!/usr/bin/env python3
"""
Tiered watermark robustness evaluation (IR Table 3.4: T1/T2/T3 suite).

Loads a trained run's encoder/decoder and reports, per transformation tier and
per attack: AUC, TPR@FPR=1%, attribution accuracy, and a composite score.
Emits a Markdown table to stdout plus JSON + CSV artefacts.

This does NOT retrain anything. It is the honest reporting harness for the
robustness claim the report pre-commits to (detection reported per tier, never
collapsed into a single robustness number).

Usage:
    python -m watermark.scripts.tiered_eval --run_dir outputs/dashboard_runs/<id>
    python -m watermark.scripts.tiered_eval --run_dir <id> --n_items 512 --tiers T1,T2,T3
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import torch

# torch.istft/stft emit a deprecated-resize UserWarning on some builds; harmless here.
warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")

from watermark.config import DEVICE
from watermark.evaluation.attacks import ATTACKS, TIERS, apply_attack_safe
from watermark.evaluation.probe import ProbeItem, _probe_once
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.training.dataset import WatermarkDataset


def _extract_state_dict(ckpt: object, which: str) -> dict:
    """Accept a plain state_dict, {'encoder'/'decoder': sd}, or {'state_dict': sd}."""
    if isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt  # type: ignore[return-value]
    if isinstance(ckpt, dict):
        if which in ckpt and isinstance(ckpt[which], dict):
            return ckpt[which]  # type: ignore[return-value]
        sd = ckpt.get("state_dict")
        if isinstance(sd, dict):
            return sd
    raise TypeError(f"Unsupported checkpoint format for {which}: {type(ckpt)}")


def _is_noop_attack(items: list[ProbeItem], attack_name: str, n_probe: int = 4) -> bool:
    """True if the attack leaves audio essentially unchanged (e.g. missing codec
    backend silently falling back to identity). Prevents reporting untested
    robustness as a pass."""
    fn = ATTACKS[attack_name]
    max_diff = 0.0
    for it in items[: max(1, n_probe)]:
        x = it.audio.detach().cpu()
        y = apply_attack_safe(x, fn)
        max_diff = max(max_diff, float((y - x).abs().max().item()))
    return max_diff < 1e-6


def _load_models(run_dir: Path, num_classes: int, device: torch.device):
    enc = OverlapAddEncoder(WatermarkEncoder(num_classes=num_classes)).to(device)
    dec = SlidingWindowDecoder(WatermarkDecoder(num_classes=num_classes)).to(device)
    enc_path = run_dir / "encoder.pt"
    dec_path = run_dir / "decoder.pt"
    if not enc_path.exists() or not dec_path.exists():
        raise FileNotFoundError(
            f"Need encoder.pt and decoder.pt in {run_dir}. "
            f"Found encoder={enc_path.exists()} decoder={dec_path.exists()}."
        )
    enc.load_state_dict(_extract_state_dict(torch.load(enc_path, map_location="cpu"), "encoder"), strict=True)
    dec.load_state_dict(_extract_state_dict(torch.load(dec_path, map_location="cpu"), "decoder"), strict=True)
    enc.eval()
    dec.eval()
    return enc, dec


def _resolve_manifest(run_dir: Path, override: str | None) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        return p
    for name in ("manifest_test.json", "manifest_val.json", "manifest.json"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No manifest found in {run_dir} (tried test/val/manifest).")


def _row_metrics(m: dict) -> dict:
    auc = float(m.get("mini_auc", float("nan")))
    tpr = float(m.get("tpr_at_fpr_1pct", float("nan")))
    idacc = float(m.get("id_acc_pos", float("nan")))
    wm = float(m.get("wm_acc", float("nan")))
    pos_rate = float(m.get("pred_pos_rate", float("nan")))
    comp = tpr * idacc if (tpr == tpr and idacc == idacc) else float("nan")
    return {
        "auc": auc, "tpr_at_fpr_1pct": tpr, "id_acc_pos": idacc,
        "wm_acc": wm, "pred_pos_rate": pos_rate, "composite": comp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tiered watermark robustness eval (T1/T2/T3)")
    ap.add_argument("--run_dir", required=True, help="Run directory with encoder.pt/decoder.pt/config.json")
    ap.add_argument("--manifest", default=None, help="Override manifest (default: run's test split)")
    ap.add_argument("--n_items", type=int, default=512, help="Max probe items")
    ap.add_argument("--tiers", default="T1,T2,T3", help="Comma-separated tiers to run")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: <run_dir>/tiered_eval)")
    ap.add_argument("--device", default=None, help="Override device (cpu/mps/cuda)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        # allow passing just the run id
        alt = Path("outputs/dashboard_runs") / args.run_dir
        if alt.exists():
            run_dir = alt.resolve()
        else:
            print(f"Run dir not found: {run_dir}")
            return 1

    device = torch.device(args.device) if args.device else DEVICE

    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    num_classes = int(cfg.get("num_classes") or cfg.get("n_classes") or ((int(cfg.get("n_models", 8)) + 1)))
    n_models = num_classes - 1
    print(f"[tiered_eval] run={run_dir.name} num_classes={num_classes} (n_models={n_models}) device={device}")

    manifest_path = _resolve_manifest(run_dir, args.manifest)
    ds = WatermarkDataset(str(manifest_path), training=False, n_models=n_models)
    n = min(max(1, int(args.n_items)), len(ds))
    items = [ProbeItem(audio=ds[i]["audio"].detach().cpu(), y_class=int(ds[i]["y_class"].item())) for i in range(n)]
    n_pos = sum(1 for it in items if it.y_class != 0)
    print(f"[tiered_eval] manifest={manifest_path.name} items={n} (pos={n_pos}, neg={n - n_pos})")
    if n_pos == 0 or n_pos == n:
        print("[tiered_eval] WARNING: need both clean and watermarked items for AUC/TPR.")

    enc, dec = _load_models(run_dir, num_classes, device)

    def run(attack_name: str | None) -> dict:
        fn = None if attack_name in (None, "clean") else ATTACKS[attack_name]
        return _probe_once(
            items, encoder=enc, decoder=dec, device=device,
            attack_fn=fn, num_classes=num_classes, include_confusion=False,
        )

    wanted = [t.strip() for t in args.tiers.split(",") if t.strip()]

    # Baseline (clean)
    base = run(None)
    base_row = _row_metrics(base)
    quality = {
        "wm_snr_db_mean": base.get("wm_snr_db_mean"),
        "wm_snr_db_p10": base.get("wm_snr_db_p10"),
        "wm_budget_ok_frac": base.get("wm_budget_ok_frac"),
        "wm_budget_target_db": base.get("wm_budget_target_db"),
    }

    results: list[dict] = [{"tier": "-", "attack": "clean", "status": "ok", **base_row}]
    tier_summ: dict[str, dict] = {}

    for tier in wanted:
        names = TIERS.get(tier, [])
        avail_rows = []
        for name in names:
            if name not in ATTACKS:
                results.append({"tier": tier, "attack": name, "status": "missing"})
                continue
            if _is_noop_attack(items, name):
                results.append({"tier": tier, "attack": name, "status": "UNAVAILABLE (no-op)"})
                continue
            r = _row_metrics(run(name))
            results.append({"tier": tier, "attack": name, "status": "ok", **r})
            avail_rows.append(r)
        if avail_rows:
            def avg(k):
                vals = [r[k] for r in avail_rows if r[k] == r[k]]
                return sum(vals) / len(vals) if vals else float("nan")
            tier_summ[tier] = {k: avg(k) for k in ("auc", "tpr_at_fpr_1pct", "id_acc_pos", "composite")}

    # ---- Render markdown table ----
    def f(v):
        return "  -  " if (v is None or (isinstance(v, float) and v != v)) else f"{v:.3f}"

    lines = []
    lines.append(f"# Tiered robustness — run `{run_dir.name}`\n")
    lines.append(f"- items: {n} (pos={n_pos}, neg={n - n_pos}); manifest: `{manifest_path.name}`")
    lines.append(
        f"- watermark quality: SNR mean={f(quality['wm_snr_db_mean'])} dB, "
        f"p10={f(quality['wm_snr_db_p10'])} dB, "
        f"budget_ok={f(quality['wm_budget_ok_frac'])} (target {quality['wm_budget_target_db']} dB)\n"
    )
    lines.append("| tier | attack | status | AUC | TPR@FPR1% | id_acc | wm_acc | pos_rate | composite |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("status") == "ok":
            lines.append(
                f"| {r['tier']} | {r['attack']} | ok | {f(r['auc'])} | {f(r['tpr_at_fpr_1pct'])} | "
                f"{f(r['id_acc_pos'])} | {f(r['wm_acc'])} | {f(r['pred_pos_rate'])} | {f(r['composite'])} |"
            )
        else:
            lines.append(f"| {r['tier']} | {r['attack']} | {r['status']} | - | - | - | - | - | - |")
    lines.append("\n**Per-tier mean (available attacks only):**\n")
    lines.append("| tier | AUC | TPR@FPR1% | id_acc | composite |")
    lines.append("|---|---|---|---|---|")
    for tier in wanted:
        s = tier_summ.get(tier)
        if s:
            lines.append(f"| {tier} | {f(s['auc'])} | {f(s['tpr_at_fpr_1pct'])} | {f(s['id_acc_pos'])} | {f(s['composite'])} |")
        else:
            lines.append(f"| {tier} | (no available attacks) | | | |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    # ---- Write artefacts ----
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "tiered_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tiered_eval.md").write_text(table + "\n", encoding="utf-8")
    (out_dir / "tiered_eval.json").write_text(
        json.dumps({"run": run_dir.name, "n_items": n, "n_pos": n_pos, "quality": quality,
                    "rows": results, "tier_summary": tier_summ}, indent=2),
        encoding="utf-8",
    )
    with open(out_dir / "tiered_eval.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "attack", "status", "auc", "tpr_at_fpr_1pct", "id_acc_pos", "wm_acc", "pred_pos_rate", "composite"])
        for r in results:
            w.writerow([r.get("tier"), r.get("attack"), r.get("status"),
                        r.get("auc"), r.get("tpr_at_fpr_1pct"), r.get("id_acc_pos"),
                        r.get("wm_acc"), r.get("pred_pos_rate"), r.get("composite")])
    print(f"[tiered_eval] wrote: {out_dir}/tiered_eval.(md|json|csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
