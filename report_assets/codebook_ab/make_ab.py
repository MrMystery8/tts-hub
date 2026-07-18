#!/usr/bin/env python3
"""A/B listening clips: clean vs deterministic-codebook vs learned encoder (Run B)."""
import json
import sys
from pathlib import Path

import torch
import soundfile as sf

sys.path.insert(0, "/private/tmp/claude-501/-Users-ayaanminhas-Desktop-Personal-Work-tts-hub/bc714d2d-67d4-490d-b4f3-57e77c49b96f/scratchpad")

from watermark.utils.io import load_audio
from watermark.models import WatermarkEncoder, OverlapAddEncoder
from deterministic_codebook import DeterministicCodebookEncoder

OUT = Path("/private/tmp/claude-501/-Users-ayaanminhas-Desktop-Personal-Work-tts-hub/bc714d2d-67d4-490d-b4f3-57e77c49b96f/scratchpad/ab_demo")
OUT.mkdir(exist_ok=True)
RUNS = Path("outputs/dashboard_runs")
SR = 16000
CLASS_ID = 1

# ---- pick a real carrier clip, preferring a longer one (pauses expose tones)
man = json.load(open(RUNS / "sweep3_B_static_12_2" / "manifest_test.json"))
cands = [m["path"] for m in man if Path(m["path"]).exists()]
best, best_len = None, 0
for p in cands[:80]:
    try:
        a = load_audio(p, target_sr=SR)
    except Exception:
        continue
    if a.shape[-1] > best_len:
        best, best_len = p, a.shape[-1]
print(f"carrier: {best}  ({best_len/SR:.1f}s)")

audio = load_audio(best, target_sr=SR)          # (1, T)
batch = audio.unsqueeze(0)                       # (1, 1, T)
cid = torch.tensor([CLASS_ID], dtype=torch.long)


def snr_db(orig, wm):
    d = (wm - orig)
    return float(10 * torch.log10(orig.pow(2).mean() / d.pow(2).mean().clamp_min(1e-12)))


def save(name, wav):
    x = wav.squeeze().detach().cpu()
    peak = x.abs().max().clamp_min(1e-9)
    if peak > 0.99:
        x = x * (0.99 / peak)
    sf.write(str(OUT / name), x.numpy(), SR)


results = []
save("00_clean.wav", audio)

# ---- deterministic codebook variants (real codebook.json from the runs)
CB = [
    ("stable-v1 (3 tones, -30 dB, 1200-7200 Hz)", "1770012678_514ceb", "01_codebook_stable-v1.wav"),
    ("multitone-soft (16 tones, -33 dB, 1200-3800 Hz)", "1770025988_1ed3ea", "02_codebook_multitone-soft.wav"),
    ("subband-noise (64 comp, -40 dB, 1200-3800 Hz)", "1770021954_e5443c", "03_codebook_subband-noise.wav"),
]
for label, run, fname in CB:
    cbp = RUNS / run / "codebook.json"
    cfg = json.load(open(cbp))
    enc = DeterministicCodebookEncoder.load_from_codebook(num_classes=9, codebook_path=cbp)
    enc.eval()
    with torch.no_grad():
        wm = enc(batch, cid)
    save(fname, wm)
    results.append((label, fname, snr_db(batch, wm), cfg.get("style"), cfg.get("budget_target_db")))

# ---- learned encoder from the delivered model (Run B)
sd = torch.load(RUNS / "sweep3_B_static_12_2" / "encoder.pt", map_location="cpu")
base = WatermarkEncoder(num_classes=4)
lenc = OverlapAddEncoder(base)
lenc.load_state_dict(sd)   # checkpoint is the wrapped OverlapAddEncoder
lenc.eval()
with torch.no_grad():
    wmL = lenc(batch, cid)
save("04_learned_runB.wav", wmL)
results.append(("learned encoder (Run B, delivered)", "04_learned_runB.wav", snr_db(batch, wmL), "learned", -30.0))

# ---- perceptual metrics
try:
    from pesq import pesq as pesq_fn
    from pystoi import stoi as stoi_fn
    have = True
except Exception as e:
    print("perceptual libs unavailable:", e)
    have = False

print(f"\n{'variant':52} {'SNR dB':>8} {'PESQ':>6} {'STOI':>6}")
print("-" * 76)
ref = audio.squeeze().numpy()
for label, fname, s, style, budget in results:
    p = st = float("nan")
    if have:
        deg, _ = sf.read(str(OUT / fname))
        try:
            p = pesq_fn(SR, ref[: len(deg)], deg[: len(ref)], "wb")
            st = stoi_fn(ref[: len(deg)], deg[: len(ref)], SR, extended=False)
        except Exception as e:
            print("  metric fail", label, e)
    print(f"{label:52} {s:8.2f} {p:6.2f} {st:6.3f}")

print("\nfiles ->", OUT)
