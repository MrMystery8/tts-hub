from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _pick_default_prompt(hub_root: Path) -> Path | None:
    uploads = hub_root / "outputs" / "uploads"
    if not uploads.exists():
        return None
    for p in sorted(uploads.glob("*/prompt.wav")):
        return p
    return None


def _swap_used_gb() -> float | None:
    """
    Best-effort macOS swap used (GB). Returns None if not available.
    """
    import re
    import subprocess

    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    except Exception:
        return None

    # Example:
    # vm.swapusage: total = 1024.00M  used = 0.00M  free = 1024.00M  (encrypted)
    m = re.search(r"used\s*=\s*([0-9.]+)([MG])", out)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "M":
        return val / 1024.0
    return val


def _worker_rss_gb(manager: Any, model_id: str = "index-tts2") -> float | None:
    """
    Read worker RSS via HubManager private fields. Works for both baseline and patched workers.
    """
    try:
        worker = manager._workers.get(model_id)  # type: ignore[attr-defined]
        if not worker:
            return None
        proc = getattr(worker, "_proc", None)
        pid = getattr(proc, "pid", None)
        if not pid:
            return None
        import psutil

        rss = int(psutil.Process(int(pid)).memory_info().rss)
        return float(rss) / (1024.0**3)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark IndexTTS2 worker memory/latency via HubManager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hub-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the tts-hub repo root",
    )
    parser.add_argument(
        "--prompt-wav",
        type=Path,
        default=None,
        help="Reference prompt wav (if omitted, tries outputs/uploads/*/prompt.wav)",
    )
    parser.add_argument("--text", type=str, default="Hello from IndexTTS2.")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between runs (seconds)")
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument(
        "--voice-id",
        type=str,
        default="",
        help="Optional stable 32-hex voice_id (enables speaker cache). If blank, auto-generates one.",
    )
    parser.add_argument(
        "--no-voice-id",
        action="store_true",
        help="Disable voice_id (skips persistent speaker cache path).",
    )
    args = parser.parse_args()

    hub_root = args.hub_root.resolve()
    prompt_wav = (args.prompt_wav or _pick_default_prompt(hub_root))
    if not prompt_wav:
        print("ERROR: --prompt-wav is required (no default prompt.wav found).", file=sys.stderr)
        return 2
    prompt_wav = Path(prompt_wav).resolve()
    if not prompt_wav.exists():
        print(f"ERROR: prompt wav not found: {prompt_wav}", file=sys.stderr)
        return 2

    voice_id = ""
    if not args.no_voice_id:
        voice_id = args.voice_id.strip().lower() or uuid.uuid4().hex
        if len(voice_id) != 32 or any(c not in "0123456789abcdef" for c in voice_id):
            print(f"ERROR: --voice-id must be 32 hex chars, got {voice_id!r}", file=sys.stderr)
            return 2

    # Local imports so this script can be run from anywhere.
    sys.path.insert(0, str(hub_root))
    from hub.hub_manager import HubManager

    manager = HubManager(hub_root=hub_root)

    print("== IndexTTS2 benchmark ==")
    print(f"hub_root:   {hub_root}")
    print(f"prompt_wav: {prompt_wav}")
    print(f"repeat:     {args.repeat}")
    if voice_id:
        print(f"voice_id:   {voice_id} (speaker cache enabled)")
    else:
        print("voice_id:   (disabled)")
    print(
        "env: "
        f"INDEXTTS2_MPS_MEMORY_FRACTION={os.getenv('INDEXTTS2_MPS_MEMORY_FRACTION','')} "
        f"INDEXTTS2_MPS_HIGH_WATERMARK_RATIO={os.getenv('INDEXTTS2_MPS_HIGH_WATERMARK_RATIO','')} "
        f"INDEXTTS2_MPS_LOW_WATERMARK_RATIO={os.getenv('INDEXTTS2_MPS_LOW_WATERMARK_RATIO','')} "
        f"INDEXTTS2_RECYCLE_DRIVER_GB={os.getenv('INDEXTTS2_RECYCLE_DRIVER_GB','')}"
    )
    print("")

    swap0 = _swap_used_gb()
    if swap0 is not None:
        print(f"swap_used_gb (start): {swap0:.3f}")

    for i in range(args.repeat):
        req_fields: dict[str, str] = {
            "model_id": "index-tts2",
            "max_text_tokens_per_segment": str(int(args.max_text_tokens)),
            "max_mel_tokens": str(int(args.max_mel_tokens)),
            "num_beams": str(int(args.num_beams)),
        }
        if voice_id:
            req_fields["voice_id"] = voice_id

        model_request = {
            "text": args.text,
            "prompt_audio_path": str(prompt_wav),
            "emo_audio_path": None,
            "fields": req_fields,
            "hub_root": str(hub_root),
        }

        t0 = time.perf_counter()
        result = manager.generate(model_id="index-tts2", request=model_request)
        dt = time.perf_counter() - t0
        meta = result.meta or {}

        rss_gb = _worker_rss_gb(manager) or None
        swap_gb = _swap_used_gb()

        print(
            f"[{i+1:02d}/{args.repeat}] {dt:6.2f}s "
            f"cache={meta.get('speaker_cache_status','?')} "
            f"worker_rss_gb={(f'{rss_gb:.3f}' if rss_gb is not None else meta.get('rss_gb','?'))} "
            f"mps_driver_gb={meta.get('mps_driver_gb','?')} "
            f"recycle={meta.get('recycle_recommended','?')}"
            + (f" swap_used_gb={swap_gb:.3f}" if swap_gb is not None else "")
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
