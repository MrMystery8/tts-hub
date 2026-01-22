from __future__ import annotations

import gc
import sys
import time
import wave
from pathlib import Path
from typing import Any

from _worker_protocol import recv, send

_tts = None


def _log(msg: str) -> None:
    print(f"[pocket-tts] {msg}", file=sys.stderr, flush=True)


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(v: str | None, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _int(v: str | None, default: int) -> int:
    try:
        return int(float(v)) if v is not None else default
    except ValueError:
        return default


def write_wav_mono_int16(path: Path, audio: "np.ndarray", sr: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def _load_model():
    global _tts
    if _tts is not None:
        return _tts
    _log("Loading Pocket TTS model...")
    from pocket_tts import TTSModel

    _tts = TTSModel.load_model()
    return _tts


def _handle_gen(req: dict[str, Any]) -> dict[str, Any]:
    fields = dict(req.get("fields") or {})
    hub_root = Path(req.get("hub_root") or ".").resolve()
    out_dir = hub_root / "outputs" / "pocket-tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"pocket_{request_id}.wav"

    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    tts = _load_model()

    # Optionally override generation params per-request
    tts.temp = _float(fields.get("temperature"), float(getattr(tts, "temp", 0.8)))
    tts.lsd_decode_steps = _int(fields.get("lsd_decode_steps"), int(getattr(tts, "lsd_decode_steps", 8)))
    tts.noise_clamp = _float(fields.get("noise_clamp"), float(getattr(tts, "noise_clamp", 0.0))) if fields.get("noise_clamp") else getattr(tts, "noise_clamp", None)
    tts.eos_threshold = _float(fields.get("eos_threshold"), float(getattr(tts, "eos_threshold", 0.4)))

    prompt_audio_path = req.get("prompt_audio_path")
    truncate_prompt = _bool(fields.get("truncate_prompt"), False)

    # If no prompt audio is provided, fall back to a predefined voice from HF.
    voice = (fields.get("voice") or "").strip()
    if not voice:
        voice = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

    audio_prompt = str(prompt_audio_path) if prompt_audio_path else voice

    t0 = time.time()
    voice_state = tts.get_state_for_audio_prompt(audio_prompt, truncate=truncate_prompt)
    audio = tts.generate_audio(voice_state, text)
    dt = time.time() - t0

    audio_np = audio.detach().cpu().numpy()
    write_wav_mono_int16(out_path, audio_np, int(tts.sample_rate))

    gc.collect()
    return {
        "output_path": str(out_path),
        "meta": {
            "model": "PocketTTS",
            "sr": int(tts.sample_rate),
            "voice": voice if not prompt_audio_path else "prompt_audio",
            "seconds": round(dt, 3),
        },
    }


def main() -> None:
    send({"ok": True, "msg": "pocket-tts worker ready"})
    while True:
        msg = recv()
        if msg is None:
            return
        cmd = msg.get("cmd")
        try:
            if cmd == "shutdown":
                send({"ok": True})
                return
            if cmd == "unload":
                global _tts
                _tts = None
                gc.collect()
                send({"ok": True})
                continue
            if cmd == "gen":
                result = _handle_gen(msg.get("request") or {})
                send({"ok": True, "result": result})
                continue
            send({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as e:
            _log(f"ERROR: {e}")
            send({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
