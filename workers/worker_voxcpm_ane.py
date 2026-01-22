from __future__ import annotations

import gc
import sys
import time
import wave
from pathlib import Path
from typing import Any

from _worker_protocol import recv, send

_server = None


def _log(msg: str) -> None:
    print(f"[voxcpm-ane] {msg}", file=sys.stderr, flush=True)


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


def _load_server():
    global _server
    if _server is not None:
        return _server
    _log("Importing voxcpmane.server (loads CoreML models)...")
    import voxcpmane.server as server

    _server = server
    return _server


def _handle_gen(req: dict[str, Any]) -> dict[str, Any]:
    fields = dict(req.get("fields") or {})
    hub_root = Path(req.get("hub_root") or ".").resolve()
    out_dir = hub_root / "outputs" / "voxcpm-ane"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"voxcpm_{request_id}.wav"

    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    prompt_audio_path = req.get("prompt_audio_path")
    prompt_text = (fields.get("prompt_text") or "").strip()

    voice = (fields.get("voice") or "").strip() or None
    max_length = _int(fields.get("max_length"), 2048)
    cfg_value = _float(fields.get("cfg_value"), 2.0)
    inference_timesteps = _int(fields.get("inference_timesteps"), 10)

    if voice is None:
        if not prompt_audio_path:
            raise ValueError("VoxCPM-ANE requires prompt_audio (or a cached voice name).")
        if not prompt_text:
            raise ValueError("VoxCPM-ANE requires prompt_text (transcript) with prompt_audio.")

    server = _load_server()

    import numpy as np
    import threading

    chunks = []
    t0 = time.time()
    for chunk in server.generate_audio_chunks(
        text_to_generate=text,
        prompt_wav_path=str(prompt_audio_path) if prompt_audio_path else "",
        prompt_text=prompt_text,
        voice=voice,
        max_length=int(max_length),
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
        cancellation_event=threading.Event(),
    ):
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No audio produced.")

    audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    write_wav_mono_int16(out_path, audio, int(server.SAMPLE_RATE))
    dt = time.time() - t0

    gc.collect()
    return {
        "output_path": str(out_path),
        "meta": {
            "model": "VoxCPM-ANE",
            "sr": int(server.SAMPLE_RATE),
            "seconds": round(dt, 3),
        },
    }


def main() -> None:
    send({"ok": True, "msg": "voxcpm-ane worker ready"})
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
                global _server
                _server = None
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
