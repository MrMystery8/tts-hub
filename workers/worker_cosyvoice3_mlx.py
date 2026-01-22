from __future__ import annotations

import gc
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

from _worker_protocol import recv, send

_model = None
_model_id: str | None = None

COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

AVAILABLE_MODELS = {
    "fp16": "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16",
    "8bit": "mlx-community/Fun-CosyVoice3-0.5B-2512-8bit",
    "4bit": "mlx-community/Fun-CosyVoice3-0.5B-2512-4bit",
}


def _log(msg: str) -> None:
    print(f"[cosyvoice3-mlx] {msg}", file=sys.stderr, flush=True)


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float(v: str | None, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _clear_mlx() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _load_model(model_key: str):
    global _model, _model_id
    model_key = (model_key or "8bit").strip().lower()
    model_id = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS["8bit"])
    if _model is not None and _model_id == model_id:
        return _model

    _log(f"Loading model: {model_id}")
    from mlx_audio.tts.utils import load_model

    _model = load_model(model_id)
    _model_id = model_id
    _clear_mlx()
    return _model


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


def _handle_gen(req: dict[str, Any]) -> dict[str, Any]:
    fields = dict(req.get("fields") or {})
    hub_root = Path(req.get("hub_root") or ".").resolve()
    out_dir = hub_root / "outputs" / "cosyvoice3-mlx"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"cosyvoice3_{request_id}.wav"

    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    prompt_audio_path = req.get("prompt_audio_path")
    if not prompt_audio_path:
        raise ValueError("CosyVoice3 requires prompt_audio (reference voice).")

    mode = (fields.get("mode") or "zero_shot").strip()
    if mode not in {"zero_shot", "cross_lingual", "instruct"}:
        mode = "zero_shot"

    model_key = fields.get("cosy_model") or fields.get("model_variant") or "8bit"
    model = _load_model(str(model_key))

    language = (fields.get("language") or "auto").strip()
    speed = _float(fields.get("speed"), 1.0)

    ref_text = (fields.get("prompt_text") or "").strip()
    instruct_text = (fields.get("instruct_text") or "").strip()

    from mlx_audio.tts.generate import load_audio

    ref_audio = load_audio(str(prompt_audio_path), sample_rate=model.sample_rate)

    if mode == "zero_shot":
        if not ref_text:
            raise ValueError("CosyVoice3 zero_shot requires prompt_text (reference transcript).")
        formatted_ref = f"{COSYVOICE3_PROMPT_PREFIX}{ref_text}"
        gen_kwargs = dict(
            text=text,
            ref_audio=ref_audio,
            ref_text=formatted_ref,
            instruct_text=None,
            speed=float(speed),
            lang_code=language,
            verbose=_bool(fields.get("verbose"), False),
            stt_model=None,
        )
    elif mode == "cross_lingual":
        formatted_text = f"{COSYVOICE3_PROMPT_PREFIX}{text}"
        gen_kwargs = dict(
            text=formatted_text,
            ref_audio=ref_audio,
            ref_text=None,
            instruct_text=None,
            speed=float(speed),
            lang_code=language,
            verbose=_bool(fields.get("verbose"), False),
            stt_model=None,
        )
    else:  # instruct
        if not instruct_text:
            raise ValueError("CosyVoice3 instruct mode requires instruct_text.")
        formatted_instruct = f"You are a helpful assistant. {instruct_text}<|endofprompt|>"
        gen_kwargs = dict(
            text=text,
            ref_audio=ref_audio,
            ref_text=None,
            instruct_text=formatted_instruct,
            speed=float(speed),
            lang_code=language,
            verbose=_bool(fields.get("verbose"), False),
            stt_model=None,
        )

    t0 = time.time()
    results = list(model.generate(**gen_kwargs))
    if not results:
        raise RuntimeError("No audio generated.")

    last = results[-1]
    write_wav_mono_int16(out_path, last.audio, int(last.sample_rate))
    dt = time.time() - t0

    _clear_mlx()
    return {
        "output_path": str(out_path),
        "meta": {
            "model": "CosyVoice3-MLX",
            "model_id": _model_id,
            "sr": int(last.sample_rate),
            "mode": mode,
            "seconds": round(dt, 3),
        },
    }


def main() -> None:
    send({"ok": True, "msg": "cosyvoice3-mlx worker ready"})
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
                global _model, _model_id
                _model = None
                _model_id = None
                _clear_mlx()
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
