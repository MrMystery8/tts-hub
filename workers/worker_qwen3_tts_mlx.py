from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

from _worker_protocol import recv, send

_model = None
_model_id: str | None = None

# key = (voice_id or "", source_sha256, model_id)
_ref_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
ALT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"

DEFAULT_STT_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"


def _log(msg: str) -> None:
    print(f"[qwen3-tts-mlx] {msg}", file=sys.stderr, flush=True)


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _update_voice_meta(*, voice_dir: Path, update_fn) -> None:
    meta_path = voice_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta2 = update_fn(dict(meta))
        _atomic_write_json(meta_path, meta2)
    except Exception:
        return


def _clear_mlx() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _load_model(model_id: str):
    global _model, _model_id
    model_id = (model_id or "").strip() or DEFAULT_TTS_MODEL
    if model_id not in {DEFAULT_TTS_MODEL, ALT_TTS_MODEL}:
        raise ValueError("Unsupported qwen_model (8-bit only)")

    if _model is not None and _model_id == model_id:
        return _model

    _log(f"Loading model: {model_id}")
    from mlx_audio.tts import load

    _model = load(model_id)
    _model_id = model_id
    _clear_mlx()
    return _model


def _trim_silence(audio: "np.ndarray", sr: int) -> "np.ndarray":
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.size == 0 or sr <= 0:
        return audio

    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(abs_audio**2))) if abs_audio.size else 0.0
    thr = max(0.01, min(0.05, rms * 0.5))

    idx = np.nonzero(abs_audio > thr)[0]
    if idx.size == 0:
        return audio

    pad = int(sr * 0.10)
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad + 1, int(audio.shape[0]))
    if end <= start:
        return audio
    return audio[start:end]


def _write_wav_mono_int16(path: Path, audio: "np.ndarray", sr: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def _prepare_ref_prompt(
    *,
    hub_root: Path,
    request_id: str,
    prompt_audio_path: Path,
    voice_id: str | None,
    model_id: str,
    max_seconds: float,
) -> tuple[Path, str | None, str]:
    """
    Returns: (ref_audio_wav_path, audio_sha256_or_none, ref_audio_cache_status)
    """
    audio_sha = None
    try:
        audio_sha = _sha256_file(prompt_audio_path)
    except Exception:
        audio_sha = None

    cache_key = (voice_id or "", audio_sha or "", model_id)
    if audio_sha and model_id and cache_key in _ref_cache:
        ref_path = Path(str(_ref_cache[cache_key].get("ref_audio_path") or ""))
        if ref_path.exists():
            return ref_path, audio_sha, "hit_mem"

    # For saved voices, store trimmed prompt in voice caches. For uploads, store alongside prompt.
    if voice_id:
        voice_dir = (hub_root / "outputs" / "voices" / voice_id).resolve()
        cache_dir = voice_dir / "caches" / "qwen3-tts-mlx"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ref_path = cache_dir / "prompt_24k_trim.wav"
    else:
        ref_path = prompt_audio_path.with_name("prompt_24k_trim.wav")

    if voice_id and audio_sha:
        # If the disk cache is already valid per meta, reuse it.
        try:
            meta = json.loads((hub_root / "outputs" / "voices" / voice_id / "meta.json").read_text(encoding="utf-8"))
            cached = (meta.get("caches") or {}).get("qwen3-tts-mlx") or {}
            if (
                ref_path.exists()
                and cached.get("source_sha256") == audio_sha
                and cached.get("model_id") == model_id
                and cached.get("prompt_trim_path") == "caches/qwen3-tts-mlx/prompt_24k_trim.wav"
            ):
                _ref_cache[cache_key] = {"ref_audio_path": str(ref_path)}
                return ref_path, audio_sha, "hit_disk"
        except Exception:
            pass

    # Build (resample -> trim -> cap -> write)
    from mlx_audio.tts.generate import load_audio
    import numpy as np

    # Qwen3-TTS models use 24k in practice; we rely on mlx-audio to resample to the model SR.
    model = _load_model(model_id)
    target_sr = int(getattr(model, "sample_rate", 24000))
    ref_audio = load_audio(str(prompt_audio_path), sample_rate=target_sr)
    ref_audio = np.asarray(ref_audio, dtype=np.float32).squeeze()

    ref_audio = _trim_silence(ref_audio, target_sr)
    if max_seconds > 0:
        max_len = int(target_sr * float(max_seconds))
        if ref_audio.shape[0] > max_len:
            ref_audio = ref_audio[:max_len]

    _write_wav_mono_int16(ref_path, ref_audio, target_sr)
    if audio_sha:
        _ref_cache[cache_key] = {"ref_audio_path": str(ref_path)}
    return ref_path, audio_sha, "miss_built"


def _auto_transcribe(*, audio_path: Path) -> str:
    stt_model_id = (os.getenv("QWEN3_TTS_STT_MODEL") or DEFAULT_STT_MODEL).strip()
    _log(f"Auto-transcribe with STT model: {stt_model_id}")
    try:
        from mlx_audio.stt import load as load_stt
    except Exception as e:
        raise RuntimeError(f"mlx_audio.stt not available: {e}") from e

    model = load_stt(stt_model_id)
    try:
        def _extract_text(result: Any) -> str:
            if isinstance(result, str):
                return result.strip()
            if isinstance(result, dict):
                return str(result.get("text", "")).strip()
            if hasattr(result, "text"):
                return str(getattr(result, "text") or "").strip()
            return str(result).strip()

        # Some mlx-audio STT models accept a file path; others expect audio arrays.
        try:
            result = model.generate(str(audio_path))
        except Exception:
            from mlx_audio.tts.generate import load_audio

            audio = load_audio(str(audio_path), sample_rate=16000)
            result = model.generate(audio)

        text = _extract_text(result)
        if not text:
            raise RuntimeError("Empty transcript from STT")
        return text
    finally:
        try:
            del model
        except Exception:
            pass
        _clear_mlx()


def _handle_gen(req: dict[str, Any]) -> dict[str, Any]:
    fields = dict(req.get("fields") or {})
    hub_root = Path(req.get("hub_root") or ".").resolve()
    out_dir = hub_root / "outputs" / "qwen3-tts-mlx"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"qwen3_{request_id}.wav"

    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    prompt_audio_path = req.get("prompt_audio_path")
    if not prompt_audio_path:
        raise ValueError("Qwen3-TTS requires prompt_audio (reference voice).")
    prompt_audio_path = Path(str(prompt_audio_path)).resolve()

    voice_id = (fields.get("voice_id") or "").strip().lower() or None
    if voice_id and (len(voice_id) != 32 or any(c not in "0123456789abcdef" for c in voice_id)):
        voice_id = None

    qwen_model = (fields.get("qwen_model") or "").strip() or DEFAULT_TTS_MODEL
    auto_transcribe = _bool(fields.get("auto_transcribe"), True)

    language = (fields.get("qwen_language") or "auto").strip() or "auto"
    temperature = _float(fields.get("qwen_temperature"), 0.7)
    max_tokens = _int(fields.get("qwen_max_tokens"), 1200)
    top_k = _int(fields.get("qwen_top_k"), 50)
    top_p = _float(fields.get("qwen_top_p"), 1.0)
    repetition_penalty = _float(fields.get("qwen_repetition_penalty"), 1.05)
    speed = _float(fields.get("qwen_speed"), 1.0)

    max_seconds = _float(os.getenv("QWEN3_PROMPT_MAX_SECONDS"), 8.0)

    # Ensure model loaded early so we know its SR for preprocessing.
    model = _load_model(qwen_model)

    ref_wav_path, audio_sha, ref_audio_cache_status = _prepare_ref_prompt(
        hub_root=hub_root,
        request_id=request_id,
        prompt_audio_path=prompt_audio_path,
        voice_id=voice_id,
        model_id=qwen_model,
        max_seconds=max_seconds,
    )

    ref_text = (fields.get("prompt_text") or "").strip()
    ref_transcript_status = "provided" if ref_text else "missing"
    if not ref_text:
        if auto_transcribe:
            t0 = time.time()
            ref_text = _auto_transcribe(audio_path=ref_wav_path)
            ref_transcript_status = "auto_transcribed"
            _log(f"Auto-transcribed in {time.time() - t0:.2f}s")
        else:
            ref_transcript_status = "missing_error"
            raise ValueError("prompt_text required (enable auto_transcribe or provide transcript)")

    # If this is a saved voice, persist transcript and cache metadata into meta.json.
    if voice_id:
        voice_dir = (hub_root / "outputs" / "voices" / voice_id).resolve()

        def _upd(meta: dict[str, Any]) -> dict[str, Any]:
            if ref_transcript_status == "auto_transcribed" and not (meta.get("prompt_text") or "").strip():
                meta["prompt_text"] = ref_text
                meta["prompt_text_source"] = "auto"
            caches = dict(meta.get("caches") or {})
            if audio_sha:
                caches["qwen3-tts-mlx"] = {
                    "prompt_trim_path": "caches/qwen3-tts-mlx/prompt_24k_trim.wav",
                    "source_sha256": audio_sha,
                    "model_id": qwen_model,
                    "prepared_at": int(time.time()),
                }
                meta["caches"] = caches
            return meta

        _update_voice_meta(voice_dir=voice_dir, update_fn=_upd)

    # Generate (match mlx-audio's signature; pass only supported kwargs).
    sig = inspect.signature(model.generate)
    kwargs: dict[str, Any] = {}
    if "lang_code" in sig.parameters:
        kwargs["lang_code"] = language
    if "ref_audio" in sig.parameters:
        kwargs["ref_audio"] = str(ref_wav_path)
    if "ref_text" in sig.parameters:
        kwargs["ref_text"] = ref_text
    if "temperature" in sig.parameters:
        kwargs["temperature"] = float(temperature)
    if "max_tokens" in sig.parameters:
        kwargs["max_tokens"] = int(max_tokens)
    if "top_k" in sig.parameters:
        kwargs["top_k"] = int(top_k)
    if "top_p" in sig.parameters:
        kwargs["top_p"] = float(top_p)
    if "repetition_penalty" in sig.parameters:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    if "speed" in sig.parameters:
        kwargs["speed"] = float(speed)

    import numpy as np

    t0 = time.time()
    chunks: list[np.ndarray] = []
    out_sr = int(getattr(model, "sample_rate", 24000))
    for r in model.generate(text, **kwargs):
        chunks.append(np.asarray(r.audio, dtype=np.float32))
        out_sr = int(getattr(r, "sample_rate", out_sr))
    if not chunks:
        raise RuntimeError("No audio generated.")
    audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    _write_wav_mono_int16(out_path, audio, out_sr)
    dt = time.time() - t0

    _clear_mlx()
    return {
        "output_path": str(out_path),
        "meta": {
            "model": "Qwen3-TTS-MLX",
            "model_id": qwen_model,
            "sr": int(out_sr),
            "seconds": round(float(dt), 3),
            "ref_transcript_status": ref_transcript_status,
            "ref_audio_cache_status": ref_audio_cache_status,
        },
    }


def main() -> None:
    send({"ok": True, "msg": "qwen3-tts-mlx worker ready"})
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
                global _model, _model_id, _ref_cache
                _model = None
                _model_id = None
                _ref_cache = {}
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
