from __future__ import annotations

import gc
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

# Must be set before torch import for best MPS behavior
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("METAL_DEVICE_WRAPPER_TYPE", "0")

from _worker_protocol import recv, send

_model = None
_device: str | None = None
_df_model = None
_df_state = None
_novasr = None


def _log(msg: str) -> None:
    print(f"[chatterbox-mtl] {msg}", file=sys.stderr, flush=True)


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


def _get_device() -> str:
    import torch

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _clear_mps() -> None:
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()

def _get_deepfilter():
    global _df_model, _df_state
    if _df_model is not None:
        return _df_model, _df_state
    try:
        from df.enhance import init_df

        _df_model, _df_state, _ = init_df()
        _log("DeepFilterNet loaded")
        return _df_model, _df_state
    except Exception as e:
        raise RuntimeError(f"DeepFilterNet not available: {e}")


def _deepfilter_denoise(audio: "np.ndarray", sr: int) -> tuple["np.ndarray", int]:
    import numpy as np
    import torch
    import torchaudio
    from df.enhance import enhance

    model, state = _get_deepfilter()
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    if sr != 48000:
        x = torchaudio.transforms.Resample(sr, 48000)(x)
        sr = 48000
    y = enhance(model, state, x)
    return y.squeeze(0).cpu().numpy().astype(np.float32, copy=False), sr


def _get_novasr():
    global _novasr
    if _novasr is not None:
        return _novasr
    try:
        from NovaSR import FastSR

        _novasr = FastSR(half=False)
        _log("NovaSR loaded")
        return _novasr
    except Exception as e:
        raise RuntimeError(f"NovaSR not available: {e}")


def _novasr_upscale(audio: "np.ndarray", sr: int) -> tuple["np.ndarray", int]:
    import numpy as np
    import torch
    import torchaudio

    model = _get_novasr()
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if sr != 16000:
        x = torchaudio.transforms.Resample(sr, 16000)(x)
        sr = 16000
    # NovaSR expects (1, 1, samples)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() == 3 and x.shape[1] != 1:
        x = x[:, :1, :]
    y = model.infer(x).cpu().squeeze().numpy().astype(np.float32, copy=False)
    return y, 48000


def _patch_torch_load(device: str) -> None:
    import torch

    map_location = torch.device("cpu") if device == "mps" else torch.device(device)
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", map_location)
        return original_load(*args, **kwargs)

    torch.load = patched_load


def _load_model():
    global _model, _device
    if _model is not None:
        return _model

    _device = _get_device()
    _patch_torch_load(_device)

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    t0 = time.time()
    _log(f"Loading model on {_device} ...")
    _model = ChatterboxMultilingualTTS.from_pretrained(device=_device)
    _clear_mps()
    _log(f"Loaded in {time.time() - t0:.2f}s (sr={getattr(_model, 'sr', '?')})")
    return _model


def split_text_into_chunks(text: str, *, max_chars: int, language_id: str) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    delimiters = {
        "hi": ["।", "!", "?", "।।"],
        "zh": ["。", "！", "？"],
        "ja": ["。", "！", "？"],
        "ar": [".", "!", "?", "؟"],
        "default": [".", "!", "?", ";", ":"],
    }
    sentence_ends = delimiters.get(language_id, delimiters["default"])

    chunks: list[str] = []
    current = ""
    for ch in text:
        current += ch
        is_sentence_end = any(current.rstrip().endswith(d) for d in sentence_ends)
        if is_sentence_end and len(current) >= 50:
            chunks.append(current.strip())
            current = ""
        elif len(current) >= max_chars:
            last_space = current.rfind(" ")
            if last_space > max_chars // 2:
                chunks.append(current[:last_space].strip())
                current = current[last_space:].strip()
            else:
                chunks.append(current.strip())
                current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def stitch_audio_chunks(chunks: list["np.ndarray"], *, sr: int, crossfade_ms: int) -> "np.ndarray":
    import numpy as np

    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32, copy=False)

    overlap = max(0, int(sr * crossfade_ms / 1000))
    result = chunks[0].astype(np.float32, copy=False)

    for nxt in chunks[1:]:
        nxt = nxt.astype(np.float32, copy=False)
        if overlap <= 0:
            result = np.concatenate([result, nxt])
            continue

        ov = min(overlap, len(result), len(nxt))
        if ov <= 0:
            result = np.concatenate([result, nxt])
            continue

        fade_out = np.linspace(1.0, 0.0, ov, dtype=np.float32)
        fade_in = 1.0 - fade_out
        blended = result[-ov:] * fade_out + nxt[:ov] * fade_in
        result = np.concatenate([result[:-ov], blended, nxt[ov:]])

    return result


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
    out_dir = hub_root / "outputs" / "chatterbox-multilingual"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"chatterbox_{request_id}.wav"

    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    language_id = (fields.get("language_id") or "hi").strip()
    cfg_weight = _float(fields.get("cfg_weight"), 0.5)
    temperature = _float(fields.get("temperature"), 0.8)
    exaggeration = _float(fields.get("exaggeration"), 0.5)
    fast_mode = _bool(fields.get("fast_mode"), False)

    enable_chunking = _bool(fields.get("enable_chunking"), True)
    max_chunk_chars = _int(fields.get("max_chunk_chars"), 150)
    crossfade_ms = _int(fields.get("crossfade_ms"), 50)

    enable_df = _bool(fields.get("enable_df"), False)
    enable_novasr = _bool(fields.get("enable_novasr"), False)

    prompt_audio_path = req.get("prompt_audio_path")
    use_prompt = _bool(fields.get("use_prompt_audio"), bool(prompt_audio_path))

    model = _load_model()

    chunks = [text]
    if enable_chunking:
        chunks = split_text_into_chunks(text, max_chars=max_chunk_chars, language_id=language_id)

    import numpy as np

    audio_chunks: list[np.ndarray] = []
    t0 = time.time()
    for chunk in chunks:
        kwargs = {
            "language_id": language_id,
            "cfg_weight": float(cfg_weight),
            "temperature": float(temperature),
            "exaggeration": float(exaggeration),
            "fast_mode": bool(fast_mode),
        }
        if use_prompt and prompt_audio_path:
            kwargs["audio_prompt_path"] = str(prompt_audio_path)

        wav = model.generate(chunk, **kwargs)
        chunk_audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        audio_chunks.append(chunk_audio)

    if len(audio_chunks) > 1:
        final_audio = stitch_audio_chunks(audio_chunks, sr=int(model.sr), crossfade_ms=crossfade_ms)
    else:
        final_audio = audio_chunks[0]

    out_sr = int(model.sr)

    # Optional enhancement pipeline (as in the local optimized UI)
    if enable_df:
        final_audio, out_sr = _deepfilter_denoise(final_audio, out_sr)
    if enable_novasr:
        final_audio, out_sr = _novasr_upscale(final_audio, out_sr)

    if _device == "mps":
        _clear_mps()

    write_wav_mono_int16(out_path, final_audio, int(out_sr))
    dt = time.time() - t0

    return {
        "output_path": str(out_path),
        "meta": {
            "model": "ChatterboxMultilingualTTS",
            "device": _device,
            "sr": int(out_sr),
            "chunks": len(audio_chunks),
            "seconds": round(dt, 3),
        },
    }


def main() -> None:
    send({"ok": True, "msg": "chatterbox-multilingual worker ready"})
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
                global _model, _device
                _model = None
                _device = None
                global _df_model, _df_state, _novasr
                _df_model = None
                _df_state = None
                _novasr = None
                _clear_mps()
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
