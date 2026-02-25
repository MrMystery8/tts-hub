from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# MPS memory guardrails (must be set before torch import)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    # Default to no MPS high-watermark cap for IndexTTS2.
    # Override with INDEXTTS2_MPS_HIGH_WATERMARK_RATIO (or PYTORCH_* directly) if desired.
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = os.getenv(
        "INDEXTTS2_MPS_HIGH_WATERMARK_RATIO", "0.0"
    )

warnings.filterwarnings(
    "ignore",
    message=r"Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4\.53\.0\..*",
    category=FutureWarning,
)

from _worker_protocol import recv, send

_tts = None
_tts_model_dir: Path | None = None


def _log(msg: str) -> None:
    print(f"[index-tts2] {msg}", file=sys.stderr, flush=True)


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


def _parse_emo_vector(s: str | None) -> list[float] | None:
    if not s:
        return None
    s = s.strip()
    try:
        if s.startswith("["):
            vec = json.loads(s)
        else:
            vec = [float(x.strip()) for x in s.split(",") if x.strip()]
        if not isinstance(vec, list) or len(vec) != 8:
            return None
        return [float(x) for x in vec]
    except Exception:
        return None


def _clear_memory() -> None:
    try:
        from indextts.memory_utils import clear_memory

        clear_memory()
    except Exception:
        gc.collect()
        try:
            import torch

            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
        except Exception:
            pass
        gc.collect()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_fingerprint(model_dir: Path) -> str:
    """
    Practical fingerprint: config.yaml bytes + (size, mtime) for large weight files.
    """
    h = hashlib.sha256()
    cfg = model_dir / "config.yaml"
    if cfg.exists():
        h.update(cfg.read_bytes())
    for name in ("gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt"):
        p = model_dir / name
        try:
            st = p.stat()
            h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
        except Exception:
            h.update(f"{name}:missing".encode("utf-8"))
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _update_voice_meta_cache(*, voice_dir: Path, cache_key: str, entry: dict[str, Any]) -> None:
    meta_path = voice_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        caches = dict(meta.get("caches") or {})
        caches[cache_key] = entry
        meta["caches"] = caches
        _atomic_write_json(meta_path, meta)
    except Exception:
        return


def _load_model(model_dir: Path) -> Any:
    global _tts, _tts_model_dir

    if _tts is not None and _tts_model_dir == model_dir:
        return _tts

    # Ensure relative HF cache paths inside IndexTTS resolve correctly.
    os.environ.setdefault("HF_HUB_CACHE", str(model_dir / "hf_cache"))

    _log(f"Loading IndexTTS2 from {model_dir} ...")
    from indextts.infer_v2 import IndexTTS2

    t0 = time.time()
    _tts = IndexTTS2(
        model_dir=str(model_dir),
        cfg_path=str(model_dir / "config.yaml"),
        use_fp16=False,
        use_deepspeed=False,
        use_cuda_kernel=False,
    )
    _tts_model_dir = model_dir

    # Optional per-process MPS memory fraction (unset by default: no hard cap)
    try:
        import torch

        mps_mem_fraction = os.getenv("INDEXTTS2_MPS_MEMORY_FRACTION")
        if (
            mps_mem_fraction
            and hasattr(torch, "mps")
            and torch.backends.mps.is_available()
        ):
            mem_fraction = float(mps_mem_fraction)
            if 0.0 < mem_fraction <= 1.0:
                torch.mps.set_per_process_memory_fraction(mem_fraction)
                _log(f"Applied per-process MPS memory fraction={mem_fraction:.3f}")
            else:
                _log(
                    "Ignored INDEXTTS2_MPS_MEMORY_FRACTION "
                    f"(must be in (0, 1], got {mps_mem_fraction!r})"
                )
    except Exception:
        pass

    _clear_memory()
    _log(f"Loaded in {time.time() - t0:.2f}s on device={getattr(_tts, 'device', 'unknown')}")
    return _tts


def _handle_gen(req: dict[str, Any]) -> dict[str, Any]:
    fields = dict(req.get("fields") or {})
    meta: dict[str, Any] = {}

    hub_root = Path(req.get("hub_root") or ".").resolve()
    out_dir = hub_root / "outputs" / "index-tts2"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"indextts2_{request_id}.wav"

    model_dir = Path(fields.get("model_dir") or Path.cwd() / "checkpoints").resolve()
    tts = _load_model(model_dir)

    text = str(req.get("text") or "").strip()
    prompt_audio_path = req.get("prompt_audio_path")
    if not prompt_audio_path:
        raise ValueError("IndexTTS2 requires prompt_audio (reference voice).")

    voice_id = (fields.get("voice_id") or "").strip().lower() or None
    if voice_id and (len(voice_id) != 32 or any(c not in "0123456789abcdef" for c in voice_id)):
        voice_id = None
    if voice_id:
        voice_dir = (hub_root / "outputs" / "voices" / voice_id).resolve()
        cache_dir = voice_dir / "caches" / "index-tts2"
        cache_st = cache_dir / "speaker_cache_v1.safetensors"
        cache_meta = cache_dir / "speaker_cache_v1.json"

        audio_sha = None
        try:
            audio_sha = _sha256_file(Path(str(prompt_audio_path)).resolve())
        except Exception:
            audio_sha = None
        fp = _model_fingerprint(model_dir)

        loaded = False
        load_ms: float | None = None
        compute_ms: float | None = None
        save_ms: float | None = None
        if audio_sha:
            t_load0 = time.perf_counter()
            loaded = bool(
                tts.try_load_speaker_cache(
                    str(cache_st),
                    str(cache_meta),
                    expected_audio_sha256=audio_sha,
                    expected_model_fingerprint=fp,
                    prompt_path_key=str(prompt_audio_path),
                    emo_path_key=str(prompt_audio_path),
                )
            )
            load_ms = (time.perf_counter() - t_load0) * 1000.0
        if loaded:
            _log(f"Loaded speaker cache for voice_id={voice_id}")
            meta.update(
                {
                    "speaker_cache_status": "hit",
                    "speaker_cache_voice_id": voice_id,
                    "speaker_cache_load_ms": round(float(load_ms or 0.0), 2),
                }
            )
        else:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if audio_sha:
                    t_comp0 = time.perf_counter()
                    tensors = tts.compute_speaker_cache_tensors(
                        spk_audio_prompt=str(prompt_audio_path),
                        emo_audio_prompt=str(prompt_audio_path),
                        verbose=_bool(fields.get("verbose"), False),
                    )
                    compute_ms = (time.perf_counter() - t_comp0) * 1000.0
                    t_save0 = time.perf_counter()
                    tts.save_speaker_cache(
                        str(cache_st),
                        str(cache_meta),
                        audio_sha256=audio_sha,
                        model_fingerprint=fp,
                        tensors=tensors,
                    )
                    save_ms = (time.perf_counter() - t_save0) * 1000.0
                    _update_voice_meta_cache(
                        voice_dir=voice_dir,
                        cache_key="index-tts2",
                        entry={
                            "path": "caches/index-tts2/speaker_cache_v1.safetensors",
                            "meta_path": "caches/index-tts2/speaker_cache_v1.json",
                            "source_sha256": audio_sha,
                            "model_fingerprint": fp,
                            "prepared_at": int(time.time()),
                        },
                    )
                    # Ensure in-memory caches are also set for this run
                    tts.try_load_speaker_cache(
                        str(cache_st),
                        str(cache_meta),
                        expected_audio_sha256=audio_sha,
                        expected_model_fingerprint=fp,
                        prompt_path_key=str(prompt_audio_path),
                        emo_path_key=str(prompt_audio_path),
                    )
                    _log(f"Saved speaker cache for voice_id={voice_id}")
                    meta.update(
                        {
                            "speaker_cache_status": "miss_saved",
                            "speaker_cache_voice_id": voice_id,
                            "speaker_cache_load_ms": round(float(load_ms or 0.0), 2),
                            "speaker_cache_compute_ms": round(float(compute_ms or 0.0), 2),
                            "speaker_cache_save_ms": round(float(save_ms or 0.0), 2),
                        }
                    )
            except Exception as e:
                _log(f"WARNING: failed to build speaker cache: {e}")
                meta.update(
                    {
                        "speaker_cache_status": "miss_failed",
                        "speaker_cache_voice_id": voice_id,
                        "speaker_cache_error": str(e),
                    }
                )
    else:
        meta["speaker_cache_status"] = "none"

    # Emotion controls
    emo_mode = (fields.get("emo_mode") or "speaker").strip()
    emo_audio_path = req.get("emo_audio_path")
    emo_alpha = _float(fields.get("emo_alpha"), 0.65)
    emo_vector = _parse_emo_vector(fields.get("emo_vector"))
    emo_text = (fields.get("emo_text") or "").strip() or None
    use_random = _bool(fields.get("use_random"), False)

    if emo_mode not in {"speaker", "emo_ref", "emo_vector", "emo_text"}:
        emo_mode = "speaker"

    use_emo_text = emo_mode == "emo_text"
    if use_emo_text and not emo_text:
        emo_text = None

    if emo_mode == "emo_ref":
        if not emo_audio_path:
            raise ValueError("emo_mode=emo_ref requires emo_audio upload.")
        emo_audio_prompt = str(emo_audio_path)
    else:
        emo_audio_prompt = None

    if emo_mode == "emo_vector":
        if emo_vector is None:
            raise ValueError("emo_mode=emo_vector requires emo_vector (8 floats).")
    elif emo_mode != "emo_vector":
        # only pass vector in vector mode or text mode (text mode generates vectors internally)
        emo_vector = None

    # Segment controls
    max_text_tokens_per_segment = _int(fields.get("max_text_tokens_per_segment"), 120)
    max_mel_tokens = _int(fields.get("max_mel_tokens"), 1500)

    # Sampling controls (defaults aligned with index-tts webui)
    fast_mode = _bool(fields.get("fast_mode"), False)
    do_sample = _bool(fields.get("do_sample"), True)
    temperature = _float(fields.get("temperature"), 0.8)
    top_p = _float(fields.get("top_p"), 0.8)
    top_k = _int(fields.get("top_k"), 30)
    num_beams = _int(fields.get("num_beams"), 3)
    repetition_penalty = _float(fields.get("repetition_penalty"), 10.0)
    length_penalty = _float(fields.get("length_penalty"), 0.0)

    if fast_mode:
        num_beams = 1
        do_sample = False
        max_mel_tokens = min(max_mel_tokens, 1000)

    generation_kwargs = {
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "length_penalty": float(length_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    t0 = time.time()
    tts.infer(
        spk_audio_prompt=str(prompt_audio_path),
        text=text,
        output_path=str(out_path),
        emo_audio_prompt=emo_audio_prompt,
        emo_alpha=float(emo_alpha),
        emo_vector=emo_vector,
        use_emo_text=bool(use_emo_text),
        emo_text=emo_text,
        use_random=bool(use_random),
        verbose=_bool(fields.get("verbose"), False),
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **generation_kwargs,
    )
    dt = time.time() - t0

    _clear_memory()
    return {
        "output_path": str(out_path),
        "meta": {
            "model": "IndexTTS2",
            "device": str(getattr(tts, "device", "unknown")),
            "seconds": round(dt, 3),
            **meta,
        },
    }


def main() -> None:
    send({"ok": True, "msg": "index-tts2 worker ready"})
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
                global _tts, _tts_model_dir
                _tts = None
                _tts_model_dir = None
                _clear_memory()
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
