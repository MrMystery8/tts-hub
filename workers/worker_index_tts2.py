from __future__ import annotations

import gc
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
