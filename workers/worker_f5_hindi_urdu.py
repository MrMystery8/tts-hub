from __future__ import annotations

import gc
import os
import re
import sys
import time
import wave
from pathlib import Path
from typing import Any

from _worker_protocol import recv, send

# Transliteration (ITRANS → Devanagari)
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_vocoder = None
_hindi_model = None


def _log(msg: str) -> None:
    print(f"[f5-hindi-urdu] {msg}", file=sys.stderr, flush=True)


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


def normalize_roman(text: str) -> str:
    text = (text or "").strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def roman_to_deva(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # Heuristic: add 'a' after trailing consonants to reduce halant artifacts.
    words = text.split()
    processed: list[str] = []
    for w in words:
        if w and w[-1].lower() in "bcdfghjklmnpqrstvwxyz":
            w = w + "a"
        processed.append(w)
    text = " ".join(processed)

    try:
        out = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        out = re.sub(r"्(\s)", r"\1", out)
        out = re.sub(r"्$", "", out)
        return out
    except Exception:
        return text


# Default Devanagari overrides for common Urdu phonetics (copied from the local UI).
DEFAULT_DEV_OVERRIDES: dict[str, str] = {
    "फिर": "प्हिर",
    "फल": "प्हल",
    "फूल": "प्हूल",
    "फेर": "प्हेर",
    "जिंदगी": "ज़िंदगी",
    "जरूर": "ज़रूर",
    "जरूरी": "ज़रूरी",
    "जमीन": "ज़मीन",
    "जमाना": "ज़माना",
    "जोर": "ज़ोर",
    "जबान": "ज़बान",
    "जहर": "ज़हर",
    "जख्म": "ज़ख़्म",
    "जिद": "ज़िद",
    "जुल्म": "ज़ुल्म",
    "नजर": "नज़र",
    "नाज": "नाज़",
    "राज": "राज़",
    "आवाज": "आवाज़",
    "साज": "साज़",
    "बाज": "बाज़",
    "अंदाज": "अंदाज़",
    "इजाजत": "इजाज़त",
    "मजा": "मज़ा",
    "सजा": "सज़ा",
    "तराजू": "तराज़ू",
    "कदर": "क़दर",
    "वक्त": "वक़्त",
    "किस्मत": "क़िस्मत",
    "कलम": "क़लम",
    "कब्र": "क़ब्र",
    "करीब": "क़रीब",
    "कसम": "क़सम",
    "कदम": "क़दम",
    "कातिल": "क़ातिल",
    "इकरार": "इक़रार",
    "तकदीर": "तक़दीर",
    "मकसद": "मक़सद",
    "हकीकत": "हक़ीक़त",
    "खुद": "ख़ुद",
    "खुदा": "ख़ुदा",
    "खून": "ख़ून",
    "ख्वाब": "ख़्वाब",
    "खयाल": "ख़याल",
    "खबर": "ख़बर",
    "खत": "ख़त",
    "खास": "ख़ास",
    "खुश": "ख़ुश",
    "खुशी": "ख़ुशी",
    "खामोश": "ख़ामोश",
    "खतरा": "ख़तरा",
    "खराब": "ख़राब",
    "आखिर": "आख़िर",
    "गलत": "ग़लत",
    "गम": "ग़म",
    "गैर": "ग़ैर",
    "गजल": "ग़ज़ल",
    "गुस्सा": "ग़ुस्सा",
    "गरीब": "ग़रीब",
    "गायब": "ग़ायब",
    "फैसला": "फ़ैसला",
    "फिक्र": "फ़िक्र",
    "फर्क": "फ़र्क़",
    "फर्ज": "फ़र्ज़",
    "फायदा": "फ़ायदा",
    "फन": "फ़न",
    "फितरत": "फ़ितरत",
    "तारीफ": "तारीफ़",
    "सिर्फ": "सिर्फ़",
    "साफ": "साफ़",
    "इंसाफ": "इंसाफ़",
    "कैफ": "कैफ़",
    "इश्क": "इश्क़",
    "शेर": "शे'र",
    "वफा": "वफ़ा",
    "बेवफा": "बेवफ़ा",
}


def parse_overrides(text: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not text:
        return overrides
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        src, dst = line.split("=", 1)
        src, dst = src.strip(), dst.strip()
        if src and dst:
            overrides[src] = dst
    return overrides


def apply_dev_overrides(dev_text: str, overrides: dict[str, str]) -> str:
    out = dev_text
    for src, dst in overrides.items():
        out = out.replace(src, dst)
    return out


def roman_to_hindi_for_tts(roman: str, user_override_text: str, *, apply_overrides: bool) -> str:
    roman = normalize_roman(roman)
    if not roman:
        return ""
    dev = roman_to_deva(roman)
    if not apply_overrides:
        return dev
    overrides = dict(DEFAULT_DEV_OVERRIDES)
    overrides.update(parse_overrides(user_override_text))
    return apply_dev_overrides(dev, overrides)


def _load_models() -> tuple[Any, Any]:
    global _vocoder, _hindi_model
    if _vocoder is not None and _hindi_model is not None:
        return _hindi_model, _vocoder

    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    project_root = Path.cwd()
    ckpt = project_root / "models" / "hindi" / "model_2500000.safetensors"
    vocab = project_root / "models" / "hindi" / "vocab.txt"
    model_cfg = {
        "dim": 768,
        "depth": 18,
        "heads": 12,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "conv_layers": 4,
        "pe_attn_head": 1,
    }

    _log("Loading vocoder...")
    _vocoder = load_vocoder()
    _log("Loading Hindi F5 model...")
    _hindi_model = load_model(DiT, model_cfg, str(ckpt), vocab_file=str(vocab))
    return _hindi_model, _vocoder


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
    out_dir = hub_root / "outputs" / "f5-hindi-urdu"
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = req.get("request_id") or str(int(time.time()))
    out_path = out_dir / f"f5_{request_id}.wav"

    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    prompt_audio_path = req.get("prompt_audio_path")
    if not prompt_audio_path:
        raise ValueError("F5 requires prompt_audio (reference voice).")

    ref_text = (fields.get("prompt_text") or "").strip()

    roman_mode = _bool(fields.get("roman_mode"), True)
    overrides_enabled = _bool(fields.get("overrides_enabled"), True)
    overrides_text = fields.get("overrides_text") or ""

    if roman_mode:
        gen_text_hi = roman_to_hindi_for_tts(text, overrides_text, apply_overrides=overrides_enabled)
        ref_text_hi = roman_to_hindi_for_tts(ref_text, overrides_text, apply_overrides=overrides_enabled) if ref_text else ""
    else:
        gen_text_hi = text
        ref_text_hi = ref_text

    seed = _int(fields.get("seed"), -1)
    cross_fade_duration = _float(fields.get("cross_fade_duration"), 0.15)
    nfe_step = _int(fields.get("nfe_step"), 32)
    speed = _float(fields.get("speed"), 1.0)
    remove_silence = _bool(fields.get("remove_silence"), False)

    import torch
    import numpy as np
    import gradio as gr
    import soundfile as sf
    import torchaudio
    import tempfile
    from f5_tts.infer.utils_infer import (
        infer_process,
        preprocess_ref_audio_text,
        remove_silence_for_generated_wav,
        tempfile_kwargs,
    )

    if seed is None or seed < 0 or seed > 2**31 - 1:
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(int(seed))

    model, vocoder = _load_models()

    def show_info(message: str) -> None:
        _log(str(message))

    ref_audio, ref_text_final = preprocess_ref_audio_text(str(prompt_audio_path), ref_text_hi, show_info=show_info)

    t0 = time.time()
    final_wave, final_sr, _spec = infer_process(
        ref_audio,
        ref_text_final,
        gen_text_hi,
        model,
        vocoder,
        cross_fade_duration=float(cross_fade_duration),
        nfe_step=int(nfe_step),
        speed=float(speed),
        show_info=show_info,
        progress=gr.Progress(),
    )

    if remove_silence:
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, final_wave, final_sr)
            remove_silence_for_generated_wav(temp_path)
            wav_t, _ = torchaudio.load(temp_path)
            final_wave = wav_t.squeeze().cpu().numpy()
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    write_wav_mono_int16(out_path, final_wave, int(final_sr))
    dt = time.time() - t0

    gc.collect()
    return {
        "output_path": str(out_path),
        "meta": {
            "model": "F5 Hindi",
            "sr": int(final_sr),
            "seed": int(seed),
            "seconds": round(dt, 3),
            "roman_mode": bool(roman_mode),
        },
    }


def main() -> None:
    send({"ok": True, "msg": "f5-hindi-urdu worker ready"})
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
                global _vocoder, _hindi_model
                _vocoder = None
                _hindi_model = None
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
