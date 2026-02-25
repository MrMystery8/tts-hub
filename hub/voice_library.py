from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .audio_utils import FfmpegNotFoundError, ffmpeg_convert_to_wav


_VOICE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        frames = int(wf.getnframes())
    duration_s = frames / sr if sr else 0.0
    return sr, float(duration_s)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, data)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class VoiceSummary:
    id: str
    name: str
    created_at: int
    duration_s: float
    has_caches: dict[str, bool]


class VoiceLibrary:
    def __init__(self, *, hub_root: Path):
        self.hub_root = hub_root
        self.voices_root = hub_root / "outputs" / "voices"
        self.voices_root.mkdir(parents=True, exist_ok=True)

    def _voice_dir(self, voice_id: str) -> Path:
        voice_id = (voice_id or "").strip().lower()
        if not _VOICE_ID_RE.match(voice_id):
            raise ValueError("invalid voice_id")
        return (self.voices_root / voice_id).resolve()

    def _meta_path(self, voice_id: str) -> Path:
        return self._voice_dir(voice_id) / "meta.json"

    def _prompt_wav_path(self, voice_id: str) -> Path:
        return self._voice_dir(voice_id) / "prompt.wav"

    def list_voices(self) -> list[VoiceSummary]:
        voices: list[VoiceSummary] = []
        for d in sorted(self.voices_root.iterdir()) if self.voices_root.exists() else []:
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _load_json(meta_path)
                caches = meta.get("caches") or {}
                has_caches = {
                    "chatterbox-multilingual": bool((caches.get("chatterbox-multilingual") or {}).get("path")),
                    "index-tts2": bool((caches.get("index-tts2") or {}).get("path")),
                }
                voices.append(
                    VoiceSummary(
                        id=str(meta.get("id") or d.name),
                        name=str(meta.get("name") or d.name),
                        created_at=int(meta.get("created_at") or 0),
                        duration_s=float((meta.get("audio") or {}).get("duration_s") or 0.0),
                        has_caches=has_caches,
                    )
                )
            except Exception:
                continue
        return voices

    def get_voice_meta(self, voice_id: str) -> dict[str, Any]:
        meta_path = self._meta_path(voice_id)
        if not meta_path.exists():
            raise FileNotFoundError("voice not found")
        return _load_json(meta_path)

    def get_voice_audio_path(self, voice_id: str) -> Path:
        wav_path = self._prompt_wav_path(voice_id)
        if not wav_path.exists():
            raise FileNotFoundError("voice audio not found")
        return wav_path

    def create_voice(self, *, name: str, input_bytes: bytes, filename: str | None) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("name is required")

        voice_id = uuid.uuid4().hex
        voice_dir = self._voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=False)

        suffix = Path(filename or "").suffix or ".bin"
        raw_path = voice_dir / f"input{suffix}"
        raw_path.write_bytes(input_bytes)

        prompt_wav = voice_dir / "prompt.wav"
        try:
            ffmpeg_convert_to_wav(input_path=raw_path, output_path=prompt_wav, channels=1)
        except FfmpegNotFoundError:
            # fallback for environments without ffmpeg: only allow already-wav input
            if suffix.lower() == ".wav" and raw_path.exists():
                shutil.copyfile(raw_path, prompt_wav)
            else:
                raise
        finally:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass

        audio_sha = _sha256_file(prompt_wav)
        sr, duration_s = _wav_info(prompt_wav)

        meta = {
            "id": voice_id,
            "name": name,
            "created_at": int(time.time()),
            "audio": {
                "path": "prompt.wav",
                "sha256": audio_sha,
                "sr": sr,
                "duration_s": round(duration_s, 3),
            },
            "caches": {},
        }
        _atomic_write_json(voice_dir / "meta.json", meta)
        return meta

    def delete_voice(self, voice_id: str) -> None:
        voice_dir = self._voice_dir(voice_id)
        if not voice_dir.exists():
            return
        # Safety: ensure we're deleting within voices_root.
        if self.voices_root.resolve() not in voice_dir.parents:
            raise RuntimeError("refusing to delete outside voices_root")
        shutil.rmtree(voice_dir)

    def ensure_audio_meta(self, voice_id: str) -> dict[str, Any]:
        """
        Recompute sha256/sr/duration from prompt.wav and update meta.json if it changed.
        Useful if prompt.wav is replaced manually on disk.
        """
        wav_path = self.get_voice_audio_path(voice_id)
        actual_sha = _sha256_file(wav_path)
        sr, duration_s = _wav_info(wav_path)

        def _upd(meta: dict[str, Any]) -> dict[str, Any]:
            audio = dict(meta.get("audio") or {})
            if audio.get("sha256") != actual_sha or audio.get("sr") != sr or audio.get("duration_s") != round(duration_s, 3):
                audio.update(
                    {
                        "path": "prompt.wav",
                        "sha256": actual_sha,
                        "sr": sr,
                        "duration_s": round(duration_s, 3),
                    }
                )
                meta["audio"] = audio
            return meta

        return self.update_voice_meta(voice_id, _upd)

    def update_voice_meta(self, voice_id: str, update_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        meta_path = self._meta_path(voice_id)
        if not meta_path.exists():
            raise FileNotFoundError("voice not found")
        meta = _load_json(meta_path)
        meta2 = update_fn(dict(meta))
        _atomic_write_json(meta_path, meta2)
        return meta2
