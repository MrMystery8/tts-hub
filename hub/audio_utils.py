from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    if not has_ffmpeg():
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install it (macOS: `brew install ffmpeg`) and retry."
        )


def ffmpeg_convert_to_wav(
    *,
    input_path: Path,
    output_path: Path,
    sample_rate: int | None = None,
    channels: int | None = 1,
) -> None:
    """
    Convert arbitrary audio input to PCM WAV via ffmpeg.

    Args:
        input_path: Source audio file path.
        output_path: Destination .wav path.
        sample_rate: If set, resample to this Hz.
        channels: If set, force channel count (default mono).
    """
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    if channels is not None:
        cmd += ["-ac", str(int(channels))]
    if sample_rate is not None:
        cmd += ["-ar", str(int(sample_rate))]
    cmd += ["-f", "wav", str(output_path)]

    subprocess.run(cmd, check=True)


def ffmpeg_convert_output(
    *,
    input_wav_path: Path,
    output_path: Path,
) -> None:
    """Convert WAV to another format via ffmpeg based on output_path extension."""
    require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_wav_path), str(output_path)],
        check=True,
    )

