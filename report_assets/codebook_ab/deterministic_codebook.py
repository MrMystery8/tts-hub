"""
Deterministic Codebook Encoder (Decoder-only training)

This module provides a deterministic (non-trained) watermark embedder that maps an
attribution class_id (1..K) to a fixed time-domain signature.

Design goals:
- Stable codewords across retrains/runs (given a fixed key).
- Equal-strength embedding across IDs (by construction).
- MPS-safe: codebook is generated on CPU and registered as buffers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from watermark.config import BUDGET_TARGET_DB, HOP_RATIO, SAMPLE_RATE, WINDOW_SAMPLES


def _seed_u64(*, key: str, idx: int, salt: str = "wm-codebook-v1") -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(salt.encode("utf-8"))
    h.update(b"|")
    h.update(str(idx).encode("utf-8"))
    h.update(b"|")
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def _rademacher(*, length: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    # (length,) in {0,1} -> {-1,+1}
    x = torch.randint(0, 2, (int(length),), generator=g, dtype=torch.int64)
    return (x.to(dtype=torch.float32) * 2.0) - 1.0


def _bandlimit_rfft(x: torch.Tensor, *, sample_rate: int, f_low: float, f_high: float) -> torch.Tensor:
    """
    Apply a hard bandpass mask in the frequency domain.
    x: (T,) float32 (CPU)
    """
    if float(f_low) <= 0 and float(f_high) >= float(sample_rate) / 2:
        return x
    T = int(x.shape[0])
    X = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(sample_rate))
    mask = (freqs >= float(f_low)) & (freqs <= float(f_high))
    X = X * mask.to(dtype=X.dtype)
    y = torch.fft.irfft(X, n=T)
    return y


def _normalize_rms(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    rms = (x.pow(2).mean() + float(eps)).sqrt()
    return x / rms


@dataclass(frozen=True)
class CodebookConfig:
    mode: str = "deterministic_codebook"
    key: str = "default"
    sample_rate: int = int(SAMPLE_RATE)
    window_samples: int = int(WINDOW_SAMPLES)
    hop_ratio: float = float(HOP_RATIO)
    budget_target_db: float = float(BUDGET_TARGET_DB)
    # Signature shaping
    # - multitone: few tones per ID (strong separability, can be more audible)
    # - subband_noise: many tones per ID (noise-like; less "whistly" but still band-limited)
    # - noise_bandpass: legacy noise-like signature (phase-heavy; weak for attribution)
    style: str = "multitone"
    bandpass_low_hz: float = 1200.0
    bandpass_high_hz: float = 7200.0
    tone_step_hz: float = 250.0
    tones_per_id: int = 3
    # Scaling behavior
    scale_floor_rms: float = 1e-4
    # Peak safety: scale watermark down when the carrier is already near full-scale
    # (prevents clipping/harshness during playback/saving).
    peak_limit: float = 0.98

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_path(path: Path) -> "CodebookConfig":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("codebook.json must be an object")
        return CodebookConfig(**obj)


class DeterministicCodebookEncoder(nn.Module):
    """
    Deterministic embedder:
      x_wm = x + residual(class_id)

    Input audio is expected as (B, 1, T) @ 16kHz.
    class_id is expected as (B,) with values in:
      - 0: clean (no watermark)
      - 1..K: attribution classes
    """

    def __init__(self, *, num_classes: int, config: Optional[CodebookConfig] = None):
        super().__init__()
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        self.num_ids = self.num_classes - 1

        cfg = config or CodebookConfig()
        if cfg.sample_rate != int(SAMPLE_RATE):
            raise ValueError("DeterministicCodebookEncoder currently requires 16kHz internal processing")
        if int(cfg.window_samples) != int(WINDOW_SAMPLES):
            raise ValueError("Codebook window must match WINDOW_SAMPLES")
        self.config = cfg

        # Windowing for overlap-add residual.
        self.register_buffer("hann", torch.hann_window(int(cfg.window_samples)).view(1, 1, -1))

        # Precompute signatures on CPU deterministically and store as buffers.
        sigs = []
        sigs.append(torch.zeros(int(cfg.window_samples), dtype=torch.float32))  # class 0
        for class_id in range(1, self.num_classes):
            s = self._make_signature(cfg=cfg, class_id=int(class_id), num_ids=int(self.num_ids))
            sigs.append(s)
        sig = torch.stack(sigs, dim=0).view(self.num_classes, 1, int(cfg.window_samples))  # (C,1,W)
        self.register_buffer("signatures", sig)

    @staticmethod
    def _make_signature(*, cfg: CodebookConfig, class_id: int, num_ids: int) -> torch.Tensor:
        """
        Produce a deterministic signature for a given class_id.

        Important: the decoder uses magnitude spectrograms, so signatures must be distinguishable
        by their magnitude spectrum (phase-only differences won't train attribution well).
        """
        W = int(cfg.window_samples)
        sr = int(cfg.sample_rate)
        style = str(cfg.style or "noise_bandpass").strip().lower()

        if style == "multitone":
            # Fixed frequency grid.
            f0 = float(cfg.bandpass_low_hz)
            f1 = float(cfg.bandpass_high_hz)
            step = float(cfg.tone_step_hz)
            if not (0 < f0 < f1 < (sr / 2)):
                raise ValueError("invalid bandpass range for multitone codebook")
            if step <= 0:
                raise ValueError("tone_step_hz must be > 0")
            tones_per = max(1, int(cfg.tones_per_id))

            grid = []
            f = f0
            while f <= f1 + 1e-6:
                grid.append(f)
                f += step
            if len(grid) < tones_per * (cfg.window_samples // cfg.window_samples) + tones_per * 8:
                # Very conservative safety check: ensure enough bins for 8 IDs at least.
                pass

            # Allocate disjoint tones per ID to guarantee magnitude-spectrum separability.
            start = (int(class_id) - 1) * tones_per
            if start + tones_per > len(grid):
                # Wrap deterministically if grid is too small.
                start = (start % max(1, len(grid)))
            freqs = [grid[(start + i) % len(grid)] for i in range(tones_per)]

            t = torch.arange(W, dtype=torch.float32) / float(sr)
            sig = torch.zeros(W, dtype=torch.float32)
            for i, fhz in enumerate(freqs):
                phase_seed = _seed_u64(key=str(cfg.key), idx=(class_id * 1000 + i), salt="wm-codebook-phase-v1")
                # phase in [0, 2pi)
                phase = (phase_seed % 10_000_000) / 10_000_000.0 * (2.0 * 3.141592653589793)
                sig = sig + torch.sin(2.0 * 3.141592653589793 * float(fhz) * t + float(phase))
            sig = sig / float(len(freqs))
            sig = sig - sig.mean()
            return _normalize_rms(sig)

        if style == "subband_noise":
            # Many phase-randomized tones per ID, allocated inside an ID-specific sub-band.
            # With enough components this becomes noise-like (less annoying than a few stable whistles),
            # while remaining magnitude-separable across IDs (different subband energy profiles).
            f0 = float(cfg.bandpass_low_hz)
            f1 = float(cfg.bandpass_high_hz)
            step = float(cfg.tone_step_hz)
            if not (0 < f0 < f1 < (sr / 2)):
                raise ValueError("invalid bandpass range for subband_noise codebook")
            if step <= 0:
                raise ValueError("tone_step_hz must be > 0")
            tones_per = max(1, int(cfg.tones_per_id))
            if int(num_ids) <= 0:
                raise ValueError("num_ids must be > 0")

            id_idx = int(class_id) - 1
            width = max(1.0, float(f1 - f0))
            sub_w = width / float(num_ids)
            sub0 = f0 + float(id_idx) * sub_w
            sub1 = min(f1, sub0 + sub_w)
            if not (sub1 > sub0):
                sub0 = f0
                sub1 = f1

            grid: list[float] = []
            f = sub0
            while f <= sub1 + 1e-6:
                grid.append(float(f))
                f += step
            if not grid:
                grid = [0.5 * (sub0 + sub1)]

            # Deterministically sample frequencies from this ID's subband to avoid
            # overly regular spacing (which can become tonal).
            g = torch.Generator(device="cpu")
            g.manual_seed(int(_seed_u64(key=str(cfg.key), idx=(class_id * 99991 + tones_per), salt="wm-subband-v1")))
            if tones_per <= len(grid):
                idx = torch.randperm(len(grid), generator=g)[:tones_per].tolist()
                freqs = [grid[int(i)] for i in idx]
            else:
                idx = torch.randint(0, len(grid), (tones_per,), generator=g, dtype=torch.int64).tolist()
                freqs = [grid[int(i)] for i in idx]
            t = torch.arange(W, dtype=torch.float32) / float(sr)
            sig = torch.zeros(W, dtype=torch.float32)
            for i, fhz in enumerate(freqs):
                phase_seed = _seed_u64(key=str(cfg.key), idx=(class_id * 100000 + i), salt="wm-codebook-phase-v2")
                phase = (phase_seed % 10_000_000) / 10_000_000.0 * (2.0 * 3.141592653589793)
                sig = sig + torch.sin(2.0 * 3.141592653589793 * float(fhz) * t + float(phase))
            sig = sig / float(len(freqs))
            sig = sig - sig.mean()
            return _normalize_rms(sig)

        # Default / legacy: noise-like bandpassed signature (phase-heavy; weak for attribution).
        seed = _seed_u64(key=cfg.key, idx=class_id)
        s = _rademacher(length=W, seed=seed)
        s = _bandlimit_rfft(s, sample_rate=sr, f_low=float(cfg.bandpass_low_hz), f_high=float(cfg.bandpass_high_hz))
        s = s - s.mean()
        return _normalize_rms(s)

    def export_codebook(self, *, path: Path) -> None:
        path.write_text(self.config.to_json(), encoding="utf-8")

    @staticmethod
    def load_from_codebook(*, num_classes: int, codebook_path: Path) -> "DeterministicCodebookEncoder":
        cfg = CodebookConfig.from_path(codebook_path)
        return DeterministicCodebookEncoder(num_classes=int(num_classes), config=cfg)

    def forward(self, audio: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        if audio.dim() != 3 or int(audio.shape[1]) != 1:
            raise ValueError(f"expected audio (B,1,T), got {tuple(audio.shape)}")
        if class_id.dim() == 2 and int(class_id.shape[1]) == 1:
            class_id = class_id.squeeze(1)
        if class_id.dim() != 1 or int(class_id.shape[0]) != int(audio.shape[0]):
            raise ValueError(f"expected class_id (B,), got {tuple(class_id.shape)} for audio {tuple(audio.shape)}")

        B, _C, T = audio.shape
        window = int(self.config.window_samples)
        hop = int(window * float(self.config.hop_ratio))
        if hop <= 0:
            raise ValueError("hop must be > 0")

        # Pad to fit an integer number of windows.
        import math

        n_win = max(1, int(math.ceil((T - window) / hop) + 1))
        out_len = (n_win - 1) * hop + window
        pad = max(0, out_len - T)
        x = F.pad(audio, (0, pad)) if pad > 0 else audio

        # Compute number of windows for overlap-add.
        windows = x.unfold(2, window, hop)  # (B,1,N,W)
        _B, _C1, N, W = windows.shape
        assert int(W) == window

        # Select signature per sample and broadcast to all windows.
        cls = class_id.to(dtype=torch.long).clamp(min=0, max=self.num_classes - 1)
        sig = self.signatures.index_select(0, cls)  # (B,1,W)
        sig = sig.unsqueeze(2).expand(B, 1, N, W)  # (B,1,N,W)

        # Apply hann for overlap-add residual.
        sig_h = sig * self.hann  # (B,1,N,W)

        # Fold signature back to (B,1,out_len) and normalize overlap weights.
        sig_flat = sig_h.reshape(B, window, N)  # (B,W,N) for fold with C=1
        code_sum = F.fold(
            sig_flat,
            output_size=(1, out_len),
            kernel_size=(1, window),
            stride=(1, hop),
        )  # (B,1,1,out_len)
        code_sum = code_sum.squeeze(2)  # (B,1,out_len)

        norm_in = (torch.ones((B, 1, N, W), device=audio.device, dtype=audio.dtype) * self.hann).reshape(B, window, N)
        normalizer = F.fold(
            norm_in,
            output_size=(1, out_len),
            kernel_size=(1, window),
            stride=(1, hop),
        ).squeeze(2).clamp(min=1e-8)  # (B,1,out_len)

        code = code_sum / normalizer  # deterministic code waveform

        # Global scaling to meet the budget target (avoids overlap-add scaling skew).
        carrier_rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()  # (B,1,1)
        carrier_rms = torch.clamp(carrier_rms, min=float(self.config.scale_floor_rms))
        ratio = 10.0 ** (float(self.config.budget_target_db) / 10.0)
        target_delta_rms = carrier_rms * (float(ratio) ** 0.5)  # (B,1,1)

        code_rms = (code.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()  # (B,1,1)
        scale = target_delta_rms / code_rms

        # class 0 is clean/no-op
        is_clean = (cls == 0).view(B, 1, 1).to(dtype=audio.dtype, device=audio.device)
        delta = code * scale * (1.0 - is_clean)

        # Peak safety: when the carrier is close to full-scale, even a small watermark can
        # push peaks into clipping, which sounds harsh/painful. Scale delta down to fit
        # within peak_limit without altering the carrier itself.
        peak_limit = float(getattr(self.config, "peak_limit", 0.0) or 0.0)
        if peak_limit > 0:
            peak_x = x.abs().amax(dim=-1, keepdim=True)  # (B,1,1)
            peak_d = delta.abs().amax(dim=-1, keepdim=True)  # (B,1,1)
            headroom = (peak_limit - peak_x).clamp(min=0.0)
            allow = headroom / (peak_d + 1e-12)
            delta = delta * allow.clamp(min=0.0, max=1.0)

        y = x + delta
        y = y[..., :T]
        return y
