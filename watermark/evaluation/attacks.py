"""
Watermark Attacks Module

Implements audio degradations for evaluation and robustness testing.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 6.2.
"""
import torch
import torch.nn.functional as F
import torchaudio
from typing import Callable, Optional
import random
from watermark.config import SEGMENT_SAMPLES, SAMPLE_RATE

# Dictionary of attack functions
ATTACKS = {}

def register_attack(name):
    def decorator(func):
        ATTACKS[name] = func
        return func
    return decorator

@register_attack("clean")
def attack_clean(audio: torch.Tensor) -> torch.Tensor:
    return audio

def apply_attack_safe(audio: torch.Tensor, attack_fn: Callable) -> torch.Tensor:
    """
    Apply attack with strict engineering contract:
    - Input: (1, T) or (F, T) - we assume audio is (1, T)
    - Execution: CPU only (to prevent MPS buffer bugs)
    - Output: (1, T) same length as input
    """
    # 1. Validation
    assert audio.dim() == 2, f"Input must be (C, T), got {audio.shape}"
    T_orig = audio.shape[-1]
    original_device = audio.device
    
    # 2. Force CPU for safety
    audio_cpu = audio.cpu()
    
    # 3. Apply Attack
    try:
        attacked = attack_fn(audio_cpu)
    except Exception as e:
        print(f"Attack failed: {e}")
        return audio # Fallback to identity
        
    # 4. Enforce Contract (Shape & Logic)
    # Ensure it's still (C, T)
    if attacked.dim() == 1:
        attacked = attacked.unsqueeze(0)
        
    # Enforce Length (Crop/Pad)
    T_new = attacked.shape[-1]
    if T_new > T_orig:
        # Crop center
        diff = T_new - T_orig
        start = diff // 2
        attacked = attacked[..., start : start + T_orig]
    elif T_new < T_orig:
        # Pad at end
        attacked = F.pad(attacked, (0, T_orig - T_new))
        
    # 5. Final check
    assert attacked.shape == audio.shape, f"Shape mismatch: {attacked.shape} vs {audio.shape}"
    assert torch.isfinite(attacked).all(), "Attack produced NaN/Inf"
    
    return attacked.to(original_device)

@register_attack("noise_white_20db")
def attack_noise_white(audio: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """Add white noise at specified SNR."""
    # Signal power
    sig_power = audio.pow(2).mean()
    # Noise power based on SNR
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(audio) * noise_power.sqrt()
    return audio + noise

@register_attack("resample_8k")
def attack_resample_8k(audio: torch.Tensor) -> torch.Tensor:
    """Downsample to 8kHz and back to 16kHz."""
    down = torchaudio.functional.resample(audio, SAMPLE_RATE, 8000)
    up = torchaudio.functional.resample(down, 8000, SAMPLE_RATE)
    return up

# For MP3/AAC, we need external tools or torchaudio execution.
# Since we might not have ffmpeg in the environment, we'll try to use
# torchaudio with simple save/load if backend allows, or skip if not available.
# We'll implement a simulation or placeholder if codec is missing.

import shutil

# Resolve the ffmpeg binary once. torchaudio 2.9+ dropped its built-in codec
# backend, so we drive the system ffmpeg directly. If it is missing, codec
# attacks fall back to identity (the tiered-eval harness flags that as no-op).
FFMPEG_BIN = shutil.which("ffmpeg")

# Hard wall-clock cap per ffmpeg call. Without this a wedged/suspended ffmpeg
# blocks subprocess.run() forever and hangs the whole eval (observed: a single
# call frozen 6+ hours stalling an overnight sweep). On timeout the child is
# killed and the clip falls back to identity.
CODEC_TIMEOUT_SEC = 20


class CodecAttack:
    """Lossy codec round-trip (encode then decode) via the system ffmpeg binary.

    ext: container/codec to encode through ("mp3" or "m4a" for AAC).
    bitrate: target audio bitrate string, e.g. "128k".
    """

    def __init__(self, ext: str = "mp3", bitrate: str = "128k"):
        self.ext = "mp3" if ext == "mp3" else "m4a"
        self.bitrate = bitrate

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if FFMPEG_BIN is None:
            return audio  # no ffmpeg -> identity (flagged as no-op upstream)

        import tempfile
        import subprocess
        from pathlib import Path
        import soundfile as sf

        was_1d = audio.dim() == 1
        x = audio.unsqueeze(0) if was_1d else audio  # (C, T)
        # soundfile expects (T,) mono or (T, C); we use mono.
        mono = x[0].detach().cpu().numpy()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            win = tmp / "in.wav"
            enc = tmp / f"a.{self.ext}"
            wout = tmp / "out.wav"
            try:
                sf.write(str(win), mono, SAMPLE_RATE)
                codec = ["-b:a", str(self.bitrate)]
                if self.ext == "m4a":
                    codec = ["-c:a", "aac", *codec]
                subprocess.run(
                    [FFMPEG_BIN, "-nostdin", "-y", "-loglevel", "error", "-i", str(win), *codec, str(enc)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=CODEC_TIMEOUT_SEC,
                )
                subprocess.run(
                    [FFMPEG_BIN, "-nostdin", "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(SAMPLE_RATE), str(wout)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=CODEC_TIMEOUT_SEC,
                )
                y, _sr = sf.read(str(wout))
            except Exception:
                return audio  # any failure -> identity fallback

        out = torch.from_numpy(y).to(dtype=torch.float32)
        if out.dim() == 1:
            out = out.unsqueeze(0)  # (1, T)
        return out[0] if was_1d else out


@register_attack("mp3_128k")
def attack_mp3_128k(audio: torch.Tensor) -> torch.Tensor:
    return CodecAttack("mp3", "128k")(audio)

@register_attack("aac_128k")
def attack_aac_128k(audio: torch.Tensor) -> torch.Tensor:
    return CodecAttack("m4a", "128k")(audio)

# =============================================================================
# Pinned per-tier parameters (IR Table 3.4: three-tier transformation suite).
# Kept explicit so the evaluation envelope is reproducible, not ad-hoc.
# =============================================================================
TRIM_FRAC = 0.08            # T1: crop 8% from each end (desync / silence trim)
LOUDNORM_TARGET_RMS = 0.06  # T2: normalise to fixed RMS (loudness shift)
LOUDNORM_GAIN_CLAMP = (0.25, 4.0)
EQ_BASS_GAIN_DB = 4.0       # T2: mild spectral colouration
EQ_TREBLE_GAIN_DB = -4.0
BG_MIX_SNR_DB = 15.0        # T2: ambient (low-pass) background at 15 dB SNR
BG_MIX_CUTOFF_HZ = 1000.0
DENOISE_REDUCTION = 0.6     # T2: spectral-subtraction strength (0..1)
DENOISE_N_FFT = 512
DENOISE_HOP = 128
RECAPTURE_NOISE_SNR_DB = 20.0  # T3: simulated playback-and-recapture


@register_attack("trim")
def attack_trim(audio: torch.Tensor) -> torch.Tensor:
    """T1: crop a fraction off each end (desync / silence trimming).

    apply_attack_safe re-pads to the original length, so the net effect is a
    temporal shift + lost edges rather than a length change.
    """
    _c, t = audio.shape
    cut = int(TRIM_FRAC * t)
    if cut <= 0 or 2 * cut >= t:
        return audio
    return audio[..., cut : t - cut]


@register_attack("loudnorm")
def attack_loudnorm(audio: torch.Tensor) -> torch.Tensor:
    """T2: loudness normalisation to a fixed RMS target (bounded gain)."""
    rms = audio.pow(2).mean().sqrt()
    gain = float(LOUDNORM_TARGET_RMS) / float(rms.item() + 1e-8)
    lo, hi = LOUDNORM_GAIN_CLAMP
    gain = max(lo, min(hi, gain))
    return audio * gain


@register_attack("eq")
def attack_eq(audio: torch.Tensor) -> torch.Tensor:
    """T2: mild EQ colouration (bass boost + treble cut)."""
    x = torchaudio.functional.bass_biquad(audio, SAMPLE_RATE, gain=float(EQ_BASS_GAIN_DB))
    x = torchaudio.functional.treble_biquad(x, SAMPLE_RATE, gain=float(EQ_TREBLE_GAIN_DB))
    return x


@register_attack("bg_mix")
def attack_bg_mix(audio: torch.Tensor) -> torch.Tensor:
    """T2: mix in low-pass (ambient-like) background noise at a fixed SNR.

    Distinct from T1 `noise_white_20db`: coloured, lower-frequency content that
    overlaps speech energy rather than broadband hiss.
    """
    noise = torch.randn_like(audio)
    noise = torchaudio.functional.lowpass_biquad(noise, SAMPLE_RATE, cutoff_freq=float(BG_MIX_CUTOFF_HZ))
    sig_power = audio.pow(2).mean()
    noise_power = sig_power / (10 ** (float(BG_MIX_SNR_DB) / 10))
    cur = noise.pow(2).mean()
    noise = noise * (noise_power / (cur + 1e-12)).sqrt()
    return audio + noise


@register_attack("denoise")
def attack_denoise(audio: torch.Tensor) -> torch.Tensor:
    """T2: mild spectral-subtraction denoising.

    Estimates a per-frequency floor (10th percentile magnitude over time) and
    subtracts a fraction of it. This is the canonical accidental watermark
    attack: it removes exactly the kind of low-energy content a mark hides in.
    """
    n_fft = int(DENOISE_N_FFT)
    hop = int(DENOISE_HOP)
    win = torch.hann_window(n_fft)
    t = audio.shape[-1]
    spec = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    mag = spec.abs()
    phase = torch.angle(spec)
    floor = torch.quantile(mag, 0.10, dim=-1, keepdim=True)
    mag_clean = torch.clamp(mag - float(DENOISE_REDUCTION) * floor, min=0.0)
    spec_clean = torch.polar(mag_clean, phase)
    out = torch.istft(spec_clean, n_fft=n_fft, hop_length=hop, window=win, length=t)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out


@register_attack("reverb")
def attack_reverb(audio: torch.Tensor) -> torch.Tensor:
    """Simulate reverb via convolution with decaying noise."""
    # Simple RIR: decaying white noise
    rir_len = int(SAMPLE_RATE * 0.3) # 0.3s reverb
    t = torch.linspace(0, 1, rir_len, device=audio.device)
    decay = torch.exp(-5 * t)
    rir = torch.randn(rir_len, device=audio.device) * decay
    rir = rir / rir.norm(p=2) # Normalize energy
    
    # Convolve
    # Convolve
    # audio: (channels, T) or (T,)
    # Input to conv1d must be (B, C, T)
    # We treat channels as batch here? No, reverb applies to channel.
    # audio is (C, T). 
    
    # Let's assume audio is (1, T) from our contract.
    C, T = audio.shape
    
    # (B=C, In=1, T)
    x = audio.unsqueeze(1) 
    
    # Kernel (Out=1, In=1, K)
    k = rir.view(1, 1, -1)
    
    # Padding='same' is easiest if available, else classic calc
    # P = (K-1)//2 for same if K is odd. K=4800 is even.
    # Let's use simple padding and crop
    pad = rir_len - 1
    out = F.conv1d(x, k, padding=pad)
    
    # Crop to original length (remove head/tail tails)
    # Align peak? Reverb usually starts at t=0.
    # So we pad at end (convolution does this naturally without padding start)
    # Actually standard reverb: y[n] = x[n]*h[0] + ...
    # So output starts at 0.
    # We want valid part + tail?
    # Let's just crop to T
    out = out[..., :T]

    return out.squeeze(1)


@register_attack("recapture_sim")
def attack_recapture_sim(audio: torch.Tensor) -> torch.Tensor:
    """T3: SIMULATED playback-and-recapture (not a physical loop).

    Chains the dominant artefacts of speaker->air->mic: room reverb, mic/speaker
    EQ colouration, and a small additive noise floor. Labelled 'sim' so the
    report does not over-claim a true recapture measurement.
    """
    x = attack_reverb(audio)
    x = torchaudio.functional.bass_biquad(x, SAMPLE_RATE, gain=float(EQ_BASS_GAIN_DB))
    x = torchaudio.functional.treble_biquad(x, SAMPLE_RATE, gain=float(EQ_TREBLE_GAIN_DB))
    sig_power = x.pow(2).mean()
    noise_power = sig_power / (10 ** (float(RECAPTURE_NOISE_SNR_DB) / 10))
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


# =============================================================================
# Three-tier transformation suite (IR Table 3.4). Single source of truth for
# which registered attacks belong to which tier. 'clean' is the baseline.
# =============================================================================
TIERS: dict[str, list[str]] = {
    "T1": ["mp3_128k", "aac_128k", "resample_8k", "noise_white_20db", "trim"],
    "T2": ["denoise", "loudnorm", "eq", "bg_mix"],
    "T3": ["reverb", "recapture_sim"],
}


def tier_of(attack_name: str) -> Optional[str]:
    """Return the tier label ('T1'/'T2'/'T3') for an attack, or None."""
    for tier, names in TIERS.items():
        if attack_name in names:
            return tier
    return None
