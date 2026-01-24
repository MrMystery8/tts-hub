
import torch
import torchaudio
import soundfile as sf
import numpy as np
import io

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Robust audio loader that enforces a canonical format:
    - Shape: (1, T)
    - Dtype: float32
    - Sample Rate: target_sr
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate (default 16000)
        
    Returns:
        Tensor of shape (1, T)
    """
    try:
        # Prefer SoundFile for stability (no FFmpeg deps usually)
        audio_np, sr = sf.read(path)
        
        # Ensure float32
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
            
        # Convert to Tensor
        audio = torch.from_numpy(audio_np)
        
        # Handle shapes
        # SoundFile returns (T,) for mono, (T, C) for stereo
        if audio.dim() == 1:
            audio = audio.unsqueeze(0) # (1, T)
        elif audio.dim() == 2:
            audio = audio.t() # (C, T)
            
        # Convert to Mono (Mean)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        # Resample if needed
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            
        return audio
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {e}")

def save_audio(path: str, audio: torch.Tensor, sr: int = 16000):
    """
    Save audio tensor to file.
    Expects (1, T) or (T,) tensor.
    """
    audio = audio.detach().cpu()
    if audio.dim() == 2:
        audio = audio.squeeze(0)
    
    sf.write(path, audio.numpy(), sr)
