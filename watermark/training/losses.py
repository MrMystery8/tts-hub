"""
Watermark Losses Module

Contains:
- CachedSTFTLoss: Multi-resolution spectral loss with cached windows for performance.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.5.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss with CACHED windows.
    Avoids continually re-creating windows/moving to device in forward pass.
    """
    
    def __init__(self, fft_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        
        # Cache windows as buffers
        for n_fft in fft_sizes:
            self.register_buffer(f'window_{n_fft}', torch.hann_window(n_fft))
    
    def forward(self, original: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution spectral loss.
        
        Args:
            original: (B, T) original audio
            watermarked: (B, T) watermarked audio
        
        Returns:
            Scalar loss (spectral convergence + log-magnitude L1)
        """
        total = 0.0
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = getattr(self, f'window_{n_fft}')  # Use cached window!
            
            # STFT on original
            orig_spec = torch.stft(
                original, n_fft, hop, window=window, return_complex=True
            )
            
            # STFT on watermarked
            wm_spec = torch.stft(
                watermarked, n_fft, hop, window=window, return_complex=True
            )
            
            orig_mag = orig_spec.abs()
            wm_mag = wm_spec.abs()
            
            # Spectral Convergence
            sc_loss = torch.norm(orig_mag - wm_mag, p='fro') / (torch.norm(orig_mag, p='fro') + 1e-8)
            
            # Log-magnitude L1
            log_loss = F.l1_loss(torch.log(wm_mag + 1e-8), torch.log(orig_mag + 1e-8))
            
            total = total + sc_loss + log_loss
        
        return total / len(self.fft_sizes)
