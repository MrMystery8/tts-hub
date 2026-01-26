"""
Watermark Losses Module

Contains:
- CachedSTFTLoss: Multi-resolution spectral loss with cached windows.
- EnergyBudgetLoss: Penalizes watermark energy exceeding a target SNR/MSE.
- UncertaintyLossWrapper: Learns optimal weights for multi-task loss terms (Kendall et al.).
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


class EnergyBudgetLoss(nn.Module):
    """
    Penalizes watermark signal if it exceeds a power budget.
    Targeting specific SNR or simple Absolute/RMS limit.
    """
    
    def __init__(self, target_db: float = -30.0, limit_type: str = "hard"):
        """
        Args:
            target_db: Target maximum relative power in dB (e.g. -30dB).
            limit_type: 'hard' (ReLU penalty above target), 'soft' (L2 regularization).
        """
        super().__init__()
        self.target_db = target_db
        self.limit_type = limit_type
        # dB to linear scale: 10^(db/10) for POWER, 10^(db/20) for AMPLITUDE
        self.target_power_ratio = 10 ** (target_db / 10.0)
        
    def forward(self, original: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
        """
        Compute energy budget penalty.
        Assumes audio is (B, T) or (B, 1, T).
        """
        diff = watermarked - original
        
        # Calculate Power (Mean Squared Amplitude)
        # (B,)
        power_wm = diff.pow(2).mean(dim=-1)
        if power_wm.dim() > 1: power_wm = power_wm.mean(dim=-1)
        
        power_orig = original.pow(2).mean(dim=-1)
        if power_orig.dim() > 1: power_orig = power_orig.mean(dim=-1)
        
        # Target Power
        limit = power_orig * self.target_power_ratio
        
        # Avoid division by zero issues if orig is silent
        limit = torch.max(limit, torch.tensor(1e-9, device=limit.device))
        
        if self.limit_type == "hard":
            # ReLU penalty: only penalized if power_wm > limit
            excess = F.relu(power_wm - limit)
            # Normalize by limit so gradient scale is consistent 
            loss = (excess / limit).mean()
        else:
            # Soft L2: push towards zero, but "budget" implies staying UNDER.
            # Usually simple L2 on delta is enough if weighted correctly.
            loss = power_wm.mean()
            
        return loss


class UncertaintyLossWrapper(nn.Module):
    """
    Multi-task loss wrapper that learns optimal weights.
    Loss = sum( loss_i / (2 * sigma_i^2) + log(sigma_i) )
    
    Ref: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(self, num_losses: int):
        super().__init__()
        # log_vars = log(sigma^2)
        # Initialize to 0.0 (sigma=1)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """
        Combine losses with learned uncertainty weights.
        
        Args:
            losses: List of scalar tensors for each task.
        
        Returns:
            Weighted sum scalar tensor.
        """
        if len(losses) != len(self.log_vars):
            raise ValueError(f"Expected {len(self.log_vars)} losses, got {len(losses)}")
            
        final_loss = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            final_loss += precision * loss + 0.5 * self.log_vars[i]
            
        return final_loss
