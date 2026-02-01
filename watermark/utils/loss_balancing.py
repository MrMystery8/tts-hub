
import math
import torch
import torch.nn as nn
from typing import Tuple, Dict

class LossBalancer(nn.Module):
    """Abstract base class for loss balancing strategies."""
    def combine(self, 
                loss_det: torch.Tensor, 
                loss_id: torch.Tensor, 
                step: int, 
                epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

class UncertaintyBalancer(LossBalancer):
    """
    Adaptive loss balancing using Homoscedastic Uncertainty (Kendall et al.).
    
    Weights are parameterized as log variances (s).
    Loss = exp(-s) * L + s
    
    We initialize from desired effective weights (w_init) by solving:
      exp(-s) = w_init  =>  s = -log(w_init)
    """
    def __init__(self, init_weight_detect: float = 8.0, init_weight_id: float = 5.0):
        super().__init__()
        
        # Guard against zero/negative weights
        w_det = max(init_weight_detect, 1e-6)
        w_id = max(init_weight_id, 1e-6)
        
        # Initialize learnable parameters (s = log_variance)
        # s represents the noise scale. High s = High noise = Low weight.
        # w = exp(-s)
        self.s_det = nn.Parameter(torch.tensor(-math.log(w_det), dtype=torch.float32))
        self.s_id = nn.Parameter(torch.tensor(-math.log(w_id), dtype=torch.float32))

    def combine(self, 
                loss_det: torch.Tensor, 
                loss_id: torch.Tensor, 
                step: int, 
                epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Clamp s to prevent numerical instability (weights exploding or vanishing to inf)
        # Range [-6, 6] implies weights in [exp(-6), exp(6)] ~= [0.0025, 403.4]
        s_det_clamped = self.s_det.clamp(-6.0, 6.0)
        s_id_clamped = self.s_id.clamp(-6.0, 6.0)
        
        # Calculate effective weights for logging
        w_det = torch.exp(-s_det_clamped)
        w_id = torch.exp(-s_id_clamped)
        
        # Combine strictly: L = w*L_raw + log_var
        # (The +s term penalizes the model for just increasing variance to reduce loss)
        weighted_det = w_det * loss_det + s_det_clamped
        weighted_id = w_id * loss_id + s_id_clamped
        
        total_loss = weighted_det + weighted_id
        
        info = {
            "w_det": w_det.item(),
            "w_id": w_id.item(),
            "s_det": s_det_clamped.item(),
            "s_id": s_id_clamped.item(),
        }
        
        return total_loss, info
