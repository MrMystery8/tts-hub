"""
Stage 1B: Payload Training (Curriculum)

Trains the decoder to recover payload bits.
Implementation follows WATERMARK_PROJECT_PLAN.md v16, section 5.3.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_preamble_log_likelihood(msg_probs: torch.Tensor, preamble: torch.Tensor) -> torch.Tensor:
    """
    Log-likelihood for preamble selection (not hard match!).
    
    Args:
        msg_probs: (B, n_win, 32) probabilities
        preamble: (16,) preamble tensor
        
    Returns:
        (B, n_win) log likelihood scores
    """
    B, n_win, _ = msg_probs.shape
    preamble_probs = msg_probs[:, :, :16]
    preamble_exp = preamble.view(1, 1, 16).expand(B, n_win, -1)
    
    eps = 1e-7
    p = torch.clamp(preamble_probs, eps, 1 - eps)
    
    # If preamble bit is 1, add log(p), else log(1-p)
    ll = torch.where(preamble_exp == 1, torch.log(p), torch.log(1 - p))
    
    return ll.sum(dim=2)


def train_stage1b(
    decoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    preamble: torch.Tensor,
    epochs: int = 10,
    warmup: int = 3,
    top_k: int = 3,
    lr: float = 1e-4,
    log_interval: int = 10
):
    """
    Train payload with curriculum:
    - Warmup: use preamble correlation (detector not trusted yet)
    - After: use detect prob
    """
    print(f"Starting Stage 1B: Payload Training for {epochs} epochs (Warmup: {warmup})")
    
    decoder.train()
    preamble = preamble.to(device)
    
    for epoch in range(epochs):
        in_warmup = epoch < warmup
        
        # --- Optimizer & Freeze Logic ---
        if epoch == 0 or epoch == warmup:
            if in_warmup:
                print(">> WARMUP PHASE: Training message head only")
                # Freeze everything EXCEPT message head
                for n, p in decoder.named_parameters():
                    if 'head_message' not in n:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
            else:
                print(">> NORMAL PHASE: Unfreezing all parameters")
                # Unfreeze everything
                for p in decoder.parameters():
                    p.requires_grad = True
            
            # Recreate optimizer for currently trainable params
            trainable = [p for p in decoder.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=lr)
        
        total_loss = 0.0
        batches = 0
        
        for i, batch in enumerate(loader):
            # Only train on positive samples!
            # Filter mask
            has_wm = batch["has_watermark"].bool()
            if not has_wm.any():
                continue
            
            # Select only watermarked samples
            audio = batch["audio"][has_wm].to(device)
            message = batch["message"][has_wm].to(device)  # (B, 32)
            
            outputs = decoder(audio)
            
            # We need to select the best windows to train against the target message
            # logic depends on phase
            if in_warmup:
                # Use PREAMBLE log-likelihood for window selection
                ll = compute_preamble_log_likelihood(
                    outputs["all_message_probs"], preamble
                )
                _, top_idx = torch.topk(ll, min(top_k, ll.shape[1]), dim=1)
            else:
                # Use DETECTOR probability
                _, top_idx = torch.topk(
                   outputs["all_window_probs"], min(top_k, outputs["all_window_probs"].shape[1]), dim=1
                )
            
            # Gather top-k message logits
            msg_logits = outputs["all_message_logits"]  # (B, n_win, 32)
            B, n_win, bits = msg_logits.shape
            
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, bits)
            selected = torch.gather(msg_logits, 1, idx_exp)
            
            # Average logits across top-k windows
            avg_logits = selected.mean(dim=1)
            
            loss = F.binary_cross_entropy_with_logits(avg_logits, message)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            batches += 1
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

    # Ensure everything is unfrozen at end
    for p in decoder.parameters():
        p.requires_grad = True
