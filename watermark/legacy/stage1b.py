"""
Stage 1B: Payload Training (Curriculum)

Trains the decoder to recover payload bits.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.3.

LEGACY NOTE:
The current watermarking pipeline uses multiclass attribution and does not train a bit payload head.
This module is retained for older experiments only.

CORRECTION (On-the-fly Embedding):
Dataset yields CLEAN audio. We apply watermark on-the-fly to samples.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable, Optional

from watermark.config import N_MODELS, N_VERSIONS


def _sample_pair_ids(
    batch_size: int,
    *,
    n_models: int,
    n_versions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_pairs = int(n_models) * int(n_versions)
    pair_cpu = torch.randint(0, n_pairs, (int(batch_size),), device="cpu", dtype=torch.long)
    model_id = (pair_cpu % int(n_models)).to(device=device)
    version = (pair_cpu // int(n_models)).to(device=device)
    return model_id, version


def _encode_message_batch(
    *,
    preamble_16: torch.Tensor,
    model_id: torch.Tensor,
    version: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized MessageCodec.encode() for a batch (32-bit format).
    """
    if preamble_16.dim() == 2:
        pre = preamble_16.to(dtype=torch.float32)
        if pre.shape != (1, 16):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = pre.expand(int(model_id.shape[0]), -1)
    elif preamble_16.dim() == 1:
        if preamble_16.shape != (16,):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = preamble_16.to(dtype=torch.float32).view(1, 16).expand(int(model_id.shape[0]), -1)
    else:
        raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")

    msg = torch.zeros((int(model_id.shape[0]), 32), device=model_id.device, dtype=torch.float32)
    # Balanced Diet: 16-bit preamble (Robust Sync)
    msg[:, 0:16] = pre.to(device=model_id.device)

    # Shifted Identity (start at bit 16)
    for i in range(3):
        msg[:, 16 + i] = ((model_id >> i) & 1).to(torch.float32)
    for i in range(4):
        msg[:, 19 + i] = ((version >> i) & 1).to(torch.float32)

    # No copies
    return msg


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
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    preamble: torch.Tensor,
    *,
    stage: str = "s1b",
    heads_only: bool = False,
    freeze_backbone: bool = False,
    epochs: int = 10,
    warmup: int = 3,
    top_k: int = 3,
    lr: float = 1e-4,
    neg_weight: float = 0.4,
    neg_preamble_target: float = 0.5,
    unknown_ce_weight: float = 0.2,
    model_ce_weight: float = 1.0,
    version_ce_weight: float = 1.0,
    pair_ce_weight: float = 2.0,
    n_models: int = N_MODELS,
    n_versions: int = N_VERSIONS,
    preamble_weight: float = 0.2,
    model_id_weight: float = 2.0,
    version_weight: float = 0.5,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
):
    """
    Train payload with curriculum:
    - Warmup: use preamble correlation (detector not trusted yet)
    - After: use detect prob
    
    Applies encoder only to watermaked samples (which is all samples in Stage 1B context 
    if we filter correctly, but here we filter has_wm from mixed batch).
    """
    print(f"Starting Stage 1B: Payload Training for {epochs} epochs (Warmup: {warmup})")
    
    decoder.train()
    encoder.eval() # Encoder frozen
    preamble = preamble.to(device)

    # Bitwise weights (avoid constant bits dominating payload learning).
    # Layout (32 bits):
    # - 0:16 preamble
    # - 16:19 model_id
    # - 19:23 version
    # - 23:26 model_id copy
    # - 26:30 version copy
    # - 30:32 reserved (ignored)
    msg_weights = torch.zeros(32, device=device)
    # Balanced Diet Layout:
    # 0-16: Preamble
    # 16-19: Model ID
    # 19-23: Version ID
    msg_weights[0:16] = preamble_weight
    msg_weights[16:19] = model_id_weight
    msg_weights[19:23] = version_weight
    # Rest are 0.0 (ignored)

    weight_sum = msg_weights.sum().clamp(min=1e-8)
    
    for epoch in range(epochs):
        in_warmup = epoch < warmup
        
        # --- Optimizer & Freeze Logic ---
        if epoch == 0 or epoch == warmup:
            # logic:
            # 1. if heads_only=True, we freeze backbone/det regardless of epoch
            # 2. if freeze_backbone=True, we freeze backbone/det regardless of epoch (often used with heads_only)
            # 3. if in_warmup=True, we default to freezing unless overruled? 
            # Actually, standard flow: warmup -> full.
            # But the new "Snap" strategy might request permanent freezing for the entire call.
            
            should_freeze = (in_warmup or heads_only or freeze_backbone)
            
            if should_freeze:
                if in_warmup:
                    print(">> WARMUP PHASE: Training message head only")
                elif heads_only or freeze_backbone:
                    print(">> LOCKED BACKBONE PHASE: Training attribution heads only")
                
                # Freeze everything EXCEPT message/id heads
                for n, p in decoder.named_parameters():
                    is_attr_head = any(h in n for h in ("head_message", "head_model", "head_version", "head_pair"))
                    if is_attr_head:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                
                # CRITICAL: Force frozen modules to eval mode to stop BN updates!
                if hasattr(decoder, "backbone"):
                    decoder.backbone.eval()
                if hasattr(decoder, "head_detect"):
                    decoder.head_detect.eval()
                # Ensure attribution heads are in train mode
                if hasattr(decoder, "head_message"):
                    decoder.head_message.train()
                if hasattr(decoder, "head_model"):
                    decoder.head_model.train()
                if hasattr(decoder, "head_version"):
                    decoder.head_version.train()
                if hasattr(decoder, "head_pair"):
                    decoder.head_pair.train()
            else:
                print(">> NORMAL PHASE: Unfreezing Backbone (Keeping Detect Head Frozen)")
                decoder.train() # Set all to train mode (for Dropout/BN in backbone)
                
                # Unfreeze everything...
                for p in decoder.parameters():
                    p.requires_grad = True
                
                # ...BUT re-freeze the detection head to preserve Stage 1 knowledge
                if hasattr(decoder, "head_detect"):
                    decoder.head_detect.eval() # Keep it in eval mode!
                    for p in decoder.head_detect.parameters():
                        p.requires_grad = False
            
            # Recreate optimizer for currently trainable params
            trainable = [p for p in decoder.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=lr)
        
        total_loss = 0.0
        total_loss_msg_bits = 0.0
        total_loss_model_ce = 0.0
        total_loss_version_ce = 0.0
        total_loss_pair_ce = 0.0
        total_loss_neg_preamble = 0.0
        total_loss_unknown_model_ce = 0.0
        total_loss_unknown_version_ce = 0.0
        batches = 0
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None
        
        for i, batch in enumerate(loader):
            has_wm = batch["has_watermark"].bool()
            pos_mask = has_wm
            neg_mask = ~has_wm
            loss = None
            loss_msg_bits = None
            loss_model_ce = None
            loss_version_ce = None
            loss_pair_ce = None
            loss_neg_preamble = None
            loss_unk_model_ce = None
            loss_unk_version_ce = None

            # --- Positive samples (watermarked) ---
            if pos_mask.any():
                audio = batch["audio"][pos_mask].to(device)
                batch_model = batch["model_id"][pos_mask].to(device)
                batch_ver = batch["version"][pos_mask].to(device)
                ids_valid = (batch_model >= 0) & (batch_ver >= 0)
                if (~ids_valid).any():
                    rand_model, rand_ver = _sample_pair_ids(
                        int(batch_model.shape[0]),
                        n_models=n_models,
                        n_versions=n_versions,
                        device=device,
                    )
                    batch_model = torch.where(ids_valid, batch_model, rand_model)
                    batch_ver = torch.where(ids_valid, batch_ver, rand_ver)

                # Keep message encoding on CPU for MPS safety, then move to device.
                message = _encode_message_batch(
                    preamble_16=preamble[:16].to(device="cpu", dtype=torch.float32),
                    model_id=batch_model.to(device="cpu", dtype=torch.long),
                    version=batch_ver.to(device="cpu", dtype=torch.long),
                ).to(device=device, dtype=torch.float32)
            
                # === On-the-fly Embedding ===
                with torch.no_grad():
                    watermarked_audio = encoder(audio, message).detach()

                outputs = decoder(watermarked_audio)

                if in_warmup:
                    ll = compute_preamble_log_likelihood(
                        outputs["all_message_probs"], preamble
                    )
                    _, top_idx = torch.topk(ll, min(top_k, ll.shape[1]), dim=1)
                else:
                    _, top_idx = torch.topk(
                        outputs["all_window_probs"],
                        min(top_k, outputs["all_window_probs"].shape[1]),
                        dim=1
                    )

                msg_logits = outputs["all_message_logits"]  # (B, n_win, 32)
                B, n_win, bits = msg_logits.shape

                idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, bits)
                selected = torch.gather(msg_logits, 1, idx_exp)
                avg_logits = selected.mean(dim=1)

                per_bit = F.binary_cross_entropy_with_logits(avg_logits, message, reduction="none")
                loss_msg_bits = ((per_bit * msg_weights).sum(dim=1) / weight_sum).mean()
                loss = loss_msg_bits

                # Classification heads (preferable for attribution vs bitwise decode)
                model_logits = outputs.get("all_model_logits")
                if model_logits is not None:
                    n_classes = model_logits.shape[-1]
                    idx_model = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                    top_model_logits = torch.gather(model_logits, 1, idx_model).mean(dim=1)
                    loss_model_ce = F.cross_entropy(top_model_logits, batch_model)
                    loss = loss + model_ce_weight * loss_model_ce

                version_logits = outputs.get("all_version_logits")
                if version_logits is not None:
                    n_classes = version_logits.shape[-1]
                    idx_ver = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                    top_ver_logits = torch.gather(version_logits, 1, idx_ver).mean(dim=1)
                    loss_version_ce = F.cross_entropy(top_ver_logits, batch_ver)
                    loss = loss + version_ce_weight * loss_version_ce

                pair_logits = outputs.get("all_pair_logits")
                if pair_logits is not None and pair_ce_weight > 0:
                    n_classes = pair_logits.shape[-1]
                    idx_pair = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                    top_pair_logits = torch.gather(pair_logits, 1, idx_pair).mean(dim=1)
                    pair_id = (batch_ver * int(n_models) + batch_model).to(dtype=torch.long)
                    loss_pair_ce = F.cross_entropy(top_pair_logits, pair_id)
                    loss = loss + pair_ce_weight * loss_pair_ce

            # --- Negative samples (clean) ---
            if neg_mask.any() and neg_weight > 0:
                neg_audio = batch["audio"][neg_mask].to(device)
                outputs_neg = decoder(neg_audio)

                # Penalize preamble confidence on negatives (reduce false preamble hits)
                msg_logits_neg = outputs_neg["all_message_logits"]  # (B, n_win, 32)
                Bn, n_win, bits = msg_logits_neg.shape
                ll_neg = compute_preamble_log_likelihood(
                    outputs_neg["all_message_probs"], preamble
                )
                _, top_idx_neg = torch.topk(
                    ll_neg,
                    min(top_k, ll_neg.shape[1]),
                    dim=1
                )
                idx_exp_neg = top_idx_neg.unsqueeze(-1).expand(-1, -1, bits)
                selected_neg = torch.gather(msg_logits_neg, 1, idx_exp_neg)
                avg_logits_neg = selected_neg.mean(dim=1)

                neg_preamble_logits = avg_logits_neg[:, :16]
                neg_target = torch.full_like(neg_preamble_logits, float(neg_preamble_target))
                loss_neg_preamble = F.binary_cross_entropy_with_logits(neg_preamble_logits, neg_target)

                if loss is None:
                    loss = neg_weight * loss_neg_preamble
                else:
                    loss = loss + neg_weight * loss_neg_preamble

                # Encourage "unknown" attribution on clean audio (optional).
                if unknown_ce_weight > 0:
                    model_logits_neg = outputs_neg.get("all_model_logits")
                    if model_logits_neg is not None:
                        n_classes = model_logits_neg.shape[-1]
                        idx_model_neg = top_idx_neg.unsqueeze(-1).expand(-1, -1, n_classes)
                        top_model_logits_neg = torch.gather(model_logits_neg, 1, idx_model_neg).mean(dim=1)
                        unk = torch.full((top_model_logits_neg.shape[0],), n_classes - 1, device=device, dtype=torch.long)
                        loss_unk_model_ce = F.cross_entropy(top_model_logits_neg, unk)
                        loss = loss + unknown_ce_weight * loss_unk_model_ce

                    version_logits_neg = outputs_neg.get("all_version_logits")
                    if version_logits_neg is not None:
                        n_classes = version_logits_neg.shape[-1]
                        idx_ver_neg = top_idx_neg.unsqueeze(-1).expand(-1, -1, n_classes)
                        top_ver_logits_neg = torch.gather(version_logits_neg, 1, idx_ver_neg).mean(dim=1)
                        unk = torch.full((top_ver_logits_neg.shape[0],), n_classes - 1, device=device, dtype=torch.long)
                        loss_unk_version_ce = F.cross_entropy(top_ver_logits_neg, unk)
                        loss = loss + unknown_ce_weight * loss_unk_version_ce

            if loss is None:
                continue
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            if loss_msg_bits is not None:
                total_loss_msg_bits += loss_msg_bits.item()
            if loss_model_ce is not None:
                total_loss_model_ce += loss_model_ce.item()
            if loss_version_ce is not None:
                total_loss_version_ce += loss_version_ce.item()
            if loss_pair_ce is not None:
                total_loss_pair_ce += loss_pair_ce.item()
            if loss_neg_preamble is not None:
                total_loss_neg_preamble += loss_neg_preamble.item()
            if loss_unk_model_ce is not None:
                total_loss_unknown_model_ce += loss_unk_model_ce.item()
            if loss_unk_version_ce is not None:
                total_loss_unknown_version_ce += loss_unk_version_ce.item()
            batches += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                on_step(
                    {
                        "type": "step",
                        "stage": str(stage),
                        "epoch": epoch + 1,
                        "batch": int(i),
                        "n_batches": n_batches,
                        "in_warmup": bool(in_warmup),
                        "loss": float(loss.item()) if loss is not None else None,
                        "loss_msg_bits": float(loss_msg_bits.item()) if loss_msg_bits is not None else None,
                        "loss_model_ce": float(loss_model_ce.item()) if loss_model_ce is not None else None,
                        "loss_version_ce": float(loss_version_ce.item()) if loss_version_ce is not None else None,
                        "loss_pair_ce": float(loss_pair_ce.item()) if loss_pair_ce is not None else None,
                        "loss_neg_preamble": float(loss_neg_preamble.item()) if loss_neg_preamble is not None else None,
                        "loss_unknown_model_ce": float(loss_unk_model_ce.item()) if loss_unk_model_ce is not None else None,
                        "loss_unknown_version_ce": float(loss_unk_version_ce.item()) if loss_unk_version_ce is not None else None,
                        "n_pos": int(pos_mask.sum().item()),
                        "n_neg": int(neg_mask.sum().item()),
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        if on_epoch_end is not None:
            denom = max(1, batches)
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": str(stage),
                    "epoch": epoch + 1,
                    "in_warmup": bool(in_warmup),
                    "loss": avg_loss,
                    "loss_msg_bits": total_loss_msg_bits / denom,
                    "loss_model_ce": total_loss_model_ce / denom,
                    "loss_version_ce": total_loss_version_ce / denom,
                    "loss_pair_ce": total_loss_pair_ce / denom,
                    "loss_neg_preamble": total_loss_neg_preamble / denom,
                    "loss_unknown_model_ce": total_loss_unknown_model_ce / denom,
                    "loss_unknown_version_ce": total_loss_unknown_version_ce / denom,
                    "neg_weight": float(neg_weight),
                    "neg_preamble_target": float(neg_preamble_target),
                    "unknown_ce_weight": float(unknown_ce_weight),
                    "model_ce_weight": float(model_ce_weight),
                    "version_ce_weight": float(version_ce_weight),
                    "pair_ce_weight": float(pair_ce_weight),
                    "lr": opt.param_groups[0].get("lr"),
                }
            )

    # Ensure everything is unfrozen at end
    for p in decoder.parameters():
        p.requires_grad = True
