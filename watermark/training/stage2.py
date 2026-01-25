"""
Stage 2: Encoder Training

Trains the encoder to embed watermarks that obey the decoder,
while minimizing audio degradation.
Implementation follows WATERMARK_PROJECT_PLAN.md v17, section 5.4.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from typing import Callable, Optional

from watermark.training.losses import CachedSTFTLoss
from watermark.config import SAMPLE_RATE


class DifferentiableAugmenter:
    """Only transforms that preserve gradient flow."""
    def __init__(self, device: torch.device, sample_rate: int = SAMPLE_RATE, reverb_prob: float = 0.25):
        self.device = device
        self.sample_rate = sample_rate
        self.reverb_prob = reverb_prob

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self.reverb_prob > 0 and random.random() < self.reverb_prob:
            return self.apply_reverb(audio)
        transform = random.choice([self.identity, self.add_noise, self.apply_eq, self.volume_change])
        return transform(audio)
    
    def identity(self, x): return x
    
    def add_noise(self, x, snr=25):
        power = x.pow(2).mean()
        noise = torch.randn_like(x) * (power / 10**(snr/10)).sqrt()
        return x + noise
    
    def apply_eq(self, x):
        k = random.choice([3, 5, 7])
        kernel = torch.ones(1, 1, k, device=self.device) / k
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Apply conv and same pad
        out = F.conv1d(x, kernel, padding=k//2)
        if out.shape[-1] != x.shape[-1]:
             out = out[..., :x.shape[-1]]
        return out.squeeze(1)
    
    def volume_change(self, x):
        db = random.uniform(-6, 6)
        return x * 10**(db/20)

    def apply_reverb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable reverb approximation using time-domain convolution.
        Expects x as (B, T). Preserves length.
        """
        if x.dim() != 2:
            raise ValueError(f"apply_reverb expects (B, T), got {x.shape}")

        B, T = x.shape
        rir_seconds = random.uniform(0.05, 0.25)
        rir_len = max(64, int(self.sample_rate * rir_seconds))

        t = torch.linspace(0, 1, rir_len, device=self.device)
        decay_rate = random.uniform(4.0, 10.0)
        decay = torch.exp(-decay_rate * t)

        rir = torch.randn(rir_len, device=self.device) * decay
        rir[0] = rir[0] + 1.0  # direct path
        rir = rir / (rir.abs().sum() + 1e-6)

        y = F.conv1d(x.unsqueeze(1), rir.view(1, 1, -1), padding=rir_len - 1)
        y = y[..., :T].squeeze(1)

        wet = random.uniform(0.2, 0.6)
        return (1 - wet) * x + wet * y


def sample_pair_ids(
    batch_size: int,
    *,
    n_models: int,
    n_versions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Uniform sampling over (model_id, version) pairs.

    Returns:
        model_id: (B,) int64 in [0, n_models-1]
        version: (B,) int64 in [0, n_versions-1]
        pair_id: (B,) int64 in [0, n_models*n_versions-1]
    """
    # Keep sampling on CPU for MPS safety; move to target device after.
    n_pairs = int(n_models) * int(n_versions)
    pair_cpu = torch.randint(0, n_pairs, (int(batch_size),), device="cpu", dtype=torch.long)
    model_id = (pair_cpu % int(n_models)).to(device=device)
    version = (pair_cpu // int(n_models)).to(device=device)
    return model_id, version, pair_cpu.to(device=device)


def _encode_message_batch(
    *,
    preamble_16: torch.Tensor,
    model_id: torch.Tensor,
    version: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized MessageCodec.encode() for a batch.

    Args:
        preamble_16: (16,) or (1, 16) float tensor in {0,1}
        model_id: (B,) int64 tensor in [0, 7]
        version: (B,) int64 tensor in [0, 15]

    Returns:
        (B, 32) float tensor in {0,1}
    """
    if model_id.dim() != 1 or version.dim() != 1:
        raise ValueError(f"model_id/version must be 1D, got {model_id.shape} / {version.shape}")
    if model_id.shape[0] != version.shape[0]:
        raise ValueError(f"model_id/version batch mismatch: {model_id.shape} vs {version.shape}")

    B = model_id.shape[0]
    device = model_id.device

    if preamble_16.dim() == 2:
        if preamble_16.shape != (1, 16):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = preamble_16.to(device=device, dtype=torch.float32).expand(B, -1)
    elif preamble_16.dim() == 1:
        if preamble_16.shape != (16,):
            raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")
        pre = preamble_16.to(device=device, dtype=torch.float32).view(1, 16).expand(B, -1)
    else:
        raise ValueError(f"preamble_16 must be (16,) or (1,16), got {preamble_16.shape}")

    msg = torch.zeros((B, 32), device=device, dtype=torch.float32)
    msg[:, 0:16] = pre

    # Payload copy 1 (bits 16-22)
    for i in range(3):
        msg[:, 16 + i] = ((model_id >> i) & 1).to(torch.float32)
    for i in range(4):
        msg[:, 19 + i] = ((version >> i) & 1).to(torch.float32)

    # Payload copy 2 (bits 23-29)
    msg[:, 23:26] = msg[:, 16:19]
    msg[:, 26:30] = msg[:, 19:23]
    return msg


def train_stage2(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    top_k: int = 3,
    lr: float = 1e-3,
    msg_weight: float = 1.0,
    model_ce_weight: float = 1.0,
    version_ce_weight: float = 1.0,
    pair_ce_weight: float = 2.0,
    aux_weight: float = 0.5,
    qual_weight: float = 1.0,
    reverb_prob: float = 0.25,
    preamble_weight: float = 0.2,
    model_id_weight: float = 2.0,
    version_weight: float = 0.5,
    payload_on_clean: bool = True,
    payload_pos_only: bool = True,
    message_mode: str = "random_if_mixed",
    n_models: int = 8,
    n_versions: int = 16,
    log_interval: int = 10,
    step_interval: int = 0,
    on_step: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
):
    """
    Train encoder with differentiable augments only.
    
    FIX from project plan: Use TOP-K windows for loss (matches inference objective!)
    """
    print(f"Starting Stage 2: Encoder Training for {epochs} epochs")
    
    aug = DifferentiableAugmenter(device, reverb_prob=reverb_prob)
    
    # Quality loss
    stft_loss = CachedSTFTLoss().to(device)
    
    # Freeze decoder
    for p in decoder.parameters():
        p.requires_grad = False
    
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    # Bitwise weights for message loss (same rationale as Stage 1B).
    msg_weights = torch.zeros(32, device=device)
    msg_weights[0:16] = preamble_weight
    msg_weights[16:19] = model_id_weight
    msg_weights[19:23] = version_weight
    msg_weights[23:26] = model_id_weight
    msg_weights[26:30] = version_weight

    weight_sum = msg_weights.sum().clamp(min=1e-8)
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_loss_det = 0.0
        total_loss_aux = 0.0
        total_loss_msg = 0.0
        total_loss_model = 0.0
        total_loss_version = 0.0
        total_loss_pair = 0.0
        total_loss_qual = 0.0
        batches = 0
        try:
            n_batches = int(len(loader))
        except Exception:
            n_batches = None
        
        for i, batch in enumerate(loader):
            # Train only on samples meant to be watermarked according to manifest?
            # Or force all samples to be watermarked for encoder training?
            # Usually we take any audio and force watermark it for training.
            
            audio = batch["audio"].to(device)  # (B, 1, T)
            B = int(audio.shape[0])

            # Stage 2 always watermarks its carriers on-the-fly. When training on a mixed
            # manifest (watermarked positives + clean negatives for Stage 1), you can:
            # - `payload_pos_only=True`  : apply payload/ID losses only on labeled positives.
            # - `payload_pos_only=False` : apply payload/ID losses on all carriers (recommended
            #   when you want the encoder to strongly learn identity, since all carriers are
            #   watermarked in Stage 2 anyway).
            if payload_pos_only and ("has_watermark" in batch):
                pos_mask = (batch["has_watermark"].to(device) > 0.5)
            else:
                pos_mask = torch.ones((B,), device=device, dtype=torch.bool)

            if message_mode not in {"batch", "random", "random_if_mixed"}:
                raise ValueError(
                    f"message_mode must be one of {{'batch','random','random_if_mixed'}}, got {message_mode!r}"
                )

            # Build per-sample targets:
            # - If IDs are missing (e.g., model_id/version == -1), sample a balanced random pair.
            # - Do not allow constant default IDs to dominate supervision.
            batch_model = batch.get("model_id", torch.full((B,), -1, dtype=torch.long)).to(device)
            batch_ver = batch.get("version", torch.full((B,), -1, dtype=torch.long)).to(device)
            ids_valid = (batch_model >= 0) & (batch_ver >= 0)

            rand_model, rand_ver, _rand_pair = sample_pair_ids(B, n_models=n_models, n_versions=n_versions, device=device)

            if message_mode == "random":
                target_model = rand_model
                target_ver = rand_ver
            elif message_mode == "batch":
                # Use provided IDs for labeled positives; fall back to random for unlabeled.
                # (Allows mixed manifests without silently collapsing to a constant default.)
                target_model = torch.where(ids_valid, batch_model, rand_model)
                target_ver = torch.where(ids_valid, batch_ver, rand_ver)
            else:  # random_if_mixed
                # Overwrite only unlabeled samples (and always avoid invalid IDs).
                target_model = torch.where(ids_valid, batch_model, rand_model)
                target_ver = torch.where(ids_valid, batch_ver, rand_ver)

            preamble_16_cpu = batch["message"][:1, :16].to(device="cpu", dtype=torch.float32)
            message = _encode_message_batch(
                preamble_16=preamble_16_cpu,
                model_id=target_model.to(device="cpu", dtype=torch.long),
                version=target_ver.to(device="cpu", dtype=torch.long),
            ).to(device=device, dtype=torch.float32)
            
            # Embed
            wm = encoder(audio, message)
            
            # Augment (gradient-safe)
            # DifferentiableAugmenter needs to handle (B, 1, T) or we adapt
            # Current augmenter expects (B, T).
            # We can update augmenter or squeeze/unsqueeze. 
            # Let's fix augmenter to be shape agnostic or handle (B, 1, T).
            # For now, squeeze to (B, T) then unsqueeze after?
            # Augmenter returns (B, T) implies loss of channel info.
            # Best: Update augmenter to handle (B, 1, T).
            # For immediate fix: squeeze, augment, unsqueeze.
            augmented = aug(wm.squeeze(1)).unsqueeze(1)
            
            # Decode
            # - Detection should be trained against (potentially) attacked audio to build robustness.
            # - Payload/attribution is much harder; training it directly under heavy augments early
            #   often collapses into “presence-only”. By default we supervise payload on the clean
            #   watermarked audio while still supervising detection on augmented audio.
            outputs_det = decoder(augmented)
            outputs_payload = decoder(wm) if payload_on_clean else outputs_det
            
            # 1. Quality Loss
            # STFT loss expects (B, T) usually? 
            # CachedSTFTLoss usually takes (B, T).
            loss_qual = stft_loss(audio.squeeze(1), wm.squeeze(1))
            
            # 2. Detection & Message Loss (Top-K AND All-Window Aux)
            detect = outputs_det["all_window_probs"]  # (B, n_win)
            detect_logits = outputs_det["all_window_logits"]
            msg_logits = outputs_payload["all_message_logits"]
            model_logits = outputs_payload.get("all_model_logits")  # (B, n_win, n_models+1)
            version_logits = outputs_payload.get("all_version_logits")  # (B, n_win, n_versions+1)
            pair_logits = outputs_payload.get("all_pair_logits")  # (B, n_win, n_pairs+1)
            
            B, n_win = detect.shape
            k = min(top_k, n_win)
            
            # Top-k by detection probability
            _, top_idx = torch.topk(detect, k, dim=1)
            
            # Gather top-k detect logits
            top_det_logits = torch.gather(detect_logits, 1, top_idx)
            
            # Detection loss (Target = 1) - PRIMARY
            loss_det = F.binary_cross_entropy_with_logits(
                top_det_logits.mean(dim=1),
                torch.ones(B, device=device)
            )
            
            # AUXILIARY: All-window detection loss
            # Ensures every window contributes gradients, preventing "sparse gradient" plateau
            loss_aux = F.binary_cross_entropy_with_logits(
                detect_logits,
                torch.ones_like(detect_logits)
            )
            
            # Gather top-k message logits
            top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, msg_logits.shape[-1])
            top_msg_logits = torch.gather(msg_logits, 1, top_idx_exp)
            
            # Message loss
            avg_msg_logits = top_msg_logits.mean(dim=1)
            per_bit = F.binary_cross_entropy_with_logits(avg_msg_logits, message, reduction="none")
            per_sample = (per_bit * msg_weights).sum(dim=1) / weight_sum
            if pos_mask.any():
                loss_msg = per_sample[pos_mask].mean()
            else:
                loss_msg = torch.tensor(0.0, device=device)

            # Attribution losses (classification heads).
            loss_model = torch.tensor(0.0, device=device)
            if model_logits is not None and model_ce_weight > 0 and pos_mask.any():
                n_classes = model_logits.shape[-1]
                top_model_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                avg_model_logits = torch.gather(model_logits, 1, top_model_idx).mean(dim=1)
                loss_model = F.cross_entropy(avg_model_logits[pos_mask], target_model[pos_mask])

            loss_version = torch.tensor(0.0, device=device)
            if version_logits is not None and version_ce_weight > 0 and pos_mask.any():
                n_classes = version_logits.shape[-1]
                top_ver_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                avg_ver_logits = torch.gather(version_logits, 1, top_ver_idx).mean(dim=1)
                loss_version = F.cross_entropy(avg_ver_logits[pos_mask], target_ver[pos_mask])

            loss_pair = torch.tensor(0.0, device=device)
            if pair_logits is not None and pair_ce_weight > 0 and pos_mask.any():
                n_classes = pair_logits.shape[-1]
                top_pair_idx = top_idx.unsqueeze(-1).expand(-1, -1, n_classes)
                avg_pair_logits = torch.gather(pair_logits, 1, top_pair_idx).mean(dim=1)
                pair_id = (target_ver * int(n_models) + target_model).to(dtype=torch.long)
                loss_pair = F.cross_entropy(avg_pair_logits[pos_mask], pair_id[pos_mask])
            
            # Combined Loss
            # Weighted: 1.0 Det + aux_weight * Aux + msg_weight * Msg + CE(model/version) + qual_weight * Qual
            loss = (
                loss_det
                + aux_weight * loss_aux
                + msg_weight * loss_msg
                + model_ce_weight * loss_model
                + version_ce_weight * loss_version
                + pair_ce_weight * loss_pair
                + qual_weight * loss_qual
            )
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            total_loss_det += loss_det.item()
            total_loss_aux += loss_aux.item()
            total_loss_msg += loss_msg.item()
            total_loss_model += loss_model.item()
            total_loss_version += loss_version.item()
            total_loss_pair += loss_pair.item()
            total_loss_qual += loss_qual.item()
            batches += 1

            if on_step is not None and int(step_interval) > 0 and (i % int(step_interval) == 0):
                on_step(
                    {
                        "type": "step",
                        "stage": "s2",
                        "epoch": epoch + 1,
                        "batch": int(i),
                        "n_batches": n_batches,
                        "loss": float(loss.item()),
                        "loss_det": float(loss_det.item()),
                        "loss_aux": float(loss_aux.item()),
                        "loss_msg": float(loss_msg.item()),
                        "loss_model_ce": float(loss_model.item()),
                        "loss_version_ce": float(loss_version.item()),
                        "loss_pair_ce": float(loss_pair.item()),
                        "loss_qual": float(loss_qual.item()),
                        "msg_weight": float(msg_weight),
                        "model_ce_weight": float(model_ce_weight),
                        "version_ce_weight": float(version_ce_weight),
                        "pair_ce_weight": float(pair_ce_weight),
                        "payload_on_clean": bool(payload_on_clean),
                        "payload_pos_only": bool(payload_pos_only),
                    }
                )
            
            if i % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {i} | Loss: {loss.item():.4f} (Qual: {loss_qual.item():.4f}, Det: {loss_det.item():.4f})")
        
        avg_loss = total_loss / batches if batches > 0 else 0
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        if on_epoch_end is not None:
            denom = max(1, batches)
            on_epoch_end(
                {
                    "type": "epoch",
                    "stage": "s2",
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "loss_det": total_loss_det / denom,
                    "loss_aux": total_loss_aux / denom,
                    "loss_msg": total_loss_msg / denom,
                    "loss_model_ce": total_loss_model / denom,
                    "loss_version_ce": total_loss_version / denom,
                    "loss_pair_ce": total_loss_pair / denom,
                    "loss_qual": total_loss_qual / denom,
                    "msg_weight": float(msg_weight),
                    "model_ce_weight": float(model_ce_weight),
                    "version_ce_weight": float(version_ce_weight),
                    "pair_ce_weight": float(pair_ce_weight),
                    "aux_weight": float(aux_weight),
                    "qual_weight": float(qual_weight),
                    "reverb_prob": float(reverb_prob),
                    "payload_on_clean": bool(payload_on_clean),
                    "payload_pos_only": bool(payload_pos_only),
                    "lr": opt.param_groups[0].get("lr"),
                }
            )
