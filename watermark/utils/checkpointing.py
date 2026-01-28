"""
Checkpointing utilities for watermark training.

This module provides CheckpointManager to handle saving/loading of model checkpoints
with support for best and last checkpoints based on probe metrics.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints during training.
    
    Supports:
    - Last checkpoint (overwrite-safe)
    - Best checkpoint (based on metric)
    - Resume functionality
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        save_last: bool = True,
        save_best: bool = True,
        best_metric: str = "tpr_at_fpr_1pct",
        best_mode: str = "max",
        save_every: int = 1,
        ckpt_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            run_dir: Directory to save checkpoints in
            save_last: Whether to save last checkpoint
            save_best: Whether to save best checkpoint
            best_metric: Metric name to use for best checkpoint selection
            best_mode: "max" to maximize metric, "min" to minimize
            save_every: Save last checkpoint every N epochs (0 to save every epoch)
            ckpt_dir: Override default checkpoints directory (default: <run_dir>/checkpoints)
        """
        self.run_dir = Path(run_dir)

        self.checkpoints_dir = Path(ckpt_dir).expanduser().resolve() if ckpt_dir else (self.run_dir / "checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.should_save_last = save_last
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.save_every = save_every
        
        # Track best metric value
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_stage: Optional[str] = None
        
        # Determine comparison function based on mode
        self._compare_func = max if best_mode == "max" else min

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        epoch: int,
        stage: str,
        global_step: Optional[int] = None,
        optimizer_encoder: Optional[torch.optim.Optimizer] = None,
        optimizer_decoder: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a checkpoint to the specified filepath.
        
        Args:
            filepath: Path to save checkpoint to
            encoder: Encoder model
            decoder: Decoder model
            epoch: Current epoch
            stage: Current training stage
            global_step: Global step (optional)
            optimizer_encoder: Encoder optimizer (optional)
            optimizer_decoder: Decoder optimizer (optional)
            metrics: Metrics dictionary (optional)
            args: Training arguments (optional)
        """
        payload = {
            "schema": 1,
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "encoder": self._to_cpu_state_dict(encoder.state_dict()),
            "decoder": self._to_cpu_state_dict(decoder.state_dict()),
            "opt_encoder": optimizer_encoder.state_dict() if optimizer_encoder else None,
            "opt_decoder": optimizer_decoder.state_dict() if optimizer_decoder else None,
            "metrics": metrics,
            "args": args,
        }

        # Save to temporary file first, then atomically replace
        temp_path = str(filepath) + ".tmp"
        torch.save(payload, temp_path)
        os.replace(temp_path, str(filepath))

    def _to_cpu_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move state dict tensors to CPU to avoid device serialization issues."""
        return {k: v.detach().to("cpu") for k, v in state_dict.items()}

    def save_last(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        stage: str,
        epoch: int,
        global_step: Optional[int] = None,
        optimizer_encoder: Optional[torch.optim.Optimizer] = None,
        optimizer_decoder: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save last checkpoint."""
        # Fix: save_every=0 means "save every epoch", not "disable"
        if not self.should_save_last or (self.save_every > 0 and epoch % self.save_every != 0):
            return

        self.save_checkpoint(
            filepath=self.checkpoints_dir / "last.pt",
            encoder=encoder,
            decoder=decoder,
            epoch=epoch,
            stage=stage,
            global_step=global_step,
            optimizer_encoder=optimizer_encoder,
            optimizer_decoder=optimizer_decoder,
            metrics=metrics,
            args=args,
        )

        # Also save metadata
        meta = {
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
        }
        self._save_json_atomic(self.checkpoints_dir / "last_meta.json", meta)

    def maybe_save_best(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        stage: str,
        epoch: int,
        probe_metrics: Optional[Dict[str, Any]],
        global_step: Optional[int] = None,
        optimizer_encoder: Optional[torch.optim.Optimizer] = None,
        optimizer_decoder: Optional[torch.optim.Optimizer] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save best checkpoint if current metrics are better than previous best.
        
        Args:
            encoder: Encoder model
            decoder: Decoder model
            stage: Current training stage
            epoch: Current epoch
            probe_metrics: Dictionary of probe metrics
            global_step: Global step (optional)
            optimizer_encoder: Encoder optimizer (optional)
            optimizer_decoder: Decoder optimizer (optional)
            args: Training arguments (optional)
            
        Returns:
            True if best checkpoint was updated, False otherwise
        """
        if not self.save_best or not probe_metrics:
            return False

        # Get current metric value
        current_value = self._extract_metric_value(probe_metrics)
        if current_value is None:
            return False

        # Check if this is a better value
        is_better = False
        if self.best_value is None:
            is_better = True
        elif self.best_mode == "max":
            is_better = current_value > self.best_value
        else:  # min
            is_better = current_value < self.best_value

        # Apply guardrail rule if using id_acc_pos (ensure detection hasn't collapsed)
        if is_better and self.best_metric == "id_acc_pos":
            # Check both reverb and non-reverb TPR values for guardrail
            tpr_value = probe_metrics.get("tpr_at_fpr_1pct_reverb", probe_metrics.get("tpr_at_fpr_1pct"))
            if tpr_value is not None and tpr_value < 0.30:
                # Detection has collapsed, don't save this as best
                is_better = False

        if not is_better:
            return False

        # Update best values
        self.best_value = current_value
        self.best_epoch = epoch
        self.best_stage = stage

        # Save best checkpoint
        self.save_checkpoint(
            filepath=self.checkpoints_dir / "best.pt",
            encoder=encoder,
            decoder=decoder,
            epoch=epoch,
            stage=stage,
            global_step=global_step,
            optimizer_encoder=optimizer_encoder,
            optimizer_decoder=optimizer_decoder,
            metrics=probe_metrics,
            args=args,
        )

        # Also save best metadata
        meta = {
            "stage": stage,
            "epoch": epoch,
            "metric_name": self.best_metric,
            "metric_value": current_value,
            "global_step": global_step,
        }
        self._save_json_atomic(self.checkpoints_dir / "best_meta.json", meta)

        # Copy best weights to run directory for easy access (atomic save)
        best_encoder_path = self.run_dir / "encoder.pt"
        best_decoder_path = self.run_dir / "decoder.pt"

        # Save CPU state dicts to avoid device issues
        encoder_state_cpu = self._to_cpu_state_dict(encoder.state_dict())
        decoder_state_cpu = self._to_cpu_state_dict(decoder.state_dict())

        # Atomic save to prevent corruption on crash
        temp_encoder_path = str(best_encoder_path) + ".tmp"
        temp_decoder_path = str(best_decoder_path) + ".tmp"

        torch.save(encoder_state_cpu, temp_encoder_path)
        os.replace(temp_encoder_path, str(best_encoder_path))

        torch.save(decoder_state_cpu, temp_decoder_path)
        os.replace(temp_decoder_path, str(best_decoder_path))

        return True

    def _extract_metric_value(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract the value of the best metric from the metrics dict."""
        # Special handling for tpr_at_fpr_1pct: prefer reverb variant if available
        if self.best_metric == "tpr_at_fpr_1pct":
            for k in ("tpr_at_fpr_1pct_reverb", "tpr_at_fpr_1pct"):
                if k in metrics and isinstance(metrics[k], (int, float)):
                    return float(metrics[k])
            return None

        # Handle other metrics normally
        if self.best_metric in metrics and isinstance(metrics[self.best_metric], (int, float)):
            return float(metrics[self.best_metric])

        return None

    def _save_json_atomic(self, filepath: Path, data: Any) -> None:
        """Atomically save JSON data."""
        temp_path = str(filepath) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, str(filepath))

    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a checkpoint from the specified filepath.
        
        Args:
            filepath: Path to load checkpoint from
            
        Returns:
            Dictionary containing checkpoint data
        """
        return torch.load(filepath, map_location="cpu")

    def resume_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        optimizer_encoder: Optional[torch.optim.Optimizer] = None,
        optimizer_decoder: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to resume from
            encoder: Encoder model to load weights into
            decoder: Decoder model to load weights into
            optimizer_encoder: Encoder optimizer to load state into (optional)
            optimizer_decoder: Decoder optimizer to load state into (optional)
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        ckpt = self.load_checkpoint(checkpoint_path)
        
        # Load model weights
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        
        # Load optimizer states if provided
        if optimizer_encoder and ckpt.get("opt_encoder"):
            optimizer_encoder.load_state_dict(ckpt["opt_encoder"])
        if optimizer_decoder and ckpt.get("opt_decoder"):
            optimizer_decoder.load_state_dict(ckpt["opt_decoder"])
        
        return ckpt

    @staticmethod
    def get_default_best_metric(epochs_s1: int, epochs_s2: int, epochs_s1b_post: int) -> str:
        """
        Get the default best metric based on training schedule.

        Args:
            epochs_s1: Number of stage 1 epochs
            epochs_s2: Number of stage 2 epochs
            epochs_s1b_post: Number of stage 3 epochs

        Returns:
            Default metric name
        """
        if epochs_s1b_post > 0:
            # ID finetune mode - prefer attribution accuracy
            return "id_acc_pos"
        else:
            # Detection-first mode - prefer detection robustness
            return "tpr_at_fpr_1pct"  # Default to basic tpr, reverb variant handled in extraction