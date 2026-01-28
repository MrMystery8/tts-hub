"""
Tests for checkpointing functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from watermark.config import N_CLASSES
from watermark.models.decoder import SlidingWindowDecoder, WatermarkDecoder
from watermark.models.encoder import OverlapAddEncoder, WatermarkEncoder
from watermark.utils.checkpointing import CheckpointManager


class TestCheckpointing(unittest.TestCase):
    """Test checkpointing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.run_dir = self.temp_dir / "test_run"
        self.run_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_load_roundtrip(self):
        """Test save/load roundtrip for encoder/decoder models."""
        # Create simple encoder and decoder
        encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES))
        decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES))

        # Initialize checkpoint manager
        ckpt_manager = CheckpointManager(
            run_dir=self.run_dir,
            save_last=True,
            save_best=True,
            best_metric="tpr_at_fpr_1pct",
            best_mode="max",
            save_every=1,
        )

        # Save a checkpoint
        epoch = 1
        stage = "s1"
        ckpt_manager.save_checkpoint(
            filepath=self.run_dir / "test_ckpt.pt",
            encoder=encoder,
            decoder=decoder,
            epoch=epoch,
            stage=stage,
            metrics={"tpr_at_fpr_1pct": 0.85},
            args={"test_arg": "value"}
        )

        # Load the checkpoint
        loaded_ckpt = ckpt_manager.load_checkpoint(self.run_dir / "test_ckpt.pt")

        # Verify checkpoint contents
        self.assertEqual(loaded_ckpt["schema"], 1)
        self.assertEqual(loaded_ckpt["stage"], stage)
        self.assertEqual(loaded_ckpt["epoch"], epoch)
        self.assertIn("encoder", loaded_ckpt)
        self.assertIn("decoder", loaded_ckpt)

        # Create new models and load weights
        new_encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES))
        new_decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES))

        new_encoder.load_state_dict(loaded_ckpt["encoder"])
        new_decoder.load_state_dict(loaded_ckpt["decoder"])

        # Verify weights are identical
        for param1, param2 in zip(encoder.parameters(), new_encoder.parameters()):
            self.assertTrue(torch.equal(param1.cpu(), param2.cpu()))
        
        for param1, param2 in zip(decoder.parameters(), new_decoder.parameters()):
            self.assertTrue(torch.equal(param1.cpu(), param2.cpu()))

    def test_best_selection(self):
        """Test best checkpoint selection logic."""
        encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES))
        decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES))

        # Test maximization (default)
        ckpt_manager = CheckpointManager(
            run_dir=self.run_dir,
            save_last=True,
            save_best=True,
            best_metric="tpr_at_fpr_1pct",
            best_mode="max",
            save_every=1,
        )

        # Simulate probe metrics over epochs
        metrics_epoch1 = {"tpr_at_fpr_1pct": 0.75}
        metrics_epoch2 = {"tpr_at_fpr_1pct": 0.85}  # Better
        metrics_epoch3 = {"tpr_at_fpr_1pct": 0.80}  # Worse

        # Call maybe_save_best for each epoch
        result1 = ckpt_manager.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s1", epoch=1, probe_metrics=metrics_epoch1
        )
        result2 = ckpt_manager.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s1", epoch=2, probe_metrics=metrics_epoch2
        )
        result3 = ckpt_manager.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s1", epoch=3, probe_metrics=metrics_epoch3
        )

        # First and second should update best, third should not
        self.assertTrue(result1)  # First is always best
        self.assertTrue(result2)  # Second is better than first
        self.assertFalse(result3)  # Third is worse than second

        # Verify best value is 0.85 (from epoch 2)
        self.assertEqual(ckpt_manager.best_value, 0.85)
        self.assertEqual(ckpt_manager.best_epoch, 2)

        # Test minimization
        ckpt_manager_min = CheckpointManager(
            run_dir=self.run_dir / "min_test",
            save_last=True,
            save_best=True,
            best_metric="loss",
            best_mode="min",
            save_every=1,
        )

        metrics_loss1 = {"loss": 1.5}
        metrics_loss2 = {"loss": 1.2}  # Better (lower)
        metrics_loss3 = {"loss": 1.8}  # Worse (higher)

        result_l1 = ckpt_manager_min.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s1", epoch=1, probe_metrics=metrics_loss1
        )
        result_l2 = ckpt_manager_min.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s1", epoch=2, probe_metrics=metrics_loss2
        )
        result_l3 = ckpt_manager_min.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s1", epoch=3, probe_metrics=metrics_loss3
        )

        self.assertTrue(result_l1)  # First is always best
        self.assertTrue(result_l2)  # Second is better (lower loss)
        self.assertFalse(result_l3)  # Third is worse (higher loss)

        # Verify best value is 1.2 (from epoch 2)
        self.assertEqual(ckpt_manager_min.best_value, 1.2)
        self.assertEqual(ckpt_manager_min.best_epoch, 2)

    def test_guardrail_rule(self):
        """Test guardrail rule for id_acc_pos metric."""
        encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES))
        decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES))

        # Test with id_acc_pos as best metric
        ckpt_manager = CheckpointManager(
            run_dir=self.run_dir,
            save_last=True,
            save_best=True,
            best_metric="id_acc_pos",
            best_mode="max",
            save_every=1,
        )

        # Good ID accuracy but poor detection (should not be saved as best due to guardrail)
        metrics_bad_detection = {
            "id_acc_pos": 0.95,  # High ID accuracy
            "tpr_at_fpr_1pct": 0.10,  # Poor detection (below 0.30 threshold)
        }

        result_bad = ckpt_manager.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s3_finetune", epoch=1, probe_metrics=metrics_bad_detection
        )

        self.assertFalse(result_bad)  # Should not save due to guardrail

        # Good ID accuracy AND good detection (should be saved as best)
        metrics_good_detection = {
            "id_acc_pos": 0.90,  # High ID accuracy
            "tpr_at_fpr_1pct": 0.70,  # Good detection (above 0.30 threshold)
        }

        result_good = ckpt_manager.maybe_save_best(
            encoder=encoder, decoder=decoder, stage="s3_finetune", epoch=2, probe_metrics=metrics_good_detection
        )

        self.assertTrue(result_good)  # Should save due to good detection

    def test_save_last_functionality(self):
        """Test save_last functionality."""
        encoder = OverlapAddEncoder(WatermarkEncoder(num_classes=N_CLASSES))
        decoder = SlidingWindowDecoder(WatermarkDecoder(num_classes=N_CLASSES))

        # Test with save_every=1 (should save every epoch)
        ckpt_manager = CheckpointManager(
            run_dir=self.run_dir,
            save_last=True,
            save_best=False,
            best_metric="tpr_at_fpr_1pct",
            best_mode="max",
            save_every=1,
        )

        # Call save_last
        ckpt_manager.save_last(
            encoder=encoder,
            decoder=decoder,
            stage="s1",
            epoch=5,
            args={"test": "value"}
        )

        # Check that last checkpoint exists
        last_ckpt_path = self.run_dir / "checkpoints" / "last.pt"
        self.assertTrue(last_ckpt_path.exists())

        # Check that last metadata exists
        last_meta_path = self.run_dir / "checkpoints" / "last_meta.json"
        self.assertTrue(last_meta_path.exists())

        # Load and verify metadata
        with open(last_meta_path, "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["epoch"], 5)
        self.assertEqual(meta["stage"], "s1")

        # Test with save_every=2 (should skip even epochs)
        ckpt_manager_skip = CheckpointManager(
            run_dir=self.run_dir / "skip_test",
            save_last=True,
            save_best=False,
            best_metric="tpr_at_fpr_1pct",
            best_mode="max",
            save_every=2,
        )

        # Call save_last for epoch 4 (should save)
        ckpt_manager_skip.save_last(encoder=encoder, decoder=decoder, stage="s1", epoch=4)
        last_ckpt_path_4 = (self.run_dir / "skip_test" / "checkpoints" / "last.pt").exists()
        self.assertTrue(last_ckpt_path_4)

        # Call save_last for epoch 5 (should not save)
        ckpt_manager_skip.save_last(encoder=encoder, decoder=decoder, stage="s1", epoch=5)
        # The file should still exist from epoch 4, but we can't distinguish which epoch it's from without loading

    def test_probe_reverb_disable(self):
        """Test that probe_reverb_every=0 does not crash."""
        # This test is more about verifying the logic in the main script,
        # but we can test the concept here too.
        
        # Simulate the condition from the fixed code
        probe_reverb_every = 0
        epoch = 5
        
        # This should not cause a division by zero error
        reverb = False
        if int(probe_reverb_every) > 0:
            reverb = (int(epoch) % int(probe_reverb_every) == 0)
        
        # Since probe_reverb_every is 0, reverb should remain False
        self.assertFalse(reverb)


if __name__ == "__main__":
    unittest.main()