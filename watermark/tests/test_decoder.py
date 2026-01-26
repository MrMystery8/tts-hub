"""
Unit tests for WatermarkDecoder (multiclass attribution) and related classes.
"""

import torch

from watermark.config import CLASS_CLEAN, N_CLASSES
from watermark.models.decoder import AttributionDecisionRule, SlidingWindowDecoder, WatermarkDecoder


class TestWatermarkDecoder:
    def test_forward_output_shapes(self):
        decoder = WatermarkDecoder(num_classes=N_CLASSES)
        audio = torch.randn(4, 16000)
        outputs = decoder(audio)

        assert outputs["class_logits"].shape == (4, N_CLASSES)
        assert outputs["class_probs"].shape == (4, N_CLASSES)
        assert outputs["wm_prob"].shape == (4,)

    def test_mps_safe_mel_buffers(self):
        decoder = WatermarkDecoder(num_classes=N_CLASSES)
        assert hasattr(decoder, "mel_fb")
        assert isinstance(decoder.mel_fb, torch.Tensor)
        assert hasattr(decoder, "stft_window")


class TestSlidingWindowDecoder:
    def test_forward_long_audio(self):
        base = WatermarkDecoder(num_classes=N_CLASSES)
        decoder = SlidingWindowDecoder(base, window=16000, hop_ratio=0.5)
        audio = torch.randn(2, 48000)
        outputs = decoder(audio)

        assert outputs["n_windows"] == 5
        assert outputs["all_window_class_logits"].shape == (2, 5, N_CLASSES)
        assert outputs["all_window_class_probs"].shape == (2, 5, N_CLASSES)
        assert outputs["all_window_wm_probs"].shape == (2, 5)

        assert outputs["clip_class_logits"].shape == (2, N_CLASSES)
        assert outputs["clip_class_probs"].shape == (2, N_CLASSES)
        assert outputs["clip_wm_prob"].shape == (2,)

    def test_forward_short_audio(self):
        base = WatermarkDecoder(num_classes=N_CLASSES)
        decoder = SlidingWindowDecoder(base, window=16000)
        audio = torch.randn(2, 8000)
        outputs = decoder(audio)
        assert outputs["n_windows"] == 1
        assert outputs["all_window_wm_probs"].shape == (2, 1)


class TestAttributionDecisionRule:
    def test_decision_positive(self):
        rule = AttributionDecisionRule(wm_threshold=0.8)
        # Pretend the model predicts class 3 with high confidence and low clean prob.
        logits = torch.zeros(1, N_CLASSES)
        logits[0, 3] = 5.0
        outputs = {"clip_wm_prob": torch.tensor(0.95), "clip_class_logits": logits}
        d = rule.decide(outputs)
        assert d["positive"] is True
        assert d["pred_class"] == 3
        assert d["pred_model_id"] == 2  # class-1

    def test_decision_negative(self):
        rule = AttributionDecisionRule(wm_threshold=0.8)
        logits = torch.zeros(1, N_CLASSES)
        logits[0, int(CLASS_CLEAN)] = 5.0
        outputs = {"clip_wm_prob": torch.tensor(0.1), "clip_class_logits": logits}
        d = rule.decide(outputs)
        assert d["positive"] is False
        assert d["pred_class"] == int(CLASS_CLEAN)
        assert d["pred_model_id"] is None

