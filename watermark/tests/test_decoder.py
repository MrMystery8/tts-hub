"""
Unit tests for WatermarkDecoder and related classes
"""
import torch
import pytest
from watermark.models.decoder import WatermarkDecoder, SlidingWindowDecoder, ClipDecisionRule
from watermark.models.codec import MessageCodec


class TestWatermarkDecoder:
    """Tests for WatermarkDecoder."""
    
    def test_forward_output_shapes(self):
        """Forward pass should return dict with correct shapes."""
        decoder = WatermarkDecoder(msg_bits=32, n_models=8)
        # Batch size 4, 16000 samples (1 sec)
        audio = torch.randn(4, 16000)
        outputs = decoder(audio)
        
        assert outputs["detect_logit"].shape == (4, 1)
        assert outputs["message_logits"].shape == (4, 32)
        assert outputs["model_logits"].shape == (4, 9)  # 8 models + 1 unknown
        assert outputs["detect_prob"].shape == (4, 1)
        assert outputs["message_probs"].shape == (4, 32)
    
    def test_mps_safe_mel(self):
        """Mel filterbank should be a buffer (not recreated)."""
        decoder = WatermarkDecoder()
        assert hasattr(decoder, 'mel_fb')
        assert isinstance(decoder.mel_fb, torch.Tensor)
        assert hasattr(decoder, 'stft_window')


class TestSlidingWindowDecoder:
    """Tests for SlidingWindowDecoder."""
    
    def test_forward_long_audio(self):
        """Should handle audio longer than window."""
        base_decoder = WatermarkDecoder()
        decoder = SlidingWindowDecoder(base_decoder, window=16000, hop_ratio=0.5)
        
        # 3 seconds audio (approx 5 windows: 0, 0.5, 1, 1.5, 2)
        audio = torch.randn(2, 48000)
        outputs = decoder(audio)
        
        # Check window splitting
        assert outputs["n_windows"] == 5
        assert outputs["all_window_probs"].shape == (2, 5)
        assert outputs["all_message_probs"].shape == (2, 5, 32)
        
        # Check aggregation
        assert outputs["clip_detect_prob"].shape == (2,)
        assert outputs["clip_detect_logit"].shape == (2,)
    
    def test_forward_short_audio(self):
        """Should handle audio shorter than window (via padding)."""
        base_decoder = WatermarkDecoder()
        decoder = SlidingWindowDecoder(base_decoder, window=16000)
        
        # 0.5 seconds audio
        audio = torch.randn(2, 8000)
        outputs = decoder(audio)
        
        # Should be padded to 1 window
        assert outputs["n_windows"] == 1
        assert outputs["all_window_probs"].shape == (2, 1)


class TestClipDecisionRule:
    """Tests for decision logic."""
    
    def test_decision_positive(self):
        """Should detect positive if windows are confident."""
        rule = ClipDecisionRule(detect_threshold=0.8, preamble_min=15)
        codec = MessageCodec()
        
        # Construct fake outputs for a single clip
        # 3 windows, all confident
        n_win = 3
        
        # High detection prob
        clip_prob = 0.95
        win_probs = torch.tensor([0.9, 0.95, 0.85])
        
        # Valid messages (model_id=3, version=1) with perfect preamble
        valid_msg = codec.encode(3, 1)
        msg_probs = valid_msg.unsqueeze(0).expand(n_win, -1)
        
        outputs = {
            "clip_detect_prob": clip_prob,
            "all_window_probs": win_probs,
            "all_message_probs": msg_probs,
        }
        
        result = rule.decide(outputs, codec)
        assert result["positive"] is True
        assert result["model_id"] == 3
        assert result["vote_count"] == 3
    
    def test_decision_negative_low_clip_prob(self):
        """Should be negative if clip probability is low."""
        rule = ClipDecisionRule(detect_threshold=0.8)
        codec = MessageCodec()
        
        outputs = {
            "clip_detect_prob": 0.5,  # Too low
             # These don't matter if clip prob is low
            "all_window_probs": torch.tensor([0.9]),
            "all_message_probs": torch.zeros(1, 32),
        }
        
        result = rule.decide(outputs, codec)
        assert result["positive"] is False
        assert result["reason"] == "clip_detect_low"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
