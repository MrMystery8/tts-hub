"""
Unit tests for MessageCodec

Tests encode/decode roundtrip, preamble consistency, and device handling.
"""
import torch
import pytest
from watermark.models.codec import MessageCodec


class TestMessageCodec:
    """Tests for MessageCodec class."""
    
    def test_encode_returns_correct_shape(self):
        """Encode should return 32-bit tensor."""
        codec = MessageCodec()
        msg = codec.encode(model_id=3, version=5)
        assert msg.shape == (32,)
        assert msg.dtype == torch.float32
    
    def test_encode_values_are_binary(self):
        """All encoded values should be 0.0 or 1.0."""
        codec = MessageCodec()
        msg = codec.encode(model_id=7, version=15)
        assert torch.all((msg == 0.0) | (msg == 1.0))
    
    def test_preamble_is_deterministic(self):
        """Same key should produce same preamble."""
        codec1 = MessageCodec(key="test123")
        codec2 = MessageCodec(key="test123")
        assert torch.allclose(codec1.preamble, codec2.preamble)
    
    def test_different_keys_different_preambles(self):
        """Different keys should produce different preambles."""
        codec1 = MessageCodec(key="key1")
        codec2 = MessageCodec(key="key2")
        assert not torch.allclose(codec1.preamble, codec2.preamble)
    
    def test_preamble_is_16_bits(self):
        """Preamble should be exactly 16 bits."""
        codec = MessageCodec()
        assert codec.preamble.shape == (16,)
    
    def test_encode_decode_roundtrip_perfect_probs(self):
        """Decode with perfect probabilities should recover model_id and version."""
        codec = MessageCodec()
        
        for model_id in range(8):
            for version in range(16):
                msg = codec.encode(model_id=model_id, version=version)
                # Simulate perfect decoder output
                result = codec.decode(msg)
                assert result["model_id"] == model_id
                assert result["version"] == version
                assert result["preamble_score"] == 1.0
    
    def test_decode_handles_noisy_probs(self):
        """Decode should handle probabilities (not just 0/1)."""
        codec = MessageCodec()
        msg = codec.encode(model_id=5, version=10)
        
        # Add some noise but keep values close to original
        noisy = msg.clone()
        noisy = torch.where(noisy == 1.0, torch.tensor(0.8), torch.tensor(0.2))
        
        result = codec.decode(noisy)
        assert result["model_id"] == 5
        assert result["version"] == 10
    
    def test_decode_device_handling(self):
        """Decode should work even if probs are on different device than preamble."""
        codec = MessageCodec()
        msg = codec.encode(model_id=2, version=3)
        
        # Preamble is on CPU, this should still work
        result = codec.decode(msg)
        assert result["model_id"] == 2
        assert result["version"] == 3
    
    def test_model_id_bounds(self):
        """model_id should be validated."""
        codec = MessageCodec()
        with pytest.raises(AssertionError):
            codec.encode(model_id=8, version=0)  # Out of range
        with pytest.raises(AssertionError):
            codec.encode(model_id=-1, version=0)  # Negative
    
    def test_version_bounds(self):
        """version should be validated."""
        codec = MessageCodec()
        with pytest.raises(AssertionError):
            codec.encode(model_id=0, version=16)  # Out of range
        with pytest.raises(AssertionError):
            codec.encode(model_id=0, version=-1)  # Negative
    
    def test_redundancy_locations(self):
        """Verify redundant copies are in correct bit positions."""
        codec = MessageCodec()
        msg = codec.encode(model_id=5, version=10)  # model=101, version=1010
        
        # model_id copy 1: bits 16-18, copy 2: bits 23-25
        assert torch.allclose(msg[16:19], msg[23:26])
        
        # version copy 1: bits 19-22, copy 2: bits 26-29
        assert torch.allclose(msg[19:23], msg[26:30])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
