"""
Message Codec for Audio Watermarking

Handles encoding model_id and version into a 32-bit message tensor,
and decoding probability vectors back to structured information.

Message format:
- Bits 0-15:  Preamble (16-bit, keyed pseudo-random)
- Bits 16-18: Model ID (3 bits, 0-7)
- Bits 19-22: Version (4 bits, 0-15)
- Bits 23-25: Model ID copy (redundancy)
- Bits 26-29: Version copy (redundancy)
- Bits 30-31: Reserved

Effective payload: 7 bits with 2× soft-average redundancy
"""
import hashlib
import torch


class MessageCodec:
    """
    Encodes model_id/version into 32-bit message tensor.
    Decodes probability vectors with soft-averaging (NOT OR logic!).
    
    BUG FIX from project plan: preamble moved to device in decode()
    to avoid CPU/GPU mismatch.
    """
    
    def __init__(self, key: str = "fyp2026"):
        """
        Initialize codec with a key for preamble generation.
        
        Args:
            key: String key for deterministic preamble generation.
        """
        h = hashlib.sha256(key.encode()).digest()
        self.preamble = torch.tensor(
            [int(b) for b in format(int.from_bytes(h[:2], 'big'), '016b')],
            dtype=torch.float32
        )
    
    def encode(self, model_id: int, version: int = 1) -> torch.Tensor:
        """
        Encode model_id and version into a 32-bit message tensor.
        
        Args:
            model_id: Model identifier (0-7, 3 bits)
            version: Version number (0-15, 4 bits)
        
        Returns:
            torch.Tensor: Shape (32,), float32, values 0.0 or 1.0
        """
        assert 0 <= model_id < 8, f"model_id must be 0-7, got {model_id}"
        assert 0 <= version < 16, f"version must be 0-15, got {version}"
        
        msg = torch.zeros(32)
        msg[0:16] = self.preamble
        
        # Payload copy 1 (bits 16-22)
        for i in range(3):
            msg[16 + i] = (model_id >> i) & 1
        for i in range(4):
            msg[19 + i] = (version >> i) & 1
        
        # Payload copy 2 (bits 23-29, for soft-average)
        msg[23:26] = msg[16:19]  # model_id copy
        msg[26:30] = msg[19:23]  # version copy
        
        return msg
    
    def decode(self, probs: torch.Tensor) -> dict:
        """
        Decode probability vector to message with soft-averaging.
        
        Uses soft-averaging for redundant bits (NOT OR logic!).
        OR-logic `(a+b)>=1` biases toward 1s - WRONG.
        Soft-average: `(p1+p2)/2` then threshold - CORRECT.
        
        Args:
            probs: Shape (32,) probability vector from decoder
        
        Returns:
            dict with:
                - valid: bool (always True, let DecisionRule threshold)
                - model_id: int (0-7)
                - version: int (0-15)
                - preamble_score: float (fraction of matching preamble bits)
                - confidence: float (mean confidence of payload bits)
        """
        # FIX: Move preamble to same device as probs!
        preamble = self.preamble.to(probs.device)
        
        preamble_bits = (probs[0:16] > 0.5).int()
        preamble_match = (preamble_bits == preamble.int()).sum().item()
        
        # SOFT-AVERAGE: average probs from both copies, then threshold
        model_probs = (probs[16:19] + probs[23:26]) / 2
        ver_probs = (probs[19:23] + probs[26:30]) / 2
        
        model_bits = (model_probs > 0.5).int()
        ver_bits = (ver_probs > 0.5).int()
        
        model_id = model_bits[0] + 2 * model_bits[1] + 4 * model_bits[2]
        version = ver_bits[0] + 2 * ver_bits[1] + 4 * ver_bits[2] + 8 * ver_bits[3]
        
        return {
            "valid": True,  # Let DecisionRule be the source of truth for thresholds
            "model_id": model_id.item(),
            "version": version.item(),
            "preamble_score": preamble_match / 16,
            "confidence": torch.cat([model_probs, ver_probs]).mean().item(),
        }
    
    def get_preamble(self) -> torch.Tensor:
        """Return the preamble tensor (for training use)."""
        return self.preamble.clone()
