"""
Message Codec for Audio Watermarking

Handles encoding model_id and version into a 32-bit message tensor,
using CRC-16 for error validation instead of fragile preambles.

LEGACY NOTE:
The current watermarking pipeline uses multiclass attribution (class 0 = clean, class 1..K = models)
and does not use a bit payload codec. This module is retained for older experiments only.

Message format (32 bits):
- Bits 0-7:   Model ID (8 bits, 0-255)
- Bits 8-15:  Version (8 bits, 0-255)
- Bits 16-31: CRC-16-CCITT (16 bits)

Effective payload: 16 bits + 16 bits robust error check.
"""
import torch

class MessageCodec:
    """
    Encodes ID/Version into 32-bit message with CRC-16.
    Decodes probability vectors verifying CRC.
    """
    
    # Fixed Scramble Mask (0xA5 for ID, 0x5A for Version)
    # 0xA5 = 10100101 (4 ones)
    # 0x5A = 01011010 (4 ones)
    # Total = 0xA55A (Balanced)
    SCRAMBLE_MASK = 0xA55A

    def __init__(self):
        # Precompute CRC table (CCITT-FALSE: Poly 0x1021, init 0xFFFF)
        self.crc_table = self._generate_crc_table(0x1021)
        
    def _generate_crc_table(self, poly: int) -> list[int]:
        table = []
        for i in range(256):
            crc = i << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ poly
                else:
                    crc = crc << 1
            table.append(crc & 0xFFFF)
        return table
        
    def _compute_crc(self, data_bytes: bytes) -> int:
        crc = 0xFFFF
        for b in data_bytes:
            # CCITT-FALSE
            idx = (crc >> 8) ^ b
            crc = ((crc << 8) & 0xFFFF) ^ self.crc_table[idx]
        return crc
    
    def _scramble(self, val_16bit: int) -> int:
        return val_16bit ^ self.SCRAMBLE_MASK
        
    def encode(self, model_id: int, version: int) -> torch.Tensor:
        """
        Encode model_id and version into a 32-bit message tensor.
        Applies XOR Scrambling to ensure balanced bits (avoiding all-zeros).
        
        Args:
            model_id: 8 bits (0-255)
            version: 8 bits (0-255)
        
        Returns:
            torch.Tensor: Shape (32,), values 0.0 or 1.0
        """
        if not (0 <= model_id < 256):
            raise ValueError(f"model_id must be 0-255, got {model_id}")
        if not (0 <= version < 256):
            raise ValueError(f"version must be 0-255, got {version}")
            
        # 1. Combine
        raw_payload = (model_id << 8) | version
        
        # 2. Scramble
        scrambled_payload = self._scramble(raw_payload)
        
        # 3. CRC (on scrambled payload)
        # We process high byte (ID slot) then low byte (Ver slot)
        scram_hi = (scrambled_payload >> 8) & 0xFF
        scram_lo = scrambled_payload & 0xFF
        data = bytes([scram_hi, scram_lo])
        crc = self._compute_crc(data)
        
        # Build 32 bits
        msg = torch.zeros(32)
        
        # 0-7: Scrambled ID Slot
        for i in range(8):
            msg[i] = (scram_hi >> i) & 1
            
        # 8-15: Scrambled Version Slot
        for i in range(8):
            msg[8 + i] = (scram_lo >> i) & 1
            
        # 16-31: CRC
        for i in range(16):
            msg[16 + i] = (crc >> i) & 1
            
        return msg
    
    def decode(self, probs: torch.Tensor) -> dict:
        """
        Decode probability vector to message + validation.
        
        Args:
            probs: Shape (32,) probability vector (0.0-1.0)
        
        Returns:
            dict with:
                - valid: bool (CRC check pass)
                - model_id: int
                - version: int
                - confidence: float
        """
        bits = (probs > 0.5).int()
        
        # Extract Scrambled Fields
        scram_hi = 0
        for i in range(8):
            if bits[i]: scram_hi |= (1 << i)
            
        scram_lo = 0
        for i in range(8):
            if bits[8+i]: scram_lo |= (1 << i)
            
        crc_received = 0
        for i in range(16):
            if bits[16+i]: crc_received |= (1 << i)
            
        # Validate (on scrambled data)
        data = bytes([scram_hi, scram_lo])
        crc_calculated = self._compute_crc(data)
        
        valid = (crc_received == crc_calculated)
        
        # Unscramble
        scrambled_payload = (scram_hi << 8) | scram_lo
        raw_payload = self._scramble(scrambled_payload)
        
        model_id = (raw_payload >> 8) & 0xFF
        version = raw_payload & 0xFF
        
        # Confidence
        confidence = (probs - 0.5).abs().mean().item() * 2
        
        return {
            "valid": valid,
            "model_id": model_id,
            "version": version,
            "confidence": confidence,
            "crc_pass": valid
        }
