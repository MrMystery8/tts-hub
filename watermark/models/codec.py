"""
LEGACY SHIM: bit-payload MessageCodec.

The current watermarking pipeline uses multiclass attribution (no bit payload codec).
This module is kept only so older experiments/imports can still resolve.
"""

from watermark.legacy.codec import MessageCodec

__all__ = ["MessageCodec"]

