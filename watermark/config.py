"""
Watermark Configuration Constants

All constants used across the watermarking system.
Matches WATERMARK_PROJECT_PLAN.md v16 specifications.
"""
import torch

# =============================================================================
# Audio Parameters
# =============================================================================
SAMPLE_RATE = 16000  # 16kHz for all processing
SEGMENT_SECONDS = 3.0  # Fixed segment duration for training
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_SECONDS)  # 48000 samples

# =============================================================================
# Window Parameters (for sliding window processing)
# =============================================================================
WINDOW_SAMPLES = 16000  # 1 second window
HOP_RATIO = 0.5  # 50% overlap
HOP_SAMPLES = int(WINDOW_SAMPLES * HOP_RATIO)  # 8000 samples

# =============================================================================
# Message Format
# =============================================================================
MSG_BITS = 32
PREAMBLE_BITS = 16  # Known sync pattern
PAYLOAD_BITS = 7  # 3 model_id + 4 version
N_MODELS = 8  # Maximum number of model IDs (3 bits)

# =============================================================================
# Top-K Aggregation
# =============================================================================
TOP_K = 3  # Number of top windows for aggregation

# =============================================================================
# Encoder Parameters
# =============================================================================
ENCODER_HIDDEN = 32
ENCODER_GROUPS = 4  # GroupNorm groups

# =============================================================================
# Decoder Parameters
# =============================================================================
DECODER_N_FFT = 512
DECODER_N_MELS = 80

# =============================================================================
# Device Detection
# =============================================================================
def get_device() -> torch.device:
    """Get the best available device (MPS for Mac, CUDA for GPU, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# =============================================================================
# Preamble Key (for deterministic preamble generation)
# =============================================================================
DEFAULT_PREAMBLE_KEY = "fyp2026"
