from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    worker_entry: str
    description: str


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            id="index-tts2",
            name="IndexTTS2 (index-tts)",
            worker_entry="worker_index_tts2.py",
            description="PyTorch AR zero-shot TTS with emotion control; MPS-optimized worker mode.",
        ),
        ModelSpec(
            id="chatterbox-multilingual",
            name="Chatterbox Multilingual (chatterbox-multilingual)",
            worker_entry="worker_chatterbox_mtl.py",
            description="Multilingual TTS with optional cloning; long-form chunking+stitching.",
        ),
        ModelSpec(
            id="f5-hindi-urdu",
            name="F5 Hindi/Urdu (f5-hindi-urdu)",
            worker_entry="worker_f5_hindi_urdu.py",
            description="Hindi/Urdu voice cloning with Roman→Devanagari conversion and overrides.",
        ),
        ModelSpec(
            id="cosyvoice3-mlx",
            name="CosyVoice3-MLX (cosyvoice3-mlx)",
            worker_entry="worker_cosyvoice3_mlx.py",
            description="MLX-based CosyVoice3 with zero-shot/cross-lingual/instruct modes.",
        ),
        ModelSpec(
            id="pocket-tts",
            name="Pocket TTS (pocket-tts)",
            worker_entry="worker_pocket_tts.py",
            description="CPU-first low-latency streaming TTS with optional voice cloning.",
        ),
        ModelSpec(
            id="voxcpm-ane",
            name="VoxCPM-ANE (voxcpm-ane)",
            worker_entry="worker_voxcpm_ane.py",
            description="CoreML/ANE VoxCPM with prompt-cache voice cloning (requires transcript).",
        ),
    ]

