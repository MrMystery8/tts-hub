# TTS Hub: Unified Audio Generation Platform

> **Status:** Active Development (FYP Phase: Watermark Classifier Training)

A centralized hub for running and managing multiple Apple Silicon-optimized TTS and voice cloning models. Now integrating **Audio Provenance Watermarking** with trained supervised classification.

---

## 🏗️ Architecture

TTS Hub unifies 6 distinct inference stacks under a single API and Web UI:

| Model | Type | Architecture | Optimization |
|-------|------|--------------|--------------|
| **IndexTTS2** | Voice Cloning | Retrieval-based VC | MPS (High) |
| **Chatterbox** | Multilingual TTS | Transformer | MTL / Ane |
| **F5 Hindi/Urdu** | TTS | F5-TTS | CoreML |
| **CosyVoice3** | Voice Cloning | Flow Matching | MLX |
| **PocketTTS** | Lightweight TTS | VITS | CPU/Mobile |
| **VoxCPM-ANE** | Voice Cloning | GPT-VITS | ANE (Neural Engine) |

## 🌊 Current Focus: Watermarking w/ Trained Classifier

We are implementing an end-to-end watermarking system to detect AI-generated audio and attribute it to specific models.

**Key Components:**
- **Watermark Module (`watermark/`):** Custom WavMark-inspired encoder/decoder implementation.
- **Dataset Pipeline (`dataset/`):** Scripts to generate standard & attacked samples from all 6 models.
- **Classifier (`checkpoints/`):** PyTorch model trained from scratch to detect provenance.

👉 **See [docs/WATERMARK_PROJECT_PLAN.md](docs/WATERMARK_PROJECT_PLAN.md) for the full FYP specification.**

---

## 📂 Project Structure

```
tts-hub/
├── custom_ui/          # Unified Web Interface (HTML/JS/CSS)
├── hub/                # Core Logic (Model Registry, Process Management)
├── workers/            # Independent Worker Scripts for each Model
├── watermark/          # [NEW] Encoder/Decoder Model Implementation
├── dataset/            # [NEW] Dataset Generation & Augmentation
├── scripts/            # [NEW] Training & Utility Scripts
├── docs/               # Project Documentation
└── webui.py            # Main Entry Point
```

## 🚀 Quick Start

### 1. Prerequisites
- macOS (Apple Silicon recommended)
- `ffmpeg` installed (`brew install ffmpeg`)
- Python 3.10+

### 2. Installation
```bash
# Clone and setup environment
git clone https://github.com/yourusername/tts-hub.git
cd tts-hub
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Hub
```bash
# Start the Web UI
./run.sh
# OR manually: python webui.py
```
Access the UI at: `http://localhost:7860`

### 4. Run Watermark Training
```bash
# Quick Smoke Train (Single Run)
./.venv/bin/python -m watermark.scripts.quick_voice_smoke_train \
    --source_dir mini_benchmark_data \
    --epochs_s1 6

# Overnight Tuner (Auto-Optimize Weights)
./.venv/bin/python -m watermark.scripts.overnight_tune_s1 \
    --source_dir mini_benchmark_data \
    --out_root outputs/dashboard_runs/overnight_01
```

For full details, see **[docs/WATERMARK_RUNBOOK.md](docs/WATERMARK_RUNBOOK.md)**.

---

## 📚 Documentation

Detailed documentation has been moved to the `docs/` folder:
- **[Watermark Project Plan](docs/WATERMARK_PROJECT_PLAN.md)** - Comprehensive FYP plan
- **[Roadmap](docs/ROADMAP_AND_IMPROVEMENTS.md)** - Future features
- **[Spec Sheet](docs/SPEC_SHEET.md)** - Technical specifications
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Dev logs

---

## 🛠️ Diagnostics

If models fail to load or environment issues occur:
```bash
python tools/doctor.py
```
