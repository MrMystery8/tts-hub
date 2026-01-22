# TTS Hub

Single local Web UI that routes to multiple Apple-Silicon-friendly TTS / voice-cloning stacks:

- `index-tts` (IndexTTS2)
- `chatterbox-multilingual` (Chatterbox Multilingual)
- `f5-hindi-urdu` (F5 Hindi/Urdu)
- `cosyvoice3-mlx` (CosyVoice3-MLX)
- `pocket-tts` (Pocket TTS)
- `voxcpm-ane` (VoxCPM-ANE)

## Requirements

- `ffmpeg` available on `PATH` (macOS: `brew install ffmpeg`)
- A Python env for the hub itself:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Each model repo is expected to exist next to `tts-hub/`, and to have its own `.venv` at `<repo>/.venv/`.

## Run

From the repo root:

```bash
python3 tts-hub/webui.py --port 7891
```

Open `http://localhost:7891`.

## Quick Diagnostics

```bash
python3 tts-hub/tools/doctor.py
```

## Docs

- `tts-hub/SPEC_SHEET.md`
- `tts-hub/IMPLEMENTATION_SUMMARY.md`
- `tts-hub/ROADMAP_AND_IMPROVEMENTS.md`
