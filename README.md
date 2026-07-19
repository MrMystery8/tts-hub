# TTS Hub

TTS Hub is a local-first speech generation and voice preservation system for Apple Silicon. It presents multiple text-to-speech and voice-cloning backends through one FastAPI service, a desktop web client, and a mobile companion, with integrated audio watermarking for bounded provenance.

## Supported Runtime

The supported application flow is:

```text
run.sh -> app.py -> webui.py -> desktop/
                              -> mobile/
```

- `app.py` starts the supported desktop application.
- `webui.py` provides the FastAPI service and API routes.
- `desktop/` contains the primary static web client.
- `mobile/` contains the companion progressive web application.
- `hub/` contains model orchestration, job management, voice storage, and audio utilities.
- `workers/` contains isolated model worker entrypoints.
- `watermark/` contains provenance training, embedding, detection, and evaluation code.
- `archive/` contains unsupported historical prototypes and development records.

## Requirements

- macOS on Apple Silicon
- Python 3.9 or later
- `ffmpeg` available on `PATH`
- Node.js and npm for Playwright tests

Install `ffmpeg` with Homebrew:

```bash
brew install ffmpeg
```

## Setup

```bash
git clone https://github.com/MrMystery8/tts-hub.git
cd tts-hub
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
```

## Run the System

Preferred command:

```bash
./run.sh
```

The application is then available at `http://localhost:7896`.

Manual equivalent:

```bash
.venv/bin/python3 app.py --port 7896
```

The desktop and mobile clients use the same local API. The mobile client is available at `http://localhost:7896/mobile/` and delegates synthesis to the host device.

## Tests and Checks

Run the principal Python tests:

```bash
python3 -m pytest \
  tests/test_generation_jobs.py \
  tests/test_generation_job_api.py \
  tests/test_voice_library.py \
  tests/test_checkpointing.py -q
```

Install Chromium and run the supported end-to-end suites:

```bash
npm run e2e:install
npx playwright test \
  tests/e2e/desktop_saved_phrases.spec.ts \
  tests/e2e/mobile_app.spec.ts \
  tests/e2e/ui_render_smoke.spec.ts
```

## Public Interfaces

- `GET /api/models` lists registered backends and UI defaults.
- `POST /api/generation-jobs` submits persistent asynchronous generation work.
- `GET /api/generation-jobs` returns generation history and job state.
- `GET`, `POST`, `PATCH`, and `DELETE /api/voices` manage the local voice library.
- `/api/watermark/*` provides watermark run discovery, embedding support, detection, and attribution.
- `POST /api/generate` remains available for backward-compatible synchronous generation.

## Documentation

- [Specification](docs/SPEC_SHEET.md)
- [Implementation summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Technical report](docs/TECHNICAL_REPORT_VNEXT_MULTICLASS.md)
- [Roadmap](docs/ROADMAP_AND_IMPROVEMENTS.md)
- [Mobile companion](docs/mobile.md)

Historical UI variants are retained under [archive/ui-prototypes](archive/ui-prototypes/README.md) for traceability. They are not part of the supported application or submission evidence.
