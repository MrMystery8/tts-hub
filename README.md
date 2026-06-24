# TTS Hub

TTS Hub is a local Apple Silicon-focused audio generation hub that runs multiple TTS and voice-cloning backends behind one FastAPI service and one web UI.

## Current Runtime Layout

- Active frontend source: `frontend/`
- Active frontend launcher: `new_webui.py`
- Active backend/API app: `webui.py`
- Default launcher script: `run.sh`
- Legacy static UI kept in repo: `custom_ui/`

What is actually used today:
- `run.sh` starts `.venv/bin/python3 new_webui.py --port 7891`
- `new_webui.py` builds or serves the React app from `frontend/dist`
- `new_webui.py` then calls `webui.create_app(...)`
- `webui.py` exposes the API routes and serves the built frontend

So yes: the active UI is the React frontend in `frontend/`, served through `new_webui.py`. `custom_ui/` is legacy and is not the default path anymore.

## Project Structure

```text
tts-hub/
├── frontend/          # Active React UI source
├── custom_ui/         # Legacy static UI
├── hub/               # Core services and model/runtime management
├── workers/           # Model worker entrypoints
├── watermark/         # Watermarking and provenance tooling
├── docs/              # Project documentation
├── new_webui.py       # React UI launcher
├── webui.py           # FastAPI app and API routes
└── run.sh             # Default local launcher
```

## Requirements

- macOS on Apple Silicon is the intended environment
- `ffmpeg` on `PATH`
- `python3`
- `npm`

Install `ffmpeg` with Homebrew if needed:

```bash
brew install ffmpeg
```

## Setup

```bash
git clone <your-repo-url>
cd tts-hub
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
```

## Run The System

Preferred:

```bash
./run.sh
```

That starts the app on:

```text
http://localhost:7891
```

Manual equivalent:

```bash
.venv/bin/python3 new_webui.py --port 7891
```

Notes:
- `new_webui.py` will build the React frontend if `frontend/dist/index.html` is missing.
- The FastAPI app is mounted from `webui.py`.
- The UI talks to the backend on the same origin in production.

## Frontend Development

Run the backend:

```bash
.venv/bin/python3 webui.py --port 7891
```

In another terminal, run the React dev server:

```bash
npm run ui:dev
```

Vite runs on:

```text
http://localhost:5173
```

The Vite config proxies `/api` to `http://127.0.0.1:7891`.

## Tests And Checks

Python tests:

```bash
python3 -m pytest tests/test_generation_jobs.py tests/test_generation_job_api.py tests/test_voice_library.py -q
```

Frontend build:

```bash
npm run ui:build
```

Playwright smoke test:

```bash
npx playwright install chromium
npx playwright test tests/e2e/ui_render_smoke.spec.ts
```

## Current UI Behavior

The current React UI uses async generation jobs in addition to the legacy direct generate route:

- New async flow: `/api/generation-jobs`
- Backward-compatible direct route: `/api/generate`

The async flow is what the current React UI uses for queued, persistent, cancellable runs.

## Documentation

- [docs/SPEC_SHEET.md](docs/SPEC_SHEET.md)
- [docs/IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)
- [docs/ROADMAP_AND_IMPROVEMENTS.md](docs/ROADMAP_AND_IMPROVEMENTS.md)
- [docs/FYP_COMPLETION_TASK_PLAN.md](docs/FYP_COMPLETION_TASK_PLAN.md)
