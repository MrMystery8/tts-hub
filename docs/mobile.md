# TTS Hub Mobile (phone → laptop delegation)

A touch-optimized PWA served by the existing FastAPI app at `/mobile`.
It uses the same API as the desktop UI, including the persistent
generation-job queue — submit a job from your phone, lock it, come back
later and the finished audio is waiting.

## What it does

- **Generate** — pick model + saved voice, type text, choose wav/mp3/flac,
  optional watermark, submit. Jobs run asynchronously on the laptop. Reference
  audio can come from a saved voice, phone upload, or an in-browser recording.
- **Jobs** — live status (queued → generating → completed) with auto-refresh,
  inline playback, download, cancel, delete. Active-job count badge on the tab.
- **Voices** — browse and preview the voice library, set a voice for
  generation, add a new voice (your phone's mic works via the file picker).
- **Model options** — the Options sheet exposes the same worker parameters as
  the desktop Generate screen, grouped into Basics and Advanced sections.

## Reference modes and validation

- **Saved** uses a voice from the shared library. If it has a transcript, the
  server injects that transcript automatically for F5, CosyVoice, Qwen, and
  VoxCPM.
- **Upload** stages an audio file from the phone. Add its verbatim transcript
  when required, and optionally save the staged clip to the voice library.
- **Record** captures microphone audio in the browser and supports previewing
  and saving it before generation.
- **None** is available only for models that can run without reference audio.

The mobile UI validates model requirements before queueing, while the backend
remains authoritative and returns any worker-specific error to the same screen.
IndexTTS2 emotion-reference mode also requires a separate emotion clip.

## Running

Nothing extra to run — `run.sh` (or `claude_exact.py`) serves it automatically:

- Desktop UI: `http://localhost:7896/`
- Mobile UI:  `http://localhost:7896/mobile/`

## Reaching it from your phone (Tailscale)

1. Install Tailscale on the laptop: `brew install --cask tailscale`,
   open it, sign in (Google/GitHub/etc.).
2. Install the Tailscale app on your phone and sign in with the same account.
3. Find the laptop's Tailscale name/IP (`tailscale status` or the app UI).
4. On the phone, open `http://<laptop-tailscale-name>:7896/mobile/`.

Optional but recommended — proper HTTPS via Tailscale (needed for full PWA
install prompts and required by browsers for microphone recording outside
`localhost`):

```bash
tailscale serve --bg 7896
```

Then use `https://<laptop-name>.<tailnet>.ts.net/mobile/` on the phone.

### Install to home screen

- **Android (Chrome):** menu → "Add to Home screen" / "Install app".
- **iOS (Safari):** share sheet → "Add to Home Screen".

It opens fullscreen with its own icon, like a native app.

## Notes

- The laptop must be awake for jobs to process. To keep the server running
  with the lid closed on AC power: `sudo pmset -c disablesleep 1`
  (or use `caffeinate -s .venv/bin/python3 claude_exact.py --port 7896`).
- Jobs are stored under `outputs/generations/` and survive server restarts;
  anything interrupted mid-run is marked failed on startup.
- Everything stays inside your private tailnet — no ports exposed to the
  internet.

## QA checklist

Before committing mobile changes:

1. Run `node --check mobile/app.js` and `node --check mobile/sw.js`.
2. Run `.venv/bin/python -m pytest -q`.
3. Run `npx playwright test tests/e2e/mobile_app.spec.ts`.
4. Queue a short Pocket TTS run from `/mobile/`, wait for completion, play it,
   download it, reload the page, and confirm the completed run is restored.
5. On a phone over Tailscale HTTPS, verify Add to Home Screen, microphone
   permission, recording preview/upload, reopening after backgrounding, and
   safe-area spacing around the header and bottom tabs.
