from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hub.audio_utils import FfmpegNotFoundError, ffmpeg_convert_output, ffmpeg_convert_to_wav, has_ffmpeg
from hub.hub_manager import HubManager


def create_app(*, hub_root: Path) -> FastAPI:
    ui_dir = hub_root / "custom_ui"
    static_dir = ui_dir / "static"

    manager = HubManager(hub_root)

    app = FastAPI(title="TTS Hub", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (ui_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/api/models")
    def models():
        return {
            "models": [
                {"id": m.id, "name": m.name, "description": m.description}
                for m in manager.list_models()
            ]
        }

    @app.get("/api/info")
    def info():
        return {
            "ffmpeg": {"available": has_ffmpeg()},
            "time": int(time.time()),
        }

    @app.get("/api/status")
    def status(model_id: str = None):
        """Get status of one or all models."""
        statuses = {}
        for spec in manager.list_models():
            mid = spec.id
            if model_id and mid != model_id:
                continue
            worker = manager._workers.get(mid)
            is_loaded = worker is not None and worker.is_alive()
            gen_stats = manager.get_generation_stats(mid)
            statuses[mid] = {
                "loaded": is_loaded,
                "device": gen_stats.get("device", "unknown"),
                "last_generation_time": gen_stats.get("last_time"),
                "last_generation_duration_ms": gen_stats.get("last_duration_ms"),
                "total_generations": gen_stats.get("total", 0),
            }
        if model_id:
            return statuses.get(model_id, {"error": "Model not found"})
        return {"models": statuses}

    @app.post("/api/unload")
    async def unload(request: Request):
        form = await request.form()
        model_id = str(form.get("model_id") or "").strip()
        if not model_id:
            return JSONResponse({"error": "model_id is required"}, status_code=400)
        try:
            manager.unload(model_id)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return {"ok": True}

    @app.post("/api/generate")
    async def generate(request: Request):
        form = await request.form()

        model_id = str(form.get("model_id") or "").strip()
        text = str(form.get("text") or "").strip()
        output_format = str(form.get("output_format") or "wav").strip().lower()

        if not model_id:
            return JSONResponse({"error": "model_id is required"}, status_code=400)
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        if output_format not in {"wav", "mp3", "flac"}:
            return JSONResponse({"error": "output_format must be wav|mp3|flac"}, status_code=400)

        task_id = uuid.uuid4().hex
        uploads_root = manager.uploads_root / task_id
        uploads_root.mkdir(parents=True, exist_ok=True)

        def _save_upload(field: str) -> Path | None:
            up = form.get(field)
            if up is None:
                return None
            filename = getattr(up, "filename", None) or f"{field}.bin"
            suffix = Path(filename).suffix or ".bin"
            out_path = uploads_root / f"{field}{suffix}"
            out_path.write_bytes(up.file.read())
            return out_path

        prompt_audio_in = _save_upload("prompt_audio")
        emo_audio_in = _save_upload("emo_audio")

        # Convert uploaded audio to wav (keep worker dependencies smaller / consistent).
        prompt_audio_wav = None
        emo_audio_wav = None
        try:
            if prompt_audio_in:
                prompt_audio_wav = uploads_root / "prompt.wav"
                ffmpeg_convert_to_wav(input_path=prompt_audio_in, output_path=prompt_audio_wav)
            if emo_audio_in:
                emo_audio_wav = uploads_root / "emo.wav"
                ffmpeg_convert_to_wav(input_path=emo_audio_in, output_path=emo_audio_wav)
        except FfmpegNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        model_request = {
            "text": text,
            "prompt_audio_path": str(prompt_audio_wav) if prompt_audio_wav else None,
            "emo_audio_path": str(emo_audio_wav) if emo_audio_wav else None,
            # pass-through fields (model workers decide what they need)
            "fields": {k: str(v) for k, v in form.items() if k not in {"prompt_audio", "emo_audio"}},
            "hub_root": str(manager.hub_root),
        }

        try:
            result = manager.generate(model_id=model_id, request=model_request)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        out_path = result.output_path
        if not out_path.exists():
            return JSONResponse({"error": f"Worker returned missing output: {out_path}"}, status_code=500)

        # Optional conversion
        final_path = out_path
        if output_format != "wav":
            final_path = out_path.with_suffix(f".{output_format}")
            try:
                ffmpeg_convert_output(input_wav_path=out_path, output_path=final_path)
            except Exception as e:
                return JSONResponse({"error": f"ffmpeg output conversion failed: {e}"}, status_code=500)

        media_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
        }[output_format]

        return FileResponse(
            str(final_path),
            media_type=media_type,
            filename=f"{model_id}_{task_id}.{output_format}",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified TTS Hub (MLX/MPS/ANE) Web UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7891)
    args = parser.parse_args()

    hub_root = Path(__file__).resolve().parent
    app = create_app(hub_root=hub_root)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
