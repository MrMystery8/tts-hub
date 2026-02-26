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
from hub.voice_library import VoiceLibrary
from hub.watermark_service import WatermarkService


def create_app(*, hub_root: Path) -> FastAPI:
    ui_dir = hub_root / "custom_ui"
    static_dir = ui_dir / "static"

    manager = HubManager(hub_root)
    voices = VoiceLibrary(hub_root=hub_root)
    watermark = WatermarkService(hub_root=hub_root)

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

    @app.get("/api/voices")
    def list_voices():
        return {
            "voices": [
                {
                    "id": v.id,
                    "name": v.name,
                    "created_at": v.created_at,
                    "duration_s": v.duration_s,
                    "has_caches": v.has_caches,
                }
                for v in voices.list_voices()
            ]
        }

    @app.post("/api/voices")
    async def create_voice(request: Request):
        form = await request.form()
        name = str(form.get("name") or "").strip()
        prompt_text = str(form.get("prompt_text") or "").strip() or None
        up = form.get("prompt_audio")
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        if up is None:
            return JSONResponse({"error": "prompt_audio is required"}, status_code=400)
        filename = getattr(up, "filename", None) or "prompt.bin"
        try:
            meta = voices.create_voice(
                name=name,
                input_bytes=up.file.read(),
                filename=filename,
                prompt_text=prompt_text,
            )
        except FfmpegNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return meta

    @app.get("/api/voices/{voice_id}")
    def get_voice(voice_id: str):
        try:
            meta = voices.get_voice_meta(voice_id)
        except FileNotFoundError:
            return JSONResponse({"error": "voice not found"}, status_code=404)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return meta

    @app.get("/api/voices/{voice_id}/audio")
    def get_voice_audio(voice_id: str):
        try:
            wav_path = voices.get_voice_audio_path(voice_id)
        except FileNotFoundError:
            return JSONResponse({"error": "voice not found"}, status_code=404)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return FileResponse(str(wav_path), media_type="audio/wav", filename="prompt.wav")

    @app.delete("/api/voices/{voice_id}")
    def delete_voice(voice_id: str):
        try:
            voices.delete_voice(voice_id)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True}

    @app.get("/api/watermark/runs")
    def watermark_runs():
        runs = watermark.list_runs()
        return {
            "default_run_id": watermark.get_default_run_id(),
            "runs": [
                {
                    "id": r.id,
                    "label": r.label,
                    "status": r.status,
                    "updated_at": r.updated_at,
                }
                for r in runs
            ]
        }

    @app.get("/api/watermark/run_details")
    def watermark_run_details(run_id: str | None = None):
        try:
            details = watermark.get_run_details(run_id=run_id)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return details

    @app.post("/api/watermark/detect")
    async def watermark_detect(request: Request):
        form = await request.form()

        task_id = uuid.uuid4().hex
        uploads_root = manager.uploads_root / f"wm_detect_{task_id}"
        uploads_root.mkdir(parents=True, exist_ok=True)

        up = form.get("audio")
        if up is None:
            return JSONResponse({"error": "audio is required"}, status_code=400)

        filename = getattr(up, "filename", None) or "audio.bin"
        suffix = Path(filename).suffix or ".bin"
        input_path = uploads_root / f"input{suffix}"
        input_path.write_bytes(up.file.read())

        run_id = str(form.get("watermark_run") or "").strip() or None
        try:
            wm_threshold = float(form.get("wm_threshold") or 0.35)
        except Exception:
            wm_threshold = 0.35

        # Prefer ffmpeg conversion so we can handle mp3/m4a consistently.
        detect_path = input_path
        if has_ffmpeg():
            try:
                wav_path = uploads_root / "input_16k.wav"
                ffmpeg_convert_to_wav(input_path=input_path, output_path=wav_path, sample_rate=16000, channels=1)
                detect_path = wav_path
            except Exception as e:
                return JSONResponse({"error": f"ffmpeg conversion failed: {e}"}, status_code=500)

        try:
            result = watermark.detect_from_audio_file(audio_path=detect_path, run_id=run_id, wm_threshold=wm_threshold)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        model_name = None
        if result.get("tts_model_id"):
            spec = next((m for m in manager.list_models() if m.id == result["tts_model_id"]), None)
            model_name = spec.name if spec else result["tts_model_id"]

        return {
            "detected": bool(result.get("detected")),
            "wm_prob": float(result.get("wm_prob", 0.0)),
            "model": {
                "id": result.get("pred_attr_id"),
                "name": model_name,
                "tts_model_id": result.get("tts_model_id"),
            },
            "run": {"id": result.get("run_id")},
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
        voice_id = str(form.get("voice_id") or "").strip() or None
        output_format = str(form.get("output_format") or "wav").strip().lower()
        watermark_enabled = str(form.get("watermark") or "").strip().lower() in {"1", "true", "yes", "on"}
        watermark_run = str(form.get("watermark_run") or "").strip() or None

        if not model_id:
            return JSONResponse({"error": "model_id is required"}, status_code=400)
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        if output_format not in {"wav", "mp3", "flac"}:
            return JSONResponse({"error": "output_format must be wav|mp3|flac"}, status_code=400)

        def _save_upload(field: str) -> Path | None:
            up = form.get(field)
            if up is None:
                return None
            if uploads_root is None:
                raise RuntimeError("uploads_root not initialized")
            filename = getattr(up, "filename", None) or f"{field}.bin"
            suffix = Path(filename).suffix or ".bin"
            out_path = uploads_root / f"{field}{suffix}"
            out_path.write_bytes(up.file.read())
            return out_path

        uploads_root: Path | None = None
        need_uploads_root = (voice_id is None and form.get("prompt_audio") is not None) or (form.get("emo_audio") is not None)
        if need_uploads_root:
            task_id = uuid.uuid4().hex
            uploads_root = manager.uploads_root / task_id
            uploads_root.mkdir(parents=True, exist_ok=True)

        prompt_audio_in = None
        if voice_id is None:
            prompt_audio_in = _save_upload("prompt_audio")
        emo_audio_in = _save_upload("emo_audio")

        # Convert uploaded audio to wav (keep worker dependencies smaller / consistent).
        prompt_audio_wav = None
        emo_audio_wav = None
        try:
            if voice_id:
                # Use stable on-disk voice reference; do not stage prompt upload.
                voices.ensure_audio_meta(voice_id)
                prompt_audio_wav = voices.get_voice_audio_path(voice_id)
            elif prompt_audio_in:
                prompt_audio_wav = uploads_root / "prompt.wav"
                ffmpeg_convert_to_wav(input_path=prompt_audio_in, output_path=prompt_audio_wav)
            if emo_audio_in:
                emo_audio_wav = uploads_root / "emo.wav"
                ffmpeg_convert_to_wav(input_path=emo_audio_in, output_path=emo_audio_wav)
        except FfmpegNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        except FileNotFoundError:
            return JSONResponse({"error": "voice_id not found"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        model_request = {
            "text": text,
            "prompt_audio_path": str(prompt_audio_wav) if prompt_audio_wav else None,
            "emo_audio_path": str(emo_audio_wav) if emo_audio_wav else None,
            # pass-through fields (model workers decide what they need)
            "fields": {k: str(v) for k, v in form.items() if k not in {"prompt_audio", "emo_audio"}},
            "hub_root": str(manager.hub_root),
        }

        if model_id in {"index-tts2", "f5-hindi-urdu", "cosyvoice3-mlx", "qwen3-tts-mlx"} and not prompt_audio_wav:
            return JSONResponse({"error": "prompt_audio or voice_id is required for this model"}, status_code=400)

        t0 = time.perf_counter()
        try:
            result = manager.generate(model_id=model_id, request=model_request)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        hub_worker_ms = (time.perf_counter() - t0) * 1000.0

        out_path = result.output_path
        if not out_path.exists():
            return JSONResponse({"error": f"Worker returned missing output: {out_path}"}, status_code=500)

        final_path = out_path

        # Optional watermarking (post-processing)
        if watermark_enabled:
            if out_path.suffix.lower() != ".wav":
                return JSONResponse({"error": "Watermarking requires WAV output from the worker"}, status_code=500)
            wm_path = out_path.with_name(f"{out_path.stem}_wm.wav")
            try:
                watermark.embed_into_wav(
                    input_wav_path=out_path,
                    output_wav_path=wm_path,
                    tts_model_id=model_id,
                    run_id=watermark_run,
                )
            except Exception as e:
                return JSONResponse({"error": f"watermarking failed: {e}"}, status_code=500)
            final_path = wm_path

        # Optional conversion
        if output_format != "wav":
            final_converted = final_path.with_suffix(f".{output_format}")
            try:
                ffmpeg_convert_output(input_wav_path=final_path, output_path=final_converted)
            except Exception as e:
                return JSONResponse({"error": f"ffmpeg output conversion failed: {e}"}, status_code=500)
            final_path = final_converted

        media_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
        }[output_format]

        download_id = out_path.stem
        if watermark_enabled:
            download_id = f"{download_id}_wm"

        headers: dict[str, str] = {
            "X-Hub-Worker-MS": f"{hub_worker_ms:.2f}",
        }
        meta = result.meta or {}
        # Surface a few cache-related fields for quick benchmarking/debugging.
        for k in (
            "speaker_cache_status",
            "speaker_cache_load_ms",
            "speaker_cache_compute_ms",
            "speaker_cache_save_ms",
            "speaker_cache_voice_id",
            "conds_cache_status",
            "conds_cache_load_ms",
            "conds_cache_prepare_ms",
            "conds_cache_save_ms",
            "conds_cache_voice_id",
            "ref_transcript_status",
            "ref_audio_cache_status",
            "model_id",
            "seconds",
            "sr",
        ):
            if k in meta and meta[k] is not None:
                headers[f"X-{k.replace('_', '-')}"] = str(meta[k])

        return FileResponse(
            str(final_path),
            media_type=media_type,
            filename=f"{model_id}_{download_id}.{output_format}",
            headers=headers,
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
