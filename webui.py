from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hub.audio_utils import FfmpegNotFoundError, ffmpeg_convert_output, ffmpeg_convert_to_wav, has_ffmpeg
from hub.generation_jobs import GenerationJobService, JobCancelled
from hub.hub_manager import HubManager
from hub.voice_library import VoiceLibrary
from hub.watermark_service import WatermarkService


MODEL_UI_DEFAULTS = {
    "index-tts2": {
        "capabilities": {"reference": "required", "transcript": "optional", "watermark": True, "emotion": True},
        "defaults": {"emoMode": "speaker", "emoAlpha": 0.65, "useRandom": False, "emoVector": "[0,0,0,0,0,0,0.45,0]", "emoText": "", "maxTextTokens": 120, "maxMelTokens": 1500, "fastMode": False, "doSample": True, "temperature": 0.8, "topP": 0.8, "topK": 30, "numBeams": 3, "repetitionPenalty": 10, "lengthPenalty": 0},
    },
    "qwen3-tts-mlx": {
        "capabilities": {"reference": "required", "transcript": "auto", "watermark": True},
        "defaults": {"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", "autoTranscribe": True, "language": "auto", "speed": 1, "temperature": 0.7, "maxTokens": 1200},
    },
    "chatterbox-multilingual": {
        "capabilities": {"reference": "optional", "transcript": "optional", "watermark": True},
        "defaults": {"language": "hi", "usePrompt": True, "cfgWeight": 0.5, "temperature": 0.8, "exaggeration": 0.5, "fastMode": False, "enableChunking": True, "maxChunkChars": 150, "crossfadeMs": 50, "enableDf": False, "enableNovasr": False},
    },
    "f5-hindi-urdu": {
        "capabilities": {"reference": "required", "transcript": "required", "watermark": False},
        "defaults": {"romanMode": True, "overridesEnabled": True, "overridesText": "", "crossFade": 0.15, "nfeStep": 32, "speed": 1, "removeSilence": False, "seed": -1},
    },
    "cosyvoice3-mlx": {
        "capabilities": {"reference": "required", "transcript": "mode-dependent", "watermark": False},
        "defaults": {"model": "8bit", "mode": "zero_shot", "language": "auto", "speed": 1, "instructText": ""},
    },
    "pocket-tts": {
        "capabilities": {"reference": "optional", "transcript": "optional", "watermark": False},
        "defaults": {"voice": "hf://kyutai/tts-voices/alba-mackenna/casual.wav", "temperature": 0.8, "lsdDecodeSteps": 8, "eosThreshold": 0.4, "noiseClamp": "", "truncatePrompt": False},
    },
    "voxcpm-ane": {
        "capabilities": {"reference": "conditional", "transcript": "required-with-reference", "watermark": False},
        "defaults": {"voice": "", "cfgValue": 2, "inferenceTimesteps": 10, "maxLength": 2048},
    },
}


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def create_app(*, hub_root: Path, ui_dir: Path | None = None, static_dir: Path | None = None) -> FastAPI:
    ui_dir = ui_dir or (hub_root / "desktop")
    if static_dir is None:
        static_dir = ui_dir / "static" if (ui_dir / "static").exists() else ui_dir

    manager = HubManager(hub_root)
    voices = VoiceLibrary(hub_root=hub_root)
    watermark = WatermarkService(hub_root=hub_root)

    def execute_generation_job(
        job_id: str,
        job_dir: Path,
        job_request: dict,
        set_phase,
        cancel_event: threading.Event,
    ) -> dict:
        model_id = str(job_request.get("model_id") or "")
        voice_id = str(job_request.get("voice_id") or "").strip() or None
        output_format = str(job_request.get("output_format") or "wav")
        fields = dict(job_request.get("fields") or {})
        staged_files = dict(job_request.get("staged_files") or {})

        def _hydrate_saved_voice_transcript() -> None:
            # Use the saved voice's transcript when the request didn't carry an
            # explicit one, so models that rely on a reference transcript use it
            # verbatim instead of re-transcribing (qwen auto-transcribe) or failing
            # (f5 / cosyvoice zero-shot / voxcpm). An explicit prompt_text wins.
            if voice_id and not str(fields.get("prompt_text") or "").strip():
                saved_text = str((voices.get_voice_meta(voice_id) or {}).get("prompt_text") or "").strip()
                if saved_text:
                    fields["prompt_text"] = saved_text

        prompt_audio_wav = None
        emo_audio_wav = None
        if voice_id:
            voices.ensure_audio_meta(voice_id)
            prompt_audio_wav = voices.get_voice_audio_path(voice_id)
            _hydrate_saved_voice_transcript()
        elif staged_files.get("prompt_audio"):
            prompt_audio_wav = job_dir / "prompt.wav"
            ffmpeg_convert_to_wav(input_path=Path(staged_files["prompt_audio"]), output_path=prompt_audio_wav)
        if staged_files.get("emo_audio"):
            emo_audio_wav = job_dir / "emo.wav"
            ffmpeg_convert_to_wav(input_path=Path(staged_files["emo_audio"]), output_path=emo_audio_wav)

        if model_id in {"index-tts2", "f5-hindi-urdu", "cosyvoice3-mlx", "qwen3-tts-mlx"} and not prompt_audio_wav:
            raise ValueError("prompt_audio or voice_id is required for this model")
        if cancel_event.is_set():
            raise JobCancelled("Generation cancelled.")

        model_request = {
            "text": str(job_request.get("text") or ""),
            "prompt_audio_path": str(prompt_audio_wav) if prompt_audio_wav else None,
            "emo_audio_path": str(emo_audio_wav) if emo_audio_wav else None,
            "fields": fields,
            "hub_root": str(manager.hub_root),
        }
        set_phase("generating")
        t0 = time.perf_counter()
        result = manager.generate(model_id=model_id, request=model_request)
        hub_worker_ms = (time.perf_counter() - t0) * 1000.0
        if cancel_event.is_set():
            raise JobCancelled("Generation cancelled.")

        final_path = result.output_path
        if not final_path.exists():
            raise RuntimeError(f"Worker returned missing output: {final_path}")

        if job_request.get("watermark_enabled"):
            set_phase("watermarking")
            wm_path = job_dir / "watermarked.wav"
            watermark.embed_into_wav(
                input_wav_path=final_path,
                output_wav_path=wm_path,
                tts_model_id=model_id,
                run_id=job_request.get("watermark_run"),
            )
            final_path = wm_path
        if cancel_event.is_set():
            raise JobCancelled("Generation cancelled.")

        output_path = job_dir / f"output.{output_format}"
        if output_format == "wav":
            shutil.copyfile(final_path, output_path)
        else:
            set_phase("converting")
            ffmpeg_convert_output(input_wav_path=final_path, output_path=output_path)

        meta = dict(result.meta or {})
        return {
            "worker_duration_ms": round(hub_worker_ms, 2),
            "output": {
                "path": output_path.name,
                "filename": f"{model_id}_{job_id}.{output_format}",
                "format": output_format,
                "duration_s": meta.get("seconds"),
                "sample_rate": meta.get("sr"),
            },
            "worker_meta": _json_safe(meta),
        }

    generation_jobs = GenerationJobService(
        root=hub_root / "outputs" / "generations",
        executor=execute_generation_job,
        cancel_active=manager.cancel_active_generation,
    )

    app = FastAPI(title="TTS Hub", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    mobile_dir = hub_root / "mobile"
    if mobile_dir.exists():
        app.mount("/mobile", StaticFiles(directory=str(mobile_dir), html=True), name="mobile")

    # Shared brand assets (marks, icon sources) — served to every UI from one place
    # so the desktop and mobile shells cannot drift apart. See brand/README.md.
    brand_dir = hub_root / "brand"
    if brand_dir.exists():
        app.mount("/brand", StaticFiles(directory=str(brand_dir)), name="brand")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (ui_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/api/models")
    def models():
        return {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    **MODEL_UI_DEFAULTS.get(m.id, {"capabilities": {}, "defaults": {}}),
                }
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
                    "has_transcript": v.has_transcript,
                    "compatible_models": v.compatible_models,
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

    @app.patch("/api/voices/{voice_id}")
    async def update_voice(voice_id: str, request: Request):
        content_type = request.headers.get("content-type", "")
        try:
            if content_type.startswith("multipart/form-data"):
                form = await request.form()
                kwargs: dict[str, object] = {}
                if "name" in form:
                    kwargs["name"] = str(form.get("name") or "")
                if "prompt_text" in form:
                    kwargs["prompt_text"] = str(form.get("prompt_text") or "")
                up = form.get("prompt_audio")
                if up is not None and getattr(up, "filename", None):
                    kwargs["input_bytes"] = up.file.read()
                    kwargs["filename"] = up.filename
                return voices.update_voice(voice_id, **kwargs)
            body = await request.body()
            if not body.strip():
                return JSONResponse({"error": "empty request body"}, status_code=400)
            data = json.loads(body)
            return voices.rename_voice(voice_id, str(data.get("name") or ""))
        except FileNotFoundError:
            return JSONResponse({"error": "voice not found"}, status_code=404)
        except FfmpegNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

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
        try:
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
        finally:
            shutil.rmtree(uploads_root, ignore_errors=True)

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

    @app.get("/api/generation-jobs")
    def list_generation_jobs():
        return {"jobs": generation_jobs.list()}

    @app.post("/api/generation-jobs", status_code=202)
    async def create_generation_job(request: Request):
        form = await request.form()
        model_id = str(form.get("model_id") or "").strip()
        text = str(form.get("text") or "").strip()
        voice_id = str(form.get("voice_id") or "").strip() or None
        output_format = str(form.get("output_format") or "wav").strip().lower()
        if not model_id:
            return JSONResponse({"error": "model_id is required"}, status_code=400)
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        if output_format not in {"wav", "mp3", "flac"}:
            return JSONResponse({"error": "output_format must be wav|mp3|flac"}, status_code=400)

        files: dict[str, tuple[str, bytes]] = {}
        for field in ("prompt_audio", "emo_audio"):
            upload = form.get(field)
            if upload is not None:
                files[field] = (getattr(upload, "filename", None) or f"{field}.bin", upload.file.read())

        raw_snapshot = str(form.get("request_snapshot") or "").strip()
        try:
            snapshot = json.loads(raw_snapshot) if raw_snapshot else {}
        except json.JSONDecodeError:
            return JSONResponse({"error": "request_snapshot must be valid JSON"}, status_code=400)

        job_request = {
            "model_id": model_id,
            "text": text,
            "voice_id": voice_id,
            "output_format": output_format,
            "watermark_enabled": str(form.get("watermark") or "").strip().lower() in {"1", "true", "yes", "on"},
            "watermark_run": str(form.get("watermark_run") or "").strip() or None,
            "fields": {
                key: str(value)
                for key, value in form.items()
                if key not in {"prompt_audio", "emo_audio", "request_snapshot"}
            },
            "snapshot": snapshot,
        }
        try:
            return generation_jobs.submit(job_request, files)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/generation-jobs/{job_id}/audio")
    def get_generation_job_audio(job_id: str):
        try:
            metadata = generation_jobs.get(job_id)
            path = generation_jobs.audio_path(job_id)
            output = metadata.get("output") or {}
            media_type = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "flac": "audio/flac",
            }.get(output.get("format"), "application/octet-stream")
            return FileResponse(str(path), media_type=media_type, filename=output.get("filename") or path.name)
        except FileNotFoundError:
            return JSONResponse({"error": "generation job not found"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=409)

    @app.get("/api/generation-jobs/{job_id}")
    def get_generation_job(job_id: str):
        try:
            return generation_jobs.get(job_id)
        except (FileNotFoundError, ValueError):
            return JSONResponse({"error": "generation job not found"}, status_code=404)

    @app.patch("/api/generation-jobs/{job_id}")
    async def update_generation_job(job_id: str, request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "body must be valid JSON"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

        allowed_fields = {"favorite", "label"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            return JSONResponse(
                {"error": f"unsupported field(s): {', '.join(unknown_fields)}"},
                status_code=400,
            )
        if not payload:
            return JSONResponse({"error": "provide favorite and/or label"}, status_code=400)

        favorite = payload.get("favorite")
        if favorite is not None and not isinstance(favorite, bool):
            return JSONResponse({"error": "favorite must be a boolean"}, status_code=400)

        label = payload.get("label")
        clear_label = "label" in payload and label is None
        if label is not None and not isinstance(label, str):
            return JSONResponse({"error": "label must be a string or null"}, status_code=400)

        try:
            return generation_jobs.update_meta(
                job_id,
                favorite=favorite,
                label=label,
                clear_label=clear_label,
            )
        except (FileNotFoundError, ValueError):
            return JSONResponse({"error": "generation job not found"}, status_code=404)

    @app.post("/api/generation-jobs/{job_id}/cancel")
    def cancel_generation_job(job_id: str):
        try:
            return generation_jobs.cancel(job_id)
        except (FileNotFoundError, ValueError):
            return JSONResponse({"error": "generation job not found"}, status_code=404)

    @app.delete("/api/generation-jobs/{job_id}")
    def delete_generation_job(job_id: str):
        try:
            generation_jobs.delete(job_id)
            return {"ok": True}
        except FileNotFoundError:
            return JSONResponse({"error": "generation job not found"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=409)

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
        if voice_id and not str(model_request["fields"].get("prompt_text") or "").strip():
            try:
                saved_text = str((voices.get_voice_meta(voice_id) or {}).get("prompt_text") or "").strip()
                if saved_text:
                    model_request["fields"]["prompt_text"] = saved_text
            except FileNotFoundError:
                return JSONResponse({"error": "voice_id not found"}, status_code=404)

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
        description="TTS Hub local speech synthesis server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7896)
    args = parser.parse_args()

    hub_root = Path(__file__).resolve().parent
    app = create_app(hub_root=hub_root)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
