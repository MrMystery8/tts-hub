#!/usr/bin/env python3
"""
Smoke test for the watermark integration (UI/server/service).

Runs:
1) Basic run discovery + run details
2) (Optional) FastAPI endpoints via TestClient (requires fastapi installed)
3) (Optional) Embed + detect via WatermarkService (requires torch/torchaudio/soundfile installed)

Usage:
  python3 scripts/smoke_watermark_integration.py
"""

from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path


def _make_silence_wav_bytes(*, seconds: float = 1.0, sr: int = 16000) -> bytes:
    n = int(max(1, round(seconds * sr)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    return buf.getvalue()


def _try_import(name: str):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def main() -> int:
    hub_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(hub_root))

    print("[smoke] hub_root:", hub_root)

    from hub.watermark_service import WatermarkService

    svc = WatermarkService(hub_root=hub_root)
    runs = svc.list_runs()
    print("[smoke] runs:", len(runs))
    default_run_id = svc.get_default_run_id()
    print("[smoke] default_run_id:", default_run_id)

    if not default_run_id:
        print("[smoke][FAIL] No watermark runs found. Need a run dir containing encoder.pt + decoder.pt.")
        return 2

    details = svc.get_run_details(run_id=None)
    print("[smoke] run_details.id:", details.get("id"))
    print("[smoke] run_details.status:", details.get("status"))
    if details.get("metrics"):
        print("[smoke] run_details.metrics:", details["metrics"])

    wav_bytes = _make_silence_wav_bytes(seconds=1.0, sr=16000)
    out_dir = hub_root / "outputs" / "_tmp_wm_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "silence.wav"
    wm_path = out_dir / "silence_wm.wav"
    raw_path.write_bytes(wav_bytes)

    ran_any = False

    # Optional: service-level embed + detect (requires torch stack)
    if _try_import("torch") and _try_import("torchaudio") and _try_import("soundfile"):
        print("[smoke] torch stack: OK -> running embed+detect")
        try:
            res_raw = svc.detect_from_audio_file(audio_path=raw_path, run_id=default_run_id, wm_threshold=0.8)
            print("[smoke] detect(raw):", res_raw)

            svc.embed_into_wav(
                input_wav_path=raw_path,
                output_wav_path=wm_path,
                tts_model_id="index-tts2",
                run_id=default_run_id,
            )
            print("[smoke] embedded wav bytes:", wm_path.stat().st_size)

            res_wm = svc.detect_from_audio_file(audio_path=wm_path, run_id=default_run_id, wm_threshold=0.8)
            print("[smoke] detect(embedded):", res_wm)
            ran_any = True
        except Exception as e:
            print("[smoke][FAIL] WatermarkService embed/detect failed:", repr(e))
            return 3
    else:
        print("[smoke] torch stack: missing -> skipping embed+detect")

    # Optional: API-level smoke via FastAPI TestClient
    if _try_import("fastapi"):
        if not _try_import("httpx"):
            print("[smoke] fastapi: OK but httpx missing -> skipping API tests")
        else:
            print("[smoke] fastapi: OK -> running API tests")
            try:
                from fastapi.testclient import TestClient
                import webui

                app = webui.create_app(hub_root=hub_root)
                client = TestClient(app)

                r = client.get("/api/watermark/runs")
                assert r.status_code == 200, r.text
                data = r.json()
                assert "runs" in data and "default_run_id" in data, data

                r = client.get("/api/watermark/run_details")
                assert r.status_code == 200, r.text

                files = {"audio": ("silence.wav", wav_bytes, "audio/wav")}
                r = client.post("/api/watermark/detect", files=files, data={"wm_threshold": "0.8"})
                assert r.status_code == 200, r.text
                print("[smoke] /api/watermark/detect:", r.json())
                ran_any = True
            except Exception as e:
                print("[smoke][FAIL] FastAPI endpoint smoke failed:", repr(e))
                return 4
    else:
        print("[smoke] fastapi: missing -> skipping API tests")

    if not ran_any:
        print("[smoke][FAIL] Skipped all runtime checks (missing fastapi and torch stack).")
        print("[smoke] Install deps and rerun:")
        print("  python3 -m pip install -r requirements.txt")
        print("  python3 -m pip install -e .")
        return 5

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
