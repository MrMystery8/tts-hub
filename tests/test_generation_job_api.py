import tempfile
import time
import unittest
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from hub.hub_manager import GenerateResult
from hub.voice_library import VoiceLibrary
from webui import create_app


def _wav_bytes() -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    return buf.getvalue()


class TestGenerationJobApi(unittest.TestCase):
    def test_mobile_assets_and_model_ui_metadata_are_served(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ui = root / "ui"
            static = root / "static"
            mobile = root / "mobile"
            ui.mkdir()
            static.mkdir()
            mobile.mkdir()
            (ui / "index.html").write_text("<div>test</div>", encoding="utf-8")
            (mobile / "index.html").write_text("<div>mobile</div>", encoding="utf-8")
            (mobile / "manifest.webmanifest").write_text('{"name":"TTS Hub"}', encoding="utf-8")

            client = TestClient(create_app(hub_root=root, ui_dir=ui, static_dir=static))
            self.assertEqual(client.get("/mobile/").status_code, 200)
            self.assertIn("mobile", client.get("/mobile/").text)
            self.assertEqual(client.get("/mobile/manifest.webmanifest").status_code, 200)

            models = client.get("/api/models").json()["models"]
            self.assertEqual(len(models), 7)
            for model in models:
                self.assertIn("id", model)
                self.assertIn("name", model)
                self.assertIn("description", model)
                self.assertIsInstance(model["capabilities"], dict)
                self.assertIsInstance(model["defaults"], dict)
            qwen = next(model for model in models if model["id"] == "qwen3-tts-mlx")
            self.assertTrue(qwen["defaults"]["autoTranscribe"])
            self.assertEqual(qwen["capabilities"]["reference"], "required")

    def test_job_lifecycle_and_persistent_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ui = root / "ui"
            static = root / "static"
            ui.mkdir()
            static.mkdir()
            (ui / "index.html").write_text("<div>test</div>", encoding="utf-8")

            def fake_generate(manager, *, model_id, request):
                output = manager.outputs_root / "fake.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(b"\0\0" * 1600)
                return GenerateResult(output_path=output, meta={"seconds": 0.1, "sr": 16000})

            with patch("hub.hub_manager.HubManager.generate", fake_generate):
                client = TestClient(create_app(hub_root=root, ui_dir=ui, static_dir=static))
                response = client.post(
                    "/api/generation-jobs",
                    data={
                        "model_id": "pocket-tts",
                        "text": "hello",
                        "output_format": "wav",
                        "request_snapshot": '{"modelId":"pocket-tts","text":"hello"}',
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["id"]
                self.assertEqual(response.json()["request"]["modelId"], "pocket-tts")
                self.assertEqual(response.json()["request"]["text"], "hello")

                deadline = time.time() + 3
                job = response.json()
                while time.time() < deadline and job["status"] not in {"completed", "failed"}:
                    time.sleep(0.02)
                    job = client.get(f"/api/generation-jobs/{job_id}").json()

                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["output"]["duration_s"], 0.1)
                self.assertEqual(client.get(f"/api/generation-jobs/{job_id}/audio").status_code, 200)
                self.assertEqual(client.get("/api/generation-jobs").json()["jobs"][0]["id"], job_id)
                self.assertEqual(client.delete(f"/api/generation-jobs/{job_id}").status_code, 200)
                self.assertEqual(client.get(f"/api/generation-jobs/{job_id}").status_code, 404)

    def test_generation_job_uses_saved_voice_transcript_when_prompt_text_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ui = root / "ui"
            static = root / "static"
            ui.mkdir()
            static.mkdir()
            (ui / "index.html").write_text("<div>test</div>", encoding="utf-8")
            voice = VoiceLibrary(hub_root=root).create_voice(
                name="Saved",
                input_bytes=_wav_bytes(),
                filename="sample.wav",
                prompt_text="saved reference transcript",
            )
            seen = {}

            def fake_generate(manager, *, model_id, request):
                seen["model_id"] = model_id
                seen["request"] = request
                output = manager.outputs_root / "qwen.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(_wav_bytes())
                return GenerateResult(output_path=output, meta={"seconds": 0.1, "sr": 16000})

            with patch("hub.hub_manager.HubManager.generate", fake_generate):
                client = TestClient(create_app(hub_root=root, ui_dir=ui, static_dir=static))
                response = client.post(
                    "/api/generation-jobs",
                    data={
                        "model_id": "qwen3-tts-mlx",
                        "text": "hello",
                        "voice_id": voice["id"],
                        "auto_transcribe": "1",
                        "output_format": "wav",
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["id"]

                deadline = time.time() + 3
                job = response.json()
                while time.time() < deadline and job["status"] not in {"completed", "failed"}:
                    time.sleep(0.02)
                    job = client.get(f"/api/generation-jobs/{job_id}").json()

                self.assertEqual(job["status"], "completed")
                self.assertEqual(seen["model_id"], "qwen3-tts-mlx")
                self.assertEqual(seen["request"]["fields"].get("prompt_text"), "saved reference transcript")

    def test_legacy_generate_route_still_returns_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ui = root / "ui"
            static = root / "static"
            ui.mkdir()
            static.mkdir()
            (ui / "index.html").write_text("<div>test</div>", encoding="utf-8")

            def fake_generate(manager, *, model_id, request):
                output = manager.outputs_root / "legacy.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(b"\0\0" * 1600)
                return GenerateResult(output_path=output, meta={"seconds": 0.1, "sr": 16000})

            with patch("hub.hub_manager.HubManager.generate", fake_generate):
                client = TestClient(create_app(hub_root=root, ui_dir=ui, static_dir=static))
                response = client.post(
                    "/api/generate",
                    data={
                        "model_id": "pocket-tts",
                        "text": "hello",
                        "output_format": "wav",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "audio/wav")
                self.assertIn("attachment; filename=", response.headers["content-disposition"])
                self.assertGreater(len(response.content), 0)

    def test_legacy_generate_uses_saved_voice_transcript_when_prompt_text_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ui = root / "ui"
            static = root / "static"
            ui.mkdir()
            static.mkdir()
            (ui / "index.html").write_text("<div>test</div>", encoding="utf-8")
            voice = VoiceLibrary(hub_root=root).create_voice(
                name="Saved",
                input_bytes=_wav_bytes(),
                filename="sample.wav",
                prompt_text="saved reference transcript",
            )
            seen = {}

            def fake_generate(manager, *, model_id, request):
                seen["model_id"] = model_id
                seen["request"] = request
                output = manager.outputs_root / "legacy-qwen.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(_wav_bytes())
                return GenerateResult(output_path=output, meta={"seconds": 0.1, "sr": 16000})

            with patch("hub.hub_manager.HubManager.generate", fake_generate):
                client = TestClient(create_app(hub_root=root, ui_dir=ui, static_dir=static))
                response = client.post(
                    "/api/generate",
                    data={
                        "model_id": "qwen3-tts-mlx",
                        "text": "hello",
                        "voice_id": voice["id"],
                        "auto_transcribe": "1",
                        "output_format": "wav",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(seen["model_id"], "qwen3-tts-mlx")
                self.assertEqual(seen["request"]["fields"].get("prompt_text"), "saved reference transcript")


if __name__ == "__main__":
    unittest.main()
