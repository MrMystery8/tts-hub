import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from hub.hub_manager import GenerateResult
from webui import create_app


class TestGenerationJobApi(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
