import tempfile
import threading
import time
import unittest
from pathlib import Path

from hub.generation_jobs import GenerationJobService


class TestGenerationJobService(unittest.TestCase):
    def test_completed_job_persists_output_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "generations"

            def executor(job_id, job_dir, request, set_phase, cancel_event):
                set_phase("generating")
                output = job_dir / "output.wav"
                output.write_bytes(b"audio")
                return {
                    "worker_duration_ms": 12.5,
                    "output": {"path": output.name, "format": "wav", "filename": f"{job_id}.wav"},
                }

            service = GenerationJobService(root=root, executor=executor, cancel_active=lambda _model_id: None)
            job = service.submit(
                {
                    "model_id": "test-model",
                    "text": "hello",
                    "output_format": "wav",
                    "snapshot": {"text": "hello"},
                },
                {},
            )

            deadline = time.time() + 3
            while time.time() < deadline and service.get(job["id"])["status"] not in {"completed", "failed"}:
                time.sleep(0.02)

            completed = service.get(job["id"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["request"]["text"], "hello")
            self.assertEqual(service.audio_path(job["id"]).read_bytes(), b"audio")
            self.assertEqual(service.list()[0]["id"], job["id"])

    def test_active_job_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "generations"
            release = threading.Event()
            cancel_calls: list[str] = []

            def executor(job_id, job_dir, request, set_phase, cancel_event):
                set_phase("generating")
                while not cancel_event.is_set() and not release.wait(0.02):
                    pass
                raise RuntimeError("worker terminated")

            service = GenerationJobService(
                root=root,
                executor=executor,
                cancel_active=lambda model_id: cancel_calls.append(model_id),
            )
            job = service.submit({"model_id": "test-model", "text": "hello", "output_format": "wav"}, {})

            deadline = time.time() + 3
            while time.time() < deadline and service.get(job["id"])["status"] == "queued":
                time.sleep(0.02)
            service.cancel(job["id"])

            deadline = time.time() + 3
            while time.time() < deadline and service.get(job["id"])["status"] != "cancelled":
                time.sleep(0.02)

            self.assertEqual(service.get(job["id"])["status"], "cancelled")
            self.assertEqual(cancel_calls, ["test-model"])


if __name__ == "__main__":
    unittest.main()
