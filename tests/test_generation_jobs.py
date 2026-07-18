import tempfile
import threading
import time
import unittest
import json
from pathlib import Path

from hub.generation_jobs import GenerationJobService


class TestGenerationJobService(unittest.TestCase):
    def test_legacy_metadata_defaults_and_user_fields_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "generations"
            service = GenerationJobService(root=root, executor=lambda *_args: {}, cancel_active=lambda _model_id: None)
            job_id = "a" * 32
            job_dir = root / job_id
            job_dir.mkdir()
            (job_dir / "metadata.json").write_text(
                json.dumps({"id": job_id, "status": "completed", "created_at": 1, "updated_at": 1}),
                encoding="utf-8",
            )

            legacy = service.get(job_id)
            self.assertFalse(legacy["favorite"])
            self.assertIsNone(legacy["label"])
            self.assertIsNone(legacy["favorited_at"])

            starred = service.update_meta(job_id, favorite=True, label="  " + "x" * 100 + "  ")
            self.assertTrue(starred["favorite"])
            self.assertIsNotNone(starred["favorited_at"])
            self.assertEqual(starred["label"], "x" * 80)

            renamed = service.update_meta(job_id, clear_label=True)
            self.assertIsNone(renamed["label"])
            unstarred = service.update_meta(job_id, favorite=False)
            self.assertFalse(unstarred["favorite"])
            self.assertIsNone(unstarred["favorited_at"])
            self.assertEqual(service.list()[0]["id"], job_id)

            service.delete(job_id)
            self.assertFalse(job_dir.exists())

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

    def test_cancelled_job_does_not_block_next_job(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "generations"
            first_started = threading.Event()
            allow_first_exit = threading.Event()
            cancel_calls: list[str] = []

            def executor(job_id, job_dir, request, set_phase, cancel_event):
                set_phase("generating")
                if request.get("text") == "first":
                    first_started.set()
                    while not cancel_event.is_set():
                        allow_first_exit.wait(0.02)
                    raise RuntimeError("worker terminated")
                output = job_dir / "output.wav"
                output.write_bytes(b"audio")
                return {
                    "worker_duration_ms": 5.0,
                    "output": {"path": output.name, "format": "wav", "filename": f"{job_id}.wav"},
                }

            service = GenerationJobService(
                root=root,
                executor=executor,
                cancel_active=lambda model_id: cancel_calls.append(model_id),
            )
            first = service.submit({"model_id": "test-model", "text": "first", "output_format": "wav"}, {})
            deadline = time.time() + 3
            while time.time() < deadline and not first_started.is_set():
                time.sleep(0.02)

            service.cancel(first["id"])
            deadline = time.time() + 3
            while time.time() < deadline and service.get(first["id"])["status"] != "cancelled":
                time.sleep(0.02)

            second = service.submit({"model_id": "test-model", "text": "second", "output_format": "wav"}, {})
            deadline = time.time() + 3
            while time.time() < deadline and service.get(second["id"])["status"] not in {"completed", "failed"}:
                time.sleep(0.02)

            self.assertEqual(service.get(first["id"])["status"], "cancelled")
            self.assertEqual(service.get(second["id"])["status"], "completed")
            self.assertEqual(cancel_calls, ["test-model"])


if __name__ == "__main__":
    unittest.main()
