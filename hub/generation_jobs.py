from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATES = {"completed", "failed", "cancelled"}


class JobCancelled(RuntimeError):
    pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


class GenerationJobService:
    """Serialized, persistent generation jobs for the web UI."""

    def __init__(
        self,
        *,
        root: Path,
        executor: Callable[[str, Path, dict[str, Any], Callable[[str], None], threading.Event], dict[str, Any]],
        cancel_active: Callable[[str], None],
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._executor = executor
        self._cancel_active = cancel_active
        self._queue: queue.Queue[str] = queue.Queue()
        self._requests: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._active_job_id: str | None = None
        self._repair_interrupted_jobs()
        self._thread = threading.Thread(target=self._run, name="generation-jobs", daemon=True)
        self._thread.start()

    def _job_dir(self, job_id: str) -> Path:
        normalized = (job_id or "").strip().lower()
        if len(normalized) != 32 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("invalid job_id")
        path = (self.root / normalized).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("invalid job_id")
        return path

    def _meta_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "metadata.json"

    def _load(self, job_id: str) -> dict[str, Any]:
        path = self._meta_path(job_id)
        if not path.exists():
            raise FileNotFoundError("generation job not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        metadata["updated_at"] = time.time()
        _atomic_write_json(self._meta_path(job_id), metadata)
        return metadata

    def _repair_interrupted_jobs(self) -> None:
        for path in self.root.glob("*/metadata.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if metadata.get("status") not in TERMINAL_STATES:
                    metadata["status"] = "failed"
                    metadata["phase"] = "failed"
                    metadata["error"] = "Generation was interrupted by an application restart."
                    metadata["completed_at"] = time.time()
                    metadata["updated_at"] = time.time()
                    _atomic_write_json(path, metadata)
            except Exception:
                continue

    def submit(self, request: dict[str, Any], files: dict[str, tuple[str, bytes]]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        staged_files: dict[str, str] = {}
        for field, (filename, content) in files.items():
            suffix = Path(filename).suffix or ".bin"
            staged = job_dir / f"{field}{suffix}"
            staged.write_bytes(content)
            staged_files[field] = str(staged)

        now = time.time()
        metadata = {
            "id": job_id,
            "status": "queued",
            "phase": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "request": request.get("snapshot", {}),
            "model_id": request.get("model_id"),
            "voice_id": request.get("voice_id"),
            "text": request.get("text"),
            "output_format": request.get("output_format"),
            "watermark_enabled": bool(request.get("watermark_enabled")),
            "watermark_run": request.get("watermark_run"),
            "output": None,
        }
        self._save(job_id, metadata)
        request = dict(request)
        request["staged_files"] = staged_files
        with self._lock:
            self._requests[job_id] = request
            self._cancel_events[job_id] = threading.Event()
        self._queue.put(job_id)
        return metadata

    def get(self, job_id: str) -> dict[str, Any]:
        return self._load(job_id)

    def list(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for path in self.root.glob("*/metadata.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        jobs.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
        return jobs

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            metadata = self._load(job_id)
            if metadata.get("status") in TERMINAL_STATES:
                return metadata
            event = self._cancel_events.get(job_id)
            if event:
                event.set()
            if self._active_job_id == job_id:
                model_id = str(metadata.get("model_id") or "")
                if model_id:
                    self._cancel_active(model_id)
            else:
                metadata["status"] = "cancelled"
                metadata["phase"] = "cancelled"
                metadata["completed_at"] = time.time()
                metadata = self._save(job_id, metadata)
            return metadata

    def delete(self, job_id: str) -> None:
        with self._lock:
            metadata = self._load(job_id)
            if metadata.get("status") not in TERMINAL_STATES:
                raise ValueError("active generation jobs cannot be deleted")
            shutil.rmtree(self._job_dir(job_id))
            self._requests.pop(job_id, None)
            self._cancel_events.pop(job_id, None)

    def audio_path(self, job_id: str) -> Path:
        metadata = self._load(job_id)
        if metadata.get("status") != "completed":
            raise ValueError("generation output is not available")
        relative = str((metadata.get("output") or {}).get("path") or "")
        path = (self._job_dir(job_id) / relative).resolve()
        if self._job_dir(job_id) not in path.parents or not path.exists():
            raise FileNotFoundError("generation output not found")
        return path

    def _cleanup_intermediates(self, job_id: str) -> None:
        try:
            metadata = self._load(job_id)
            keep = {"metadata.json"}
            output_path = str((metadata.get("output") or {}).get("path") or "")
            if output_path:
                keep.add(output_path)
            for path in self._job_dir(job_id).iterdir():
                if path.name not in keep:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink(missing_ok=True)
        except Exception:
            pass

    def _set_phase(self, job_id: str, phase: str) -> None:
        with self._lock:
            metadata = self._load(job_id)
            if metadata.get("status") == "cancelled":
                raise JobCancelled("Generation cancelled.")
            metadata["status"] = phase
            metadata["phase"] = phase
            if not metadata.get("started_at"):
                metadata["started_at"] = time.time()
            self._save(job_id, metadata)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    metadata = self._load(job_id)
                    if metadata.get("status") == "cancelled":
                        continue
                    request = self._requests[job_id]
                    cancel_event = self._cancel_events[job_id]
                    self._active_job_id = job_id
                self._set_phase(job_id, "preparing")
                result = self._executor(
                    job_id,
                    self._job_dir(job_id),
                    request,
                    lambda phase: self._set_phase(job_id, phase),
                    cancel_event,
                )
                if cancel_event.is_set():
                    raise JobCancelled("Generation cancelled.")
                with self._lock:
                    metadata = self._load(job_id)
                    metadata.update(result)
                    metadata["status"] = "completed"
                    metadata["phase"] = "completed"
                    metadata["completed_at"] = time.time()
                    self._save(job_id, metadata)
            except JobCancelled:
                with self._lock:
                    metadata = self._load(job_id)
                    metadata["status"] = "cancelled"
                    metadata["phase"] = "cancelled"
                    metadata["error"] = None
                    metadata["completed_at"] = time.time()
                    self._save(job_id, metadata)
            except Exception as exc:
                with self._lock:
                    metadata = self._load(job_id)
                    if self._cancel_events.get(job_id) and self._cancel_events[job_id].is_set():
                        metadata["status"] = "cancelled"
                        metadata["phase"] = "cancelled"
                        metadata["error"] = None
                    else:
                        metadata["status"] = "failed"
                        metadata["phase"] = "failed"
                        metadata["error"] = str(exc)
                    metadata["completed_at"] = time.time()
                    self._save(job_id, metadata)
            finally:
                with self._lock:
                    self._active_job_id = None
                    self._requests.pop(job_id, None)
                    self._cancel_events.pop(job_id, None)
                self._cleanup_intermediates(job_id)
                self._queue.task_done()
