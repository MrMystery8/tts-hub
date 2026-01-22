from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerConfig:
    python: Path
    worker_script: Path
    env: dict[str, str]
    cwd: Path | None = None


class SubprocessWorker:
    def __init__(self, cfg: WorkerConfig):
        self._cfg = cfg
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        env = dict(os.environ)
        env.update(self._cfg.env)

        self._proc = subprocess.Popen(
            [str(self._cfg.python), "-u", str(self._cfg.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit (logs go to terminal)
            text=True,
            env=env,
            cwd=str(self._cfg.cwd) if self._cfg.cwd else None,
        )

        # handshake
        hello = self._read_line()
        if not hello.get("ok"):
            raise WorkerError(f"Worker failed to start: {hello}")

    def _read_line(self) -> dict[str, Any]:
        assert self._proc is not None
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise WorkerError("Worker exited or produced no output.")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise WorkerError(f"Invalid worker JSON: {e}: {line[:200]!r}") from e

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.start()
            assert self._proc is not None
            assert self._proc.stdin is not None

            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()

            resp = self._read_line()
            if not resp.get("ok"):
                raise WorkerError(resp.get("error", "Worker error"))
            return resp

    def shutdown(self) -> None:
        with self._lock:
            if not self._proc or self._proc.poll() is not None:
                return
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                _ = self._read_line()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass

    def is_alive(self) -> bool:
        """Check if the worker process is running."""
        return self._proc is not None and self._proc.poll() is None

