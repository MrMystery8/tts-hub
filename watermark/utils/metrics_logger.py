from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JSONLMetricsLogger:
    """
    Minimal, low-overhead metrics logger.

    Writes one JSON object per line to a file (JSONL) and flushes each write so
    external viewers (e.g. a dashboard) can read updates in near real-time.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLMetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

