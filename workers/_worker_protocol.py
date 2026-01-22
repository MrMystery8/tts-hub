from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

# CRITICAL: Capture the REAL stdout before any library can pollute it.
# Libraries like IndexTTS print debug messages to stdout which corrupts
# the JSON protocol. We redirect sys.stdout to stderr so all library
# prints go there, and use _real_stdout for our JSON messages.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


def send(obj: dict[str, Any]) -> None:
    """Send JSON message to hub via the REAL stdout (not redirected)."""
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def recv() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


@dataclass(frozen=True)
class WorkerContext:
    hub_root: str | None = None
    model_id: str | None = None

