from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_registry import ModelSpec, get_model_specs
from .paths import resolve_model_runtime_paths
from .subprocess_worker import SubprocessWorker, WorkerConfig


@dataclass(frozen=True)
class GenerateResult:
    output_path: Path
    meta: dict[str, Any]


@dataclass
class GenerationStats:
    """Tracks generation statistics for a model."""
    total: int = 0
    last_time: float | None = None
    last_duration_ms: float | None = None
    device: str = "unknown"


class HubManager:
    def __init__(self, hub_root: Path):
        self.hub_root = hub_root
        self.outputs_root = hub_root / "outputs"
        self.uploads_root = self.outputs_root / "uploads"
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self._specs: dict[str, ModelSpec] = {s.id: s for s in get_model_specs()}
        self._workers: dict[str, SubprocessWorker] = {}
        self._stats: dict[str, GenerationStats] = {}

    def list_models(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get_generation_stats(self, model_id: str) -> dict[str, Any]:
        """Get generation statistics for a model."""
        stats = self._stats.get(model_id, GenerationStats())
        return {
            "total": stats.total,
            "last_time": stats.last_time,
            "last_duration_ms": stats.last_duration_ms,
            "device": stats.device,
        }

    def _get_worker(self, model_id: str) -> SubprocessWorker:
        if model_id in self._workers:
            return self._workers[model_id]

        spec = self._specs[model_id]
        runtime = resolve_model_runtime_paths(self.hub_root, model_id)
        worker_script = self.hub_root / "workers" / spec.worker_entry

        pythonpath = os_pathsep_join([str(p) for p in runtime.pythonpath if p.exists()])
        env = {}
        if pythonpath:
            env["PYTHONPATH"] = pythonpath

        cfg = WorkerConfig(
            python=runtime.python,
            worker_script=worker_script,
            env=env,
            cwd=runtime.repo_root,
        )
        worker = SubprocessWorker(cfg)
        self._workers[model_id] = worker
        
        # Determine device based on model
        device = "cpu"
        if "mlx" in model_id:
            device = "mlx"
        elif "ane" in model_id:
            device = "ane"
        else:
            device = "mps"  # Default to MPS for PyTorch models on Apple Silicon
        
        if model_id not in self._stats:
            self._stats[model_id] = GenerationStats(device=device)
        else:
            self._stats[model_id].device = device
            
        return worker

    def unload(self, model_id: str) -> None:
        worker = self._workers.get(model_id)
        if not worker:
            return
        try:
            worker.request({"cmd": "unload"})
        finally:
            worker.shutdown()
            self._workers.pop(model_id, None)

    def generate(self, *, model_id: str, request: dict[str, Any]) -> GenerateResult:
        worker = self._get_worker(model_id)
        req_id = uuid.uuid4().hex
        request = dict(request)
        request["request_id"] = req_id
        
        start_time = time.time()
        resp = worker.request({"cmd": "gen", "model_id": model_id, "request": request})
        duration_ms = (time.time() - start_time) * 1000
        
        # Update stats
        if model_id not in self._stats:
            self._stats[model_id] = GenerationStats()
        stats = self._stats[model_id]
        stats.total += 1
        stats.last_time = start_time
        stats.last_duration_ms = duration_ms
        
        out_path = Path(resp["result"]["output_path"])
        meta = dict(resp["result"].get("meta", {}))
        return GenerateResult(output_path=out_path, meta=meta)


def os_pathsep_join(items: list[str]) -> str:
    import os

    sep = os.pathsep
    return sep.join([x for x in items if x])

