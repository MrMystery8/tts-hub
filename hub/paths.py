from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRuntimePaths:
    model_id: str
    repo_root: Path
    python: Path
    pythonpath: list[Path]


def _pick_python(venv_dir: Path) -> Path:
    for candidate in ("python", "python3"):
        path = venv_dir / "bin" / candidate
        if path.exists():
            return path
    return venv_dir / "bin" / "python"


def resolve_model_runtime_paths(hub_root: Path, model_id: str) -> ModelRuntimePaths:
    """
    Best-effort resolver that assumes sibling repos exist next to `tts-hub/`.

    Users can override this later via config, but this should work for the current workspace.
    """
    workspace_root = hub_root.parent

    if model_id == "index-tts2":
        repo_root = workspace_root / "index-tts"
        python = _pick_python(repo_root / ".venv")
        pythonpath = [repo_root]
    elif model_id == "chatterbox-multilingual":
        repo_root = workspace_root / "chatterbox-multilingual"
        python = _pick_python(repo_root / ".venv")
        pythonpath = [repo_root / "chatterbox" / "src", repo_root / "chatterbox"]
    elif model_id == "f5-hindi-urdu":
        repo_root = workspace_root / "f5-hindi-urdu"
        python = _pick_python(repo_root / ".venv")
        pythonpath = [repo_root]
    elif model_id == "cosyvoice3-mlx":
        repo_root = workspace_root / "cosyvoice3-mlx"
        python = _pick_python(repo_root / ".venv")
        pythonpath = [repo_root]
    elif model_id == "pocket-tts":
        repo_root = workspace_root / "pocket-tts"
        python = _pick_python(repo_root / ".venv")
        pythonpath = [repo_root]
    elif model_id == "voxcpm-ane":
        repo_root = workspace_root / "voxcpm-ane"
        python = _pick_python(repo_root / ".venv")
        pythonpath = [repo_root]
    else:
        raise KeyError(f"Unknown model_id: {model_id}")

    return ModelRuntimePaths(model_id=model_id, repo_root=repo_root, python=python, pythonpath=pythonpath)

