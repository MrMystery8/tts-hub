from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from webui import create_app


def ensure_prototype_build(hub_root: Path) -> Path:
    prototype_dir = Path(__file__).resolve().parent / "client"
    dist_dir = prototype_dir / "dist"
    index_path = dist_dir / "index.html"
    if index_path.exists():
        return dist_dir

    if not shutil.which("npm"):
        raise RuntimeError("npm is required to build this archived prototype, but it was not found on PATH.")

    subprocess.run(["npm", "install"], cwd=hub_root, check=True)
    subprocess.run(["npm", "exec", "vite", "build", "--", "--config", str(prototype_dir / "vite.config.ts")], cwd=hub_root, check=True)

    if not index_path.exists():
        raise RuntimeError("Prototype build completed but client/dist/index.html is still missing.")
    return dist_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archived TTS Hub React design prototype launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7892)
    args = parser.parse_args()

    hub_root = Path(__file__).resolve().parents[3]
    dist_dir = ensure_prototype_build(hub_root)
    app = create_app(hub_root=hub_root, ui_dir=dist_dir, static_dir=dist_dir)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
