from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from webui import create_app


def ensure_claude_build(hub_root: Path) -> Path:
    claude_dir = hub_root / "claude_ui"
    dist_dir = claude_dir / "dist"
    index_path = dist_dir / "index.html"
    if index_path.exists():
        return dist_dir

    if not shutil.which("npm"):
        raise RuntimeError("npm is required to build the Claude UI, but it was not found on PATH.")

    subprocess.run(["npm", "install"], cwd=hub_root, check=True)
    subprocess.run(["npm", "run", "claude:build"], cwd=hub_root, check=True)

    if not index_path.exists():
        raise RuntimeError("Claude UI build completed but claude_ui/dist/index.html is still missing.")
    return dist_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTS Hub Claude UI launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7892)
    args = parser.parse_args()

    hub_root = Path(__file__).resolve().parent
    dist_dir = ensure_claude_build(hub_root)
    app = create_app(hub_root=hub_root, ui_dir=dist_dir, static_dir=dist_dir)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
