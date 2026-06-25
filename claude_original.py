from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTS Hub original Claude Design static reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7897)
    args = parser.parse_args()

    ui_dir = Path(__file__).resolve().parent / "claude_original"
    app = FastAPI(title="TTS Hub Claude Original Static Reference")
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(ui_dir / "index.html", media_type="text/html")

    @app.get("/support.js", include_in_schema=False)
    def support_js() -> FileResponse:
        return FileResponse(ui_dir / "support.js", media_type="text/javascript")

    @app.get("/thumbnail.webp", include_in_schema=False)
    def thumbnail() -> FileResponse:
        return FileResponse(ui_dir / "thumbnail.webp", media_type="image/webp")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
