from __future__ import annotations

import argparse
from pathlib import Path

from fastapi.responses import FileResponse, Response

from webui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTS Hub desktop application launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7896)
    args = parser.parse_args()

    hub_root = Path(__file__).resolve().parent
    ui_dir = hub_root / "desktop"
    app = create_app(hub_root=hub_root, ui_dir=ui_dir, static_dir=ui_dir)

    @app.get("/support.js", include_in_schema=False)
    def support_js() -> FileResponse:
        return FileResponse(ui_dir / "support.js", media_type="text/javascript")

    @app.get("/tour.js", include_in_schema=False)
    def tour_js() -> FileResponse:
        return FileResponse(ui_dir / "tour.js", media_type="text/javascript")

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
