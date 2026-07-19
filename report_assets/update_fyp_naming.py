#!/usr/bin/env python3
"""Apply the TTS Hub naming cleanup from Chapter 4 onward only.

The updater performs byte-level substitutions in ``word/document.xml`` and
replaces only the media object used by Figure 4.1. This keeps every byte of the
document XML before the Chapter 4 heading unchanged and avoids refreshing Word
fields in the front matter.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Ayaan Minhas-TP077859-APD3F2511CS(AI)-FYP Final Report (DRAFT).docx"
DOT = ROOT / "report_assets" / "diagrams" / "fig_4_1_system_architecture.dot"
PNG = ROOT / "report_assets" / "diagrams" / "fig_4_1_system_architecture.png"
DOCUMENT_XML = "word/document.xml"
FIGURE_MEDIA = "word/media/image61.png"
CHAPTER_MARKER = "CHAPTER 4: DESIGN AND IMPLEMENTATION"
FIGURE_SIZE = (1624, 1417)
LEGACY_RUNTIME = "claude" + "_exact"

REPLACEMENTS = (
    (
        "a local-first voice cloning hub for assistive communication, referred to throughout by its active runtime name, "
        + LEGACY_RUNTIME
        + ".",
        "a local-first voice cloning hub for assistive communication, formally presented as TTS Hub.",
    ),
    (
        "the " + LEGACY_RUNTIME + " web interface",
        "the TTS Hub desktop web interface (desktop/)",
    ),
    (
        "the desktop web interface (" + LEGACY_RUNTIME + ")",
        "the desktop web interface (desktop/)",
    ),
)


def _render_figure() -> None:
    with tempfile.TemporaryDirectory(prefix="tts-hub-figure-") as temp_dir:
        rendered = Path(temp_dir) / "figure.png"
        subprocess.run(
            ["dot", "-Tpng", "-Gdpi=144", str(DOT), "-o", str(rendered)],
            check=True,
        )
        with Image.open(rendered) as image:
            image.convert("RGB").resize(FIGURE_SIZE, Image.Resampling.LANCZOS).save(
                PNG,
                format="PNG",
                optimize=True,
            )


def _replace_after_chapter_four(xml: bytes) -> tuple[bytes, str]:
    marker = CHAPTER_MARKER.encode("utf-8")
    boundary = xml.find(marker)
    if boundary < 0:
        raise RuntimeError(f"Chapter marker not found: {CHAPTER_MARKER}")

    prefix = xml[:boundary]
    updated = xml
    for old, new in REPLACEMENTS:
        old_bytes, new_bytes = old.encode("utf-8"), new.encode("utf-8")
        old_count, new_count = updated.count(old_bytes), updated.count(new_bytes)
        if old_count == 1:
            if updated.find(old_bytes) < boundary:
                raise RuntimeError(f"Refusing to edit text before Chapter 4: {old}")
            updated = updated.replace(old_bytes, new_bytes, 1)
        elif old_count == 0 and new_count == 1:
            continue
        else:
            raise RuntimeError(
                f"Expected one old or one updated match for: {old} "
                f"(old={old_count}, new={new_count})"
            )

    if updated[:boundary] != prefix:
        raise RuntimeError("Document XML changed before the Chapter 4 boundary")
    return updated, hashlib.sha256(prefix).hexdigest()


def _rewrite_docx(updated_xml: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        prefix="tts-hub-report-", suffix=".docx", dir=ROOT, delete=False
    ) as handle:
        temporary = Path(handle.name)

    try:
        with zipfile.ZipFile(DOCX, "r") as source, zipfile.ZipFile(
            temporary, "w"
        ) as target:
            for item in source.infolist():
                if item.filename == DOCUMENT_XML:
                    payload = updated_xml
                elif item.filename == FIGURE_MEDIA:
                    payload = PNG.read_bytes()
                else:
                    payload = source.read(item.filename)
                target.writestr(item, payload)
        os.replace(temporary, DOCX)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    _render_figure()
    with zipfile.ZipFile(DOCX, "r") as archive:
        original_xml = archive.read(DOCUMENT_XML)
    updated_xml, prefix_hash = _replace_after_chapter_four(original_xml)
    _rewrite_docx(updated_xml)

    with zipfile.ZipFile(DOCX, "r") as archive:
        verified_xml = archive.read(DOCUMENT_XML)
        verified_figure = archive.read(FIGURE_MEDIA)
    boundary = verified_xml.find(CHAPTER_MARKER.encode("utf-8"))
    if hashlib.sha256(verified_xml[:boundary]).hexdigest() != prefix_hash:
        raise RuntimeError("Pre-Chapter-4 XML verification failed after saving")
    for old, _ in REPLACEMENTS:
        if old.encode("utf-8") in verified_xml[boundary:]:
            raise RuntimeError(f"Stale Chapter 4 naming remains: {old}")
    if hashlib.sha256(verified_figure).digest() != hashlib.sha256(PNG.read_bytes()).digest():
        raise RuntimeError("Figure 4.1 media verification failed")

    print(f"Updated: {DOCX}")
    print(f"Figure: {PNG} ({FIGURE_SIZE[0]}x{FIGURE_SIZE[1]})")
    print(f"Pre-Chapter-4 XML SHA-256: {prefix_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
