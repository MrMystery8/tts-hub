#!/usr/bin/env python3
"""Build labelled contact sheets for whole-document visual QA."""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "render_dir",
    nargs="?",
    type=Path,
    default=ROOT / "report_assets" / "rendered_fyp_draft_verified",
)
args = parser.parse_args()
RENDER = args.render_dir.resolve()
OUT = RENDER / "contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)

pages = sorted(RENDER.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
thumb_w, label_h, cols, rows = 220, 24, 4, 4
for sheet_index in range(0, len(pages), cols * rows):
    group = pages[sheet_index:sheet_index + cols * rows]
    thumbs = []
    for page in group:
        with Image.open(page) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((thumb_w, 320))
            thumbs.append((page, thumb.copy()))
    cell_h = max(image.height for _, image in thumbs) + label_h
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), "#d8d8d8")
    draw = ImageDraw.Draw(sheet)
    for position, (page, image) in enumerate(thumbs):
        x = (position % cols) * thumb_w + (thumb_w - image.width) // 2
        y = (position // cols) * cell_h + label_h
        draw.text((position % cols * thumb_w + 6, position // cols * cell_h + 5), page.stem, fill="black")
        sheet.paste(image, (x, y))
    first = int(group[0].stem.split("-")[1])
    last = int(group[-1].stem.split("-")[1])
    sheet.save(OUT / f"pages-{first:03d}-{last:03d}.png")

print(f"Created {len(list(OUT.glob('pages-*.png')))} contact sheets for {len(pages)} pages")
