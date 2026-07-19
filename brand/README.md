# TTS Hub — brand assets

The mark is a hexagonal "hub" node drawn as a speech bubble, with a five-bar
waveform inside. Hex = node/hub, bubble = speech, waveform = audio. All three
meanings are in the product name.

## Files

| File | Use |
|---|---|
| `mark.svg` | Primary mark. Outlined, transparent background. Brand green + light caps. |
| `mark-mono.svg` | Single colour via `currentColor`. Print, embroidery, CI badges, unknown backgrounds. |
| `icon.svg` | App-icon lockup — solid inverted mark on green, corners pre-rounded. Manifest `purpose: any`, and the SVG favicon. |
| `icon-maskable.svg` | Full-bleed square, mark scaled down to clear Android's 80%-diameter safe zone. Manifest `purpose: maskable`. |
| `icon-apple.svg` | Full-bleed square, no rounding — iOS applies its own superellipse mask and would otherwise double-round. |

## Where it is served

`webui.py` mounts this directory at **`/brand`**, so every UI shares one copy and they
cannot drift. The `desktop` client points its favicon links there directly.

`mobile/` additionally keeps its *own* generated PNG copies (`icon-192.png`,
`icon-512.png`, `icon-maskable-512.png`, `apple-touch-icon.png`, `favicon-32.png`).
That duplication is deliberate: the service worker only caches paths under `/mobile`,
so PWA install icons must live there to survive offline. The manifest therefore lists
PNGs only — no `/brand` entries.

## Rules

**Never hardcode white.** The two outer waveform caps are the *neutral*, not white.
They must resolve to the surface foreground (`var(--tx)` in the app, `#e9ebef` on
the brand dark, `#15181d` on light) or they disappear on light backgrounds. Only the
baked-background icon files may use literal `#ffffff`, because the surface underneath
them is fixed.

**Colour comes from the existing palette** — no new tokens were introduced:

| Token | Dark | Light |
|---|---|---|
| `--accent` | `#3ddc97` | `#10996a` |
| `--tx` | `#e9ebef` | `#15181d` |
| `--bg` | `#0c0d10` | `#eef0f3` |

**Minimum sizes.** Outlined mark: 20px. Below that use `icon.svg` (solid), which
holds down to 16px.

**The header mark is inlined**, not `<img>`-linked, in both `mobile/index.html` and
`desktop/index.html` - it has to read the theme's CSS custom properties, which an
`<img>` cannot. `mark.svg` is the source of truth for the path data; if you change the
geometry here, change it in those two files too.

## Regenerating the PNGs

Requires `rsvg-convert` (`brew install librsvg`). From the repo root:

```sh
rsvg-convert -w  32 -h  32 brand/icon.svg          -o brand/favicon-32.png
rsvg-convert -w 180 -h 180 brand/icon-apple.svg    -o brand/apple-touch-icon.png
rsvg-convert -w 512 -h 512 brand/icon.svg          -o brand/icon-512.png

rsvg-convert -w 192 -h 192 brand/icon.svg          -o mobile/icon-192.png
rsvg-convert -w 512 -h 512 brand/icon.svg          -o mobile/icon-512.png
rsvg-convert -w 512 -h 512 brand/icon-maskable.svg -o mobile/icon-maskable-512.png
rsvg-convert -w 180 -h 180 brand/icon-apple.svg    -o mobile/apple-touch-icon.png
rsvg-convert -w  32 -h  32 brand/icon.svg          -o mobile/favicon-32.png
```

Bump `CACHE` in `mobile/sw.js` after regenerating, or clients keep the old icons.
