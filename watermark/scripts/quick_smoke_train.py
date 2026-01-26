#!/usr/bin/env python3
"""
Legacy entrypoint stub.

The old synthetic bit-payload smoke test moved to `watermark/legacy/quick_smoke_train.py`.
The current supported smoke run is:
  `python -m watermark.scripts.quick_voice_smoke_train ...`
"""

def main() -> int:
    print("This script is legacy.")
    print("Use: `python -m watermark.scripts.quick_voice_smoke_train ...`")
    print("Legacy version: `python -m watermark.legacy.quick_smoke_train`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

