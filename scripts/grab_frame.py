#!/usr/bin/env python3
"""Setup helper: grab a single frame from a channel and save it, so you can read
off ROI polygon coordinates for config.yaml (plan §5 / §9 checklist).

Run on the homeserver (reads .env / config.yaml the same way the bot does):

    python scripts/grab_frame.py --channel 1 --subtype 0 --out spotA.jpg

Open the saved JPEG in an image editor, read pixel coords of the parking-spot
corners, and paste them into the matching spot's `polygon:` in config.yaml.
Polygons are in each frame's own pixel space (ch1=960x1080, ch3=960x576).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as a loose script: add repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parkingbot.capture import CaptureError, CaptureWorker  # noqa: E402
from parkingbot.config import ConfigError, load_settings  # noqa: E402


async def _run(channel: int, subtype: int, out: Path) -> int:
    settings = load_settings()
    worker = CaptureWorker(settings)
    try:
        data = await worker.grab(channel, subtype)
    except CaptureError as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    out.write_bytes(data)
    print(f"✅ Saved {len(data)} bytes to {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Grab one frame for ROI setup.")
    p.add_argument("--channel", type=int, required=True)
    p.add_argument("--subtype", type=int, default=0, help="0=mainstream, 1=substream")
    p.add_argument("--out", type=Path, default=Path("frame.jpg"))
    args = p.parse_args()
    try:
        return asyncio.run(_run(args.channel, args.subtype, args.out))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
