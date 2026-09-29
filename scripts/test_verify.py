#!/usr/bin/env python3
"""Setup helper: run the local VLM verifier against ONE spot and print its verdict.

Use this to validate the ollama vision model on a real night frame BEFORE trusting the
push-loop gate — especially the hard cases (dark car, top-down) where YOLO fails.

Grab a live frame straight off the recorder for a spot:

    python scripts/test_verify.py --spot "ev önü"

...or judge an existing JPEG (e.g. a saved night frame):

    python scripts/test_verify.py --spot "ev önü" --image night.jpg

Reads .env / config.yaml the same way the bot does. Needs ollama reachable at
OLLAMA_URL (default http://localhost:11434) with the configured verify.model pulled.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as a loose script: add repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from parkingbot.capture import CaptureError, CaptureWorker  # noqa: E402
from parkingbot.config import ConfigError, load_settings  # noqa: E402
from parkingbot.detection import Detector  # noqa: E402
from parkingbot.verifier import Verifier  # noqa: E402


async def _run(spot_name: str, image: Path | None) -> int:
    settings = load_settings()
    spot = next((s for s in settings.spots if s.name == spot_name), None)
    if spot is None:
        names = ", ".join(repr(s.name) for s in settings.spots)
        print(f"No spot named {spot_name!r}. Known spots: {names}", file=sys.stderr)
        return 2

    if image is not None:
        img = Detector.decode(image.read_bytes())
    else:
        subtype = int(settings.capture.get("detect_subtype", 0))
        try:
            jpeg = await CaptureWorker(settings).grab(spot.channel, subtype)
        except CaptureError as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            return 1
        img = Detector.decode(jpeg)

    verifier = Verifier(settings)
    if not verifier.enabled:
        print("verify.enabled is false in config.yaml — enable it to test.", file=sys.stderr)
        return 2

    res = await verifier.verify_spot(np.ascontiguousarray(img), spot)
    print(f"spot:       {spot.name} (ch{spot.channel})")
    print(f"vlm model:  {settings.verify.get('model')}  @ {verifier._url}")
    print(f"ok:         {res.ok}")
    print(f"occupied:   {res.occupied}")
    print(f"confidence: {res.confidence}")
    print(f"reason:     {res.reason}")
    return 0 if res.ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Run the local VLM verifier on one spot.")
    p.add_argument("--spot", required=True, help="spot name exactly as in config.yaml")
    p.add_argument("--image", type=Path, default=None, help="optional JPEG to judge instead of a live grab")
    args = p.parse_args()
    try:
        return asyncio.run(_run(args.spot, args.image))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
