#!/usr/bin/env python3
"""One-time host step: export a YOLOv8 model to ONNX into the models disk.

This is NOT a runtime dependency. ultralytics + torch are heavy and only needed
once to produce the ONNX file; the bot/container only needs onnxruntime. On a
box with a small root disk, install into a venv on the big disk, e.g.:

    export PIP_CACHE_DIR=/mnt/models/.export-cache TMPDIR=/mnt/models/.export-tmp
    python3 -m venv /mnt/models/.export-venv && . /mnt/models/.export-venv/bin/activate
    pip install --extra-index-url https://download.pytorch.org/whl/cpu ultralytics
    python scripts/export_model.py

Output goes to /mnt/models/<weights-stem>.onnx by default (the dedicated models
disk). Override with --out=/path/to/model.onnx or MODEL_OUT=/path/to/model.onnx.
Weights are gitignored.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLOv8 model to ONNX.")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="ultralytics weights to export (default: yolov8n.pt)")
    parser.add_argument("--out", default=None,
                        help="output ONNX path (default: /mnt/models/<weights-stem>.onnx)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics", file=sys.stderr)
        return 1

    default_out = f"/mnt/models/{Path(args.weights).stem}.onnx"
    target = Path(args.out or os.environ.get("MODEL_OUT", default_out))
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.weights} and exporting to ONNX (opset 12, imgsz 640)…")
    model = YOLO(args.weights)  # auto-downloads weights
    exported = model.export(format="onnx", opset=12, imgsz=640, simplify=True)

    src = Path(exported)
    if src.resolve() != target.resolve():
        shutil.copy(src, target)
    print(f"✅ Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
