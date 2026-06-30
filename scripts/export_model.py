#!/usr/bin/env python3
"""One-time host step: export YOLOv8n to ONNX into the models disk.

This is NOT a runtime dependency. ultralytics + torch are heavy and only needed
once to produce the ONNX file; the bot/container only needs onnxruntime. On a
box with a small root disk, install into a venv on the big disk, e.g.:

    export PIP_CACHE_DIR=/mnt/models/.export-cache TMPDIR=/mnt/models/.export-tmp
    python3 -m venv /mnt/models/.export-venv && . /mnt/models/.export-venv/bin/activate
    pip install --extra-index-url https://download.pytorch.org/whl/cpu ultralytics
    python scripts/export_model.py

Output goes to /mnt/models/yolov8n.onnx by default (the dedicated models disk).
Override with MODEL_OUT=/path/to/yolov8n.onnx. Weights are gitignored.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

TARGET = Path(os.environ.get("MODEL_OUT", "/mnt/models/yolov8n.onnx"))


def main() -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics", file=sys.stderr)
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading yolov8n.pt and exporting to ONNX (opset 12, imgsz 640)…")
    model = YOLO("yolov8n.pt")  # auto-downloads weights
    exported = model.export(format="onnx", opset=12, imgsz=640, simplify=True)

    src = Path(exported)
    if src.resolve() != TARGET.resolve():
        shutil.copy(src, TARGET)
    print(f"✅ Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
