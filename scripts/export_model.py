#!/usr/bin/env python3
"""One-time host step: export YOLOv8n to ONNX into ./models/yolov8n.onnx.

This is NOT a runtime dependency. ultralytics + torch are heavy and only needed
once to produce the ONNX file; the bot/container only needs onnxruntime.

Run on any machine with internet:

    pip install ultralytics
    python scripts/export_model.py

Then commit nothing (weights are gitignored) and either bake/mount ./models
into the container (docker-compose mounts it for you).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
TARGET = MODELS / "yolov8n.onnx"


def main() -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics", file=sys.stderr)
        return 1

    MODELS.mkdir(exist_ok=True)
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
