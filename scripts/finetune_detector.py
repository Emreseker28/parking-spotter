#!/usr/bin/env python3
"""Phase 7: fine-tune YOLOv8n on the labeled parking crops (Colab-ready).

Why: the stock COCO YOLOv8n misses dark / top-down cars on grayscale IR night
frames. Fine-tuning on the Phase 1 crops (ground-truth car boxes, 50+ night
cases) teaches it the exact failure mode.

Run on a free GPU (Colab T4) or a work machine with ultralytics installed:

    # 1. Prepare the dataset (see scripts/label_dataset.py) so that:
    #    data/dataset/images/{train,val}/*.jpg
    #    data/dataset/labels/{train,val}/*.txt   (YOLO format: class x y w h, normalized)
    #    data/dataset/dataset.yaml
    # 2. Run:
    python scripts/finetune_detector.py --data data/dataset/dataset.yaml --epochs 50

Output: runs/train/weights/best.onnx (opset 12, imgsz 640). Validate on held-out
night frames against the stock model; ship only if it wins at night without
regressing daytime. Copy to /mnt/models/yolov8n_ft.onnx and point model_path at
it (keep the stock yolov8n.onnx as the rollback).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on parking crops.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", default="yolov8n.pt", help="Base weights")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="parking_ft")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. In Colab, run:")
        print("  !pip install ultralytics")
        return 1

    data = Path(args.data)
    if not data.exists():
        print(f"Dataset config not found: {data}")
        print("Prepare it with scripts/label_dataset.py first.")
        return 1

    print(f"Fine-tuning {args.weights} on {data} for {args.epochs} epochs…")
    model = YOLO(args.weights)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=args.project, name=args.name)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        print(f"Training did not produce {best}")
        return 1

    print(f"Exporting {best} -> ONNX (opset 12, imgsz {args.imgsz})…")
    exported = model.export(format="onnx", opset=12, imgsz=args.imgsz)
    out = Path(exported)
    target = Path("/mnt/models/yolov8n_ft.onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() != target.resolve():
        shutil.copy(out, target)
    print(f"✅ Wrote {target}")
    print("Validate on held-out NIGHT frames vs the stock yolov8n.onnx; ship only")
    print("if it wins at night without regressing daytime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
