#!/usr/bin/env python3
"""Phase 7: prepare a YOLO dataset from the Phase 1 crops for labeling.

Copies data/frames/*.jpg into data/dataset/images/{train,val}/ and creates empty
YOLO label files (.txt) + dataset.yaml. The actual labeling (drawing the car
box, or marking "no car") is a GROUND-TRUTH task — do NOT trust the bot verdict
encoded in the filename; look at the image.

Label with CVAT / Label Studio / Roboflow (import images/, export YOLO format),
or by hand: each .txt is one line per car `0 x y w h` (normalized 0-1), or
empty for no car. Then run scripts/finetune_detector.py on a GPU (Colab).
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = Path("/mnt/storage/parking-spotter/data/frames")
DATASET = ROOT / "data" / "dataset"


def main() -> int:
    if not FRAMES.exists():
        print(f"No frames dir: {FRAMES}")
        return 1
    images = sorted(FRAMES.glob("*.jpg"))
    if not images:
        print("No frames to label yet.")
        return 1

    # 80/20 train/val split (deterministic by sorted order).
    n_val = max(1, len(images) // 5)
    val = set(images[:n_val])
    for sub in ("train", "val"):
        (DATASET / "images" / sub).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / sub).mkdir(parents=True, exist_ok=True)

    for img in images:
        split = "val" if img in val else "train"
        dst = DATASET / "images" / split / img.name
        shutil.copy2(img, dst)
        # empty label file (fill with `0 x y w h` lines, or leave empty = no car)
        (DATASET / "labels" / split / f"{img.stem}.txt").write_text("")

    yaml = DATASET / "dataset.yaml"
    yaml.write_text(
        f"path: {DATASET}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        "names:\n"
        "  0: car\n"
    )
    print(f"Prepared {len(images)} images ({len(images) - n_val} train, {n_val} val).")
    print(f"Dataset: {yaml}")
    print("\nNext: label the images (draw a box around the car, or leave the .txt")
    print("empty for no car). Ground truth from viewing, NOT the filename verdict.")
    print("Then: python scripts/finetune_detector.py --data data/dataset/dataset.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
