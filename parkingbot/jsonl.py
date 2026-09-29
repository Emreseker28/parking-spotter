"""Append-only JSONL writing with size-based rotation.

The accuracy log (and later the feedback log) grow one JSON line per event and
must never fill the disk (plan Phase 0 task 2). Rotation is size-based: when the
next line would push the file past `max_bytes`, the file is rotated to `.1`
(shifting older generations up) before the line is written. At most `keep`
generations are kept; the oldest is dropped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_KEEP = 3


def append_jsonl(path, record: dict, max_bytes: int = DEFAULT_MAX_BYTES,
                 keep: int = DEFAULT_KEEP) -> None:
    """Append one JSON record to `path`, rotating first if it would exceed max_bytes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if p.exists() and p.stat().st_size + len(line.encode("utf-8")) > max_bytes:
        _rotate(p, keep)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _rotate(p: Path, keep: int) -> None:
    """Shift p -> p.1 -> p.2 ... keeping `keep` generations (oldest dropped)."""
    if keep < 1:
        p.unlink()
        return
    oldest = p.with_name(f"{p.name}.{keep}")
    if oldest.exists():
        oldest.unlink()
    for i in range(keep - 1, 0, -1):
        src = p.with_name(f"{p.name}.{i}")
        dst = p.with_name(f"{p.name}.{i + 1}")
        if src.exists():
            os.replace(src, dst)
    os.replace(p, p.with_name(f"{p.name}.1"))
