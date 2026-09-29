"""Persistent push/debounce state (plan Phase 0 task 3).

The push loop's per-spot debounce/transition state used to live only in memory,
so a container restart mid-cycle lost it and the first reading after restart set
a new baseline. Now the state is persisted to `push.state_path` after every poll
and restored on startup — but only if the file is fresh (a stale file is ignored,
so a bot down for days does not "restore" a state from before the outage).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def save_push_state(path, state: dict) -> None:
    """Atomically write `state` as JSON (temp file + fsync + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def load_push_state(path, max_age_seconds: float) -> dict | None:
    """Load state saved by save_push_state.

    Returns None when the file is missing, unreadable, corrupt, not a JSON
    object, or older than `max_age_seconds` (stale).
    """
    p = Path(path)
    try:
        if time.time() - p.stat().st_mtime > max_age_seconds:
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
