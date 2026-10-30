"""Runtime mode state machine + SQLite persistence.

Modes: SPOT_WATCH, GUARD, PAUSED (snooze), UNKNOWN (startup, not yet verified
against the camera). The database holds settings, mode state and per-spot tracker
state only -- numbers and text, never image bytes (the container rootfs is
read-only; this file lives on the one small writable state volume).

Phase 1 ships manual switching only; GUARD itself arrives in Phase 2, so setting
it raises ``ModeUnavailable`` until ``guard_available`` is turned on.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from enum import Enum
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class Mode(str, Enum):
    UNKNOWN = "UNKNOWN"
    SPOT_WATCH = "SPOT_WATCH"
    GUARD = "GUARD"
    PAUSED = "PAUSED"


class ModeUnavailable(RuntimeError):
    """The requested mode is not implemented yet."""


class StateStore:
    """Tiny key/value store over SQLite (JSON values)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, ts REAL NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str, default=None):
        row = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except ValueError:
            return default

    def get_with_ts(self, key: str):
        row = self._db.execute("SELECT value, ts FROM kv WHERE key=?", (key,)).fetchone()
        if row is None:
            return None, None
        try:
            return json.loads(row[0]), float(row[1])
        except ValueError:
            return None, None

    def set(self, key: str, value, ts: float | None = None) -> None:
        self._db.execute(
            "INSERT INTO kv(key, value, ts) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts",
            (key, json.dumps(value), time.time() if ts is None else ts),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class ModeController:
    def __init__(self, store: StateStore, clock: Callable[[], float] = time.time,
                 guard_available: bool = False):
        self.store = store
        self._clock = clock
        self.guard_available = guard_available
        self.mode: Mode = Mode.UNKNOWN
        self.auto: bool = bool(store.get("auto", False))
        self.paused_until: float | None = store.get("paused_until")
        self.resume_mode: Mode = Mode.SPOT_WATCH
        self.persisted: Mode = self._load_persisted()
        self._entered_watch = False

    def _load_persisted(self) -> Mode:
        try:
            return Mode(self.store.get("mode", Mode.SPOT_WATCH.value))
        except ValueError:
            return Mode.SPOT_WATCH

    # ------------------------------------------------------------ properties
    @property
    def muted(self) -> bool:
        return self.mode == Mode.PAUSED

    @property
    def watching(self) -> bool:
        """SPOT_WATCH logic should run (it keeps tracking while paused, muted)."""
        return self.mode == Mode.SPOT_WATCH or (
            self.mode == Mode.PAUSED and self.resume_mode == Mode.SPOT_WATCH
        )

    @property
    def pause_remaining(self) -> float:
        if self.mode != Mode.PAUSED or self.paused_until is None:
            return 0.0
        return max(0.0, self.paused_until - self._clock())

    # ----------------------------------------------------------- transitions
    def _persist(self) -> None:
        self.store.set("mode", self.mode.value)
        self.store.set("auto", self.auto)
        self.store.set("paused_until", self.paused_until)
        self.store.set("resume_mode", self.resume_mode.value)

    def _enter(self, mode: Mode) -> None:
        # Re-arm alerts only on a real entry; pausing/resuming keeps watching.
        if mode == Mode.SPOT_WATCH and not self.watching:
            self._entered_watch = True
        self.mode = mode
        self._persist()
        log.info("mode -> %s (auto=%s)", mode.value, self.auto)

    def consume_entered_watch(self) -> bool:
        """True once after SPOT_WATCH was (re)entered, so trackers re-arm."""
        flag, self._entered_watch = self._entered_watch, False
        return flag

    def verify(self) -> Mode:
        """First successful camera reading after startup: trust (or reset) the
        persisted mode. Phase 1 can only run SPOT_WATCH, so anything else
        persisted falls back to it; a still-running pause is honoured."""
        if self.mode != Mode.UNKNOWN:
            return self.mode
        if (self.persisted == Mode.PAUSED and self.paused_until
                and self.paused_until > self._clock()):
            self.mode = Mode.PAUSED
            self.resume_mode = Mode.SPOT_WATCH
            self._persist()
            log.info("mode verified -> PAUSED (%.0fs left)", self.pause_remaining)
            return self.mode
        target = Mode.GUARD if (self.persisted == Mode.GUARD and self.guard_available) \
            else Mode.SPOT_WATCH
        self.paused_until = None
        self._enter(target)
        return self.mode

    def set_manual(self, mode: Mode) -> None:
        """/watch or /guard: manual control, overrides auto until /auto."""
        if mode == Mode.GUARD and not self.guard_available:
            raise ModeUnavailable("GUARD is not implemented yet (Phase 2)")
        if mode not in (Mode.SPOT_WATCH, Mode.GUARD):
            raise ValueError(f"cannot set {mode} manually")
        self.auto = False
        self.paused_until = None
        self._enter(mode)

    def set_auto(self, on: bool) -> None:
        self.auto = on
        self._persist()

    def pause(self, seconds: float) -> None:
        if self.mode == Mode.UNKNOWN:
            raise ValueError("mode not verified yet")
        if self.mode != Mode.PAUSED:
            self.resume_mode = self.mode
        self.paused_until = self._clock() + float(seconds)
        self._enter(Mode.PAUSED)

    def resume(self) -> bool:
        """End the snooze. Returns True if a pause was active."""
        if self.mode != Mode.PAUSED:
            return False
        self.paused_until = None
        self._enter(self.resume_mode)
        return True

    def tick(self) -> bool:
        """Expire a finished pause. Returns True when it just ended."""
        if (self.mode == Mode.PAUSED and self.paused_until is not None
                and self._clock() >= self.paused_until):
            return self.resume()
        return False
