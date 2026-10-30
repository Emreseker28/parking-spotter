"""Runtime mode state machine + SQLite persistence.

Modes: SPOT_WATCH, GUARD, PAUSED (snooze), UNKNOWN (startup, not yet verified
against the camera). The database holds settings, mode state and per-spot tracker
state only -- numbers and text, never image bytes (the container rootfs is
read-only; this file lives on the one small writable state volume).

Manual switching only (automatic switching is Phase 3). Setting GUARD raises
``ModeUnavailable`` unless ``guard_available`` is on, and needs the guarded spot.
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

    @property
    def db(self) -> sqlite3.Connection:
        """Raw connection for modules that keep their own tables (car references)."""
        return self._db

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
        self.guard_spot: str | None = store.get("guard_spot")   # name of the guarded spot
        self.resume_guard: str | None = None   # set by verify(): re-arm this spot if the car is still there
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
    def guarding(self) -> bool:
        """GUARD logic should run (it keeps tracking while paused, muted)."""
        return self.mode == Mode.GUARD or (
            self.mode == Mode.PAUSED and self.resume_mode == Mode.GUARD
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
        self.store.set("guard_spot", self.guard_spot)

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
        # A persisted GUARD is never trusted blindly: start in SPOT_WATCH and let the
        # caller re-arm only if the car is still in the spot (verified on the camera).
        if self.persisted == Mode.GUARD and self.guard_available and self.guard_spot:
            self.resume_guard = self.guard_spot
        self.paused_until = None
        self._enter(Mode.SPOT_WATCH)
        return self.mode

    def set_manual(self, mode: Mode, guard_spot: str | None = None) -> None:
        """/watch or /guard: manual control, overrides auto until /auto.
        GUARD needs the name of the spot the owner's car is in."""
        if mode == Mode.GUARD and not self.guard_available:
            raise ModeUnavailable("GUARD is not available")
        if mode not in (Mode.SPOT_WATCH, Mode.GUARD):
            raise ValueError(f"cannot set {mode} manually")
        if mode == Mode.GUARD:
            if not guard_spot:
                raise ValueError("GUARD needs the spot the car is in")
            self.guard_spot = guard_spot
        else:
            self.guard_spot = None
        self.auto = False
        self.paused_until = None
        self.resume_guard = None
        self._enter(mode)

    def auto_enter_guard(self, spot_name: str) -> None:
        """Automatic switch to GUARD (arrival matched / owner confirmed). Unlike
        ``set_manual`` this keeps ``auto`` on: automatic mode stays automatic."""
        if not self.guard_available:
            raise ModeUnavailable("GUARD is not available")
        self.guard_spot = spot_name
        self.paused_until = None
        self.resume_guard = None
        self._enter(Mode.GUARD)

    def auto_enter_watch(self) -> None:
        """Automatic switch back to SPOT_WATCH (departure); keeps ``auto`` on."""
        self.guard_spot = None
        self.paused_until = None
        self.resume_guard = None
        self._enter(Mode.SPOT_WATCH)

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
