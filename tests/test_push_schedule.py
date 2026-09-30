"""Tests for the adaptive push interval (Phase 5)."""

from __future__ import annotations

import time
import unittest


def _time(hour: int):
    return time.struct_time((2026, 9, 30, hour, 0, 0, 0, 0, 0))


class NextPushIntervalTest(unittest.TestCase):
    def _bot(self, **push):
        from parkingbot.config import load_settings
        from parkingbot.bot import ParkingBot

        s = load_settings()
        cfg = {
            "enabled": True, "interval_minutes": 10, "confirm_interval_minutes": 2,
            "debounce_frames": 3, "policy": "free_only",
            "state_path": "data/push_state.json", "state_max_age_hours": 24,
            "night_interval_minutes": 0, "night_start_hour": 23, "night_end_hour": 6,
        }
        cfg.update(push)
        s.raw["push"] = cfg
        return ParkingBot(s)

    def test_base_interval_when_no_flip_pending(self):
        b = self._bot()
        state = {"a": {"confirmed": "free", "candidate": "free", "count": 3}}
        self.assertEqual(b._next_push_interval(state), 600.0)  # 10 min

    def test_confirm_interval_when_flip_pending(self):
        b = self._bot()
        state = {"a": {"confirmed": "occupied", "candidate": "free", "count": 1}}
        self.assertEqual(b._next_push_interval(state), 120.0)  # 2 min

    def test_first_reading_is_not_pending(self):
        # confirmed=None (baseline not set) -> not a pending flip -> base interval.
        b = self._bot()
        state = {"a": {"confirmed": None, "candidate": "free", "count": 1}}
        self.assertEqual(b._next_push_interval(state), 600.0)

    def test_night_overrides_confirm(self):
        b = self._bot(night_interval_minutes=20)
        state = {"a": {"confirmed": "occupied", "candidate": "free", "count": 1}}
        # 23:00 is night -> 20 min even though a flip is pending.
        self.assertEqual(b._next_push_interval(state, now=_time(23)), 1200.0)
        # 12:00 is day -> confirm 2 min (pending).
        self.assertEqual(b._next_push_interval(state, now=_time(12)), 120.0)

    def test_night_wraps_midnight(self):
        b = self._bot(night_interval_minutes=20)
        state = {"a": {"confirmed": "free", "candidate": "free", "count": 3}}
        # 02:00 is night (wraps) -> 20 min.
        self.assertEqual(b._next_push_interval(state, now=_time(2)), 1200.0)
        # 22:00 is not night (before 23) -> base 10 min.
        self.assertEqual(b._next_push_interval(state, now=_time(22)), 600.0)

    def test_night_disabled_by_default(self):
        b = self._bot()  # night_interval_minutes=0
        state = {"a": {"confirmed": "free", "candidate": "free", "count": 3}}
        self.assertEqual(b._next_push_interval(state, now=_time(2)), 600.0)


if __name__ == "__main__":
    unittest.main()
