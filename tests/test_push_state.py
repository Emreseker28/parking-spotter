"""Tests for parkingbot.push_state (atomic save + freshness-checked load)."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from parkingbot.push_state import load_push_state, save_push_state


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "push_state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load_round_trips(self):
        state = {
            "Ön (sarı ev)": {"confirmed": "occupied", "candidate": "free", "count": 2},
            "ev önü": {"confirmed": "free", "candidate": "free", "count": 3},
        }
        save_push_state(self.path, state)
        self.assertEqual(load_push_state(self.path, max_age_seconds=3600), state)

    def test_creates_parent_directories(self):
        nested = self.path.parent / "deep" / "push_state.json"
        save_push_state(nested, {"a": {"confirmed": None, "candidate": None, "count": 0}})
        self.assertTrue(nested.exists())

    def test_overwrites_previous_state(self):
        save_push_state(self.path, {"old": True})
        save_push_state(self.path, {"new": True})
        self.assertEqual(load_push_state(self.path, max_age_seconds=3600), {"new": True})


class LoadMissingStaleCorruptTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "push_state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_push_state(self.path, max_age_seconds=3600))

    def test_stale_file_returns_none(self):
        save_push_state(self.path, {"a": 1})
        old = time.time() - 7200  # 2 h ago
        os.utime(self.path, (old, old))
        self.assertIsNone(load_push_state(self.path, max_age_seconds=3600))

    def test_fresh_file_within_max_age_loads(self):
        save_push_state(self.path, {"a": 1})
        self.assertEqual(load_push_state(self.path, max_age_seconds=3600), {"a": 1})

    def test_corrupt_json_returns_none(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        self.assertIsNone(load_push_state(self.path, max_age_seconds=3600))

    def test_non_dict_json_returns_none(self):
        self.path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        self.assertIsNone(load_push_state(self.path, max_age_seconds=3600))

    def test_empty_file_returns_none(self):
        self.path.write_text("", encoding="utf-8")
        self.assertIsNone(load_push_state(self.path, max_age_seconds=3600))


class AtomicityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "push_state.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_temp_file_left_behind(self):
        save_push_state(self.path, {"a": 1})
        self.assertFalse((self.path.parent / "push_state.json.tmp").exists())

    def test_existing_file_is_valid_json_after_save(self):
        # A reader must never observe a partially-written file: after save, the
        # path holds complete valid JSON (the temp+rename is atomic).
        save_push_state(self.path, {"spot": {"confirmed": "occupied"}})
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data, {"spot": {"confirmed": "occupied"}})

    def test_unicode_round_trips(self):
        state = {"Ön (sarı ev)": {"confirmed": "occupied"}}
        save_push_state(self.path, state)
        self.assertEqual(load_push_state(self.path, max_age_seconds=3600), state)


if __name__ == "__main__":
    unittest.main()
