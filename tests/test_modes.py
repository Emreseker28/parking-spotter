"""ModeController + StateStore."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parkingbot.modes import Mode, ModeController, ModeUnavailable, StateStore


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class ModesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "state" / "spotter.db"
        self.clock = Clock()

    def tearDown(self):
        self._tmp.cleanup()

    def ctl(self):
        return ModeController(StateStore(self.db), self.clock)

    def test_store_roundtrip_and_overwrite(self):
        s = StateStore(self.db)
        self.assertIsNone(s.get("x"))
        s.set("x", {"a": 1}); s.set("x", {"a": 2})
        self.assertEqual(s.get("x"), {"a": 2})
        self.assertEqual(StateStore(self.db).get("x"), {"a": 2})   # on disk

    def test_starts_unknown_until_verified(self):
        c = self.ctl()
        self.assertEqual(c.mode, Mode.UNKNOWN)
        self.assertFalse(c.watching)
        self.assertEqual(c.verify(), Mode.SPOT_WATCH)
        self.assertTrue(c.watching)

    def test_persisted_guard_falls_back_in_phase1(self):
        c = self.ctl(); c.guard_available = True; c.verify(); c.set_manual(Mode.GUARD)
        c2 = self.ctl()
        self.assertEqual(c2.persisted, Mode.GUARD)
        self.assertEqual(c2.verify(), Mode.SPOT_WATCH)     # GUARD not available

    def test_guard_unavailable(self):
        c = self.ctl(); c.verify()
        with self.assertRaises(ModeUnavailable):
            c.set_manual(Mode.GUARD)
        self.assertEqual(c.mode, Mode.SPOT_WATCH)

    def test_manual_overrides_auto(self):
        c = self.ctl(); c.verify(); c.set_auto(True)
        c.set_manual(Mode.SPOT_WATCH)
        self.assertFalse(c.auto)
        self.assertFalse(self.ctl().auto)

    def test_pause_resume_and_expiry(self):
        c = self.ctl(); c.verify(); c.consume_entered_watch()
        c.pause(600)
        self.assertEqual(c.mode, Mode.PAUSED)
        self.assertTrue(c.muted and c.watching)            # keeps tracking, muted
        self.assertAlmostEqual(c.pause_remaining, 600)
        self.clock.t += 599
        self.assertFalse(c.tick())
        self.clock.t += 2
        self.assertTrue(c.tick())
        self.assertEqual(c.mode, Mode.SPOT_WATCH)
        self.assertFalse(c.muted)

    def test_resume_does_not_rearm_but_real_entry_does(self):
        c = self.ctl(); c.verify(); self.assertTrue(c.consume_entered_watch())
        c.pause(60); c.resume()
        self.assertFalse(c.consume_entered_watch())        # same watch session
        c.set_manual(Mode.SPOT_WATCH)
        self.assertFalse(c.consume_entered_watch())        # already watching

    def test_pause_before_verify_rejected(self):
        with self.assertRaises(ValueError):
            self.ctl().pause(60)

    def test_pause_survives_restart_until_expiry(self):
        c = self.ctl(); c.verify(); c.pause(300)
        self.clock.t += 100
        c2 = self.ctl()
        self.assertEqual(c2.verify(), Mode.PAUSED)
        self.assertAlmostEqual(c2.pause_remaining, 200)
        self.clock.t += 300
        c3 = self.ctl()
        self.assertEqual(c3.verify(), Mode.SPOT_WATCH)     # expired while down


if __name__ == "__main__":
    unittest.main()
