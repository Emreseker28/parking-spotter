"""SpotTracker: debounce, alert/reminder/taken rules, mute, re-arm, persistence."""

from __future__ import annotations

import unittest

from parkingbot.spot_tracker import FREE, OCCUPIED, SpotTracker


def kinds(events):
    return [e.kind for e in events]


class DebounceTest(unittest.TestCase):
    def test_single_free_read_does_not_alert(self):
        t = SpotTracker("a")
        self.assertEqual(t.observe(True, 0), [])
        self.assertEqual(t.observe(False, 60), [])
        self.assertTrue(t.needs_confirm)
        self.assertEqual(t.confirmed, OCCUPIED)

    def test_two_consecutive_free_reads_alert_once(self):
        t = SpotTracker("a")
        t.observe(True, 0)
        t.observe(False, 60)
        ev = t.observe(False, 70)
        self.assertEqual(kinds(ev), ["free_alert"])
        self.assertFalse(ev[0].immediate)          # a transition was seen
        self.assertEqual(t.confirmed, FREE)
        self.assertFalse(t.needs_confirm)

    def test_passing_car_cancels_pending(self):
        t = SpotTracker("a")
        t.observe(True, 0)
        t.observe(False, 60)
        self.assertEqual(t.observe(True, 70), [])
        self.assertEqual(t.pending_free, 0)
        self.assertEqual(t.observe(False, 120), [])   # counting starts over
        self.assertEqual(t.confirmed, OCCUPIED)

    def test_already_free_on_entry_is_immediate(self):
        t = SpotTracker("a")
        t.observe(False, 0)
        ev = t.observe(False, 10)
        self.assertEqual(kinds(ev), ["free_alert"])
        self.assertTrue(ev[0].immediate)

    def test_confirm_reads_one_alerts_straight_away(self):
        t = SpotTracker("a", confirm_reads=1)
        t.observe(True, 0)
        self.assertEqual(kinds(t.observe(False, 60)), ["free_alert"])


class ReminderTest(unittest.TestCase):
    def _free(self):
        t = SpotTracker("a", remind_every=600, remind_max=3)
        t.observe(True, 0)
        t.observe(False, 60)
        t.observe(False, 70)       # alert at t=70
        return t

    def test_reminders_every_interval_max_three(self):
        t = self._free()
        self.assertEqual(t.observe(False, 130), [])
        self.assertEqual(t.observe(False, 669), [])          # 599s after the alert
        ev = t.observe(False, 670)
        self.assertEqual([(e.kind, e.n) for e in ev], [("reminder", 1)])
        self.assertEqual(kinds(t.observe(False, 1270)), ["reminder"])
        self.assertEqual(kinds(t.observe(False, 1870)), ["reminder"])
        self.assertEqual(t.observe(False, 2470), [])         # cap reached
        self.assertEqual(t.observe(False, 9999), [])

    def test_taken_after_alert_says_so_and_resets(self):
        t = self._free()
        self.assertEqual(kinds(t.observe(True, 200)), ["taken"])
        self.assertFalse(t.alerted)
        self.assertEqual(t.reminders, 0)
        # frees up again -> a fresh alert
        t.observe(False, 300)
        self.assertEqual(kinds(t.observe(False, 310)), ["free_alert"])

    def test_no_taken_message_if_never_alerted(self):
        t = SpotTracker("a")
        t.observe(False, 0)
        t.observe(False, 10, muted=True)       # confirmed free but muted
        self.assertEqual(t.observe(True, 20), [])


class MuteAndReenterTest(unittest.TestCase):
    def test_muted_tracks_silently_then_alerts_on_unmute(self):
        t = SpotTracker("a")
        t.observe(True, 0)
        t.observe(False, 60, muted=True)
        self.assertEqual(t.observe(False, 70, muted=True), [])
        self.assertEqual(t.confirmed, FREE)
        self.assertFalse(t.alerted)
        self.assertEqual(kinds(t.observe(False, 130)), ["free_alert"])

    def test_on_enter_rearms_a_free_spot(self):
        t = SpotTracker("a")
        t.observe(True, 0); t.observe(False, 1); t.observe(False, 2)
        self.assertTrue(t.alerted)
        t.on_enter()
        ev = t.observe(False, 100)
        self.assertEqual(kinds(ev), ["free_alert"])
        self.assertTrue(ev[0].immediate)

    def test_veto_returns_to_occupied(self):
        t = SpotTracker("a")
        t.observe(True, 0); t.observe(False, 1); t.observe(False, 2)
        t.veto_free()
        self.assertEqual(t.confirmed, OCCUPIED)
        self.assertFalse(t.alerted)
        self.assertEqual(t.observe(False, 100), [])     # needs two reads again


class PersistenceTest(unittest.TestCase):
    def test_roundtrip_keeps_alerted_so_restart_does_not_realert(self):
        t = SpotTracker("a")
        t.observe(True, 0); t.observe(False, 1); t.observe(False, 2)
        t2 = SpotTracker("a")
        t2.restore(t.to_dict())
        self.assertEqual(t2.confirmed, FREE)
        self.assertTrue(t2.alerted)
        self.assertEqual(t2.observe(False, 10), [])

    def test_restore_ignores_garbage(self):
        t = SpotTracker("a")
        t.restore({"confirmed": "banana"})
        self.assertIsNone(t.confirmed)


if __name__ == "__main__":
    unittest.main()
