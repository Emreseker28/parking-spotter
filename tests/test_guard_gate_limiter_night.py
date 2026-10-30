"""MotionGate, XVR event parsing, DutyLimiter, night detection."""

from __future__ import annotations

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from parkingbot.guard_gate import MotionGate, XvrEventGate, parse_events
from parkingbot.guard_limiter import DutyLimiter
from parkingbot.night import is_night, sun_times

IST = ZoneInfo("Europe/Istanbul")


def frame(level=100, blob=None, noise=0, seed=0):
    f = np.full((288, 352, 3), level, np.uint8)
    if noise:
        rng = np.random.default_rng(seed)
        f = np.clip(f.astype(int) + rng.integers(-noise, noise + 1, f.shape), 0, 255).astype(np.uint8)
    if blob:
        x1, y1, x2, y2 = blob
        f[y1:y2, x1:x2] = 230
    return f


BBOX = (70, 80, 280, 220)


class MotionGateTest(unittest.TestCase):
    def test_static_scene_is_quiet_even_with_sensor_noise(self):
        g = MotionGate(BBOX)
        res = [g.update(frame(noise=4, seed=i)) for i in range(30)]
        self.assertFalse(any(r.motion for r in res))

    def test_person_sized_blob_triggers(self):
        g = MotionGate(BBOX)
        for i in range(5):
            g.update(frame(noise=3, seed=i))
        r = g.update(frame(blob=(120, 120, 150, 190)))
        self.assertTrue(r.motion)
        self.assertFalse(r.global_change)

    def test_global_change_is_rejected_and_background_reseeded(self):
        g = MotionGate(BBOX)
        g.update(frame(100))
        r = g.update(frame(190))                       # headlight sweep / exposure jump
        self.assertTrue(r.global_change)
        self.assertFalse(r.motion)
        self.assertFalse(g.update(frame(190)).motion)  # new level is the new normal

    def test_one_or_two_pixel_speckle_is_removed(self):
        g = MotionGate(BBOX)
        g.update(frame(100))
        f = frame(100)
        for x, y in ((100, 100), (180, 150), (220, 190), (130, 170)):
            f[y, x] = 255                              # insects near the IR LEDs
        self.assertFalse(g.update(f).motion)

    def test_sensitivity_scale_raises_the_bar(self):
        small = (120, 120, 134, 140)                   # small blob, just above default threshold
        g = MotionGate(BBOX); g.update(frame())
        self.assertTrue(g.update(frame(blob=small)).motion)
        g2 = MotionGate(BBOX); g2.scale = 5.0; g2.update(frame())
        self.assertFalse(g2.update(frame(blob=small)).motion)

    def test_standing_person_not_absorbed_while_active(self):
        g = MotionGate(BBOX); g.update(frame())
        hits = [g.update(frame(blob=(120, 120, 150, 190)), active=True).motion for _ in range(60)]
        self.assertTrue(all(hits))                     # still visible after 60 frames


EVENT_STREAM = (
    "--myboundary\r\nContent-Type: text/plain\r\nContent-Length:243\r\n\r\n"
    "Code=RecordDelete;action=Pulse;index=0;data={\n   \"Channel\" : 3\n}\n\n"
    "--myboundary\r\n\r\nCode=SmartMotionHuman;action=Start;index=2;data={\n \"object\" : []\n}\n\n"
    "--myboundary\r\n\r\nCode=SmartMotionVehicle;action=Start;index=0;data={}\n"
)


class EventParseTest(unittest.TestCase):
    def test_parse_extracts_code_action_index(self):
        ev, _ = parse_events(EVENT_STREAM)
        self.assertIn(("SmartMotionHuman", "Start", 2), ev)
        self.assertIn(("RecordDelete", "Pulse", 0), ev)

    def test_split_across_chunks(self):
        carry = ""
        g = XvrEventGate("1.2.3.4", "u", "p", channel=3, clock=lambda: 50.0)
        for i in range(0, len(EVENT_STREAM), 17):      # awkward 17-byte chunks
            carry = g.feed(EVENT_STREAM[i:i + 17], carry)
        self.assertEqual(g.count, 1)                   # only index 2 (= channel 3), relevant code

    def test_only_own_channel_and_relevant_codes_count(self):
        g = XvrEventGate("h", "u", "p", channel=1, clock=lambda: 10.0, codes=("SmartMotionVehicle",))
        g.feed(EVENT_STREAM)                                              # index 0
        self.assertEqual((g.count, g.last_code), (1, "SmartMotionVehicle"))
        h = XvrEventGate("h", "u", "p", channel=1, clock=lambda: 10.0)   # default: humans only
        h.feed(EVENT_STREAM)
        self.assertEqual(h.count, 0)                                      # ch1 vehicle event ignored
        g2 = XvrEventGate("h", "u", "p", channel=9, clock=lambda: 10.0)
        g2.feed(EVENT_STREAM)
        self.assertEqual(g2.count, 0)

    def test_hint_window(self):
        t = [100.0]
        g = XvrEventGate("h", "u", "p", channel=3, clock=lambda: t[0])
        self.assertFalse(g.hint(100.0, 10))
        g.feed("Code=SmartMotionHuman;action=Start;index=2;data={}")
        self.assertTrue(g.hint(109.0, 10))
        self.assertFalse(g.hint(111.0, 10))


class LimiterTest(unittest.TestCase):
    def test_duty_is_busy_time_over_window(self):
        lim = DutyLimiter(window=60)
        for i in range(10):
            lim.record(1.0, i)                         # 10 s busy in the last minute
        self.assertAlmostEqual(lim.duty(10), 10 / 60)
        self.assertAlmostEqual(lim.duty(200), 0.0)     # all aged out

    def test_degrades_in_order_and_logs_each_step(self):
        lim = DutyLimiter(window=60, limit=0.40, base_fps=4)
        msgs, fps = [], []
        t = 0.0
        for _ in range(8):
            for k in range(30):
                lim.record(1.0, t + k)                 # 50 % duty
            t += 30
            m = lim.evaluate(t)
            if m:
                msgs.append(m); fps.append((lim.step, lim.fps, lim.gate_scale))
        self.assertEqual(fps, [(1, 4.0, 1.5), (2, 2.0, 1.5), (3, 1.0, 1.5)])
        self.assertTrue(all("limiter" in m for m in msgs))

    def test_recovers_one_step_after_calm(self):
        lim = DutyLimiter(window=60, limit=0.40, recover_after=300)
        lim.step = 3
        t = 1000.0
        self.assertIsNone(lim.evaluate(t))
        self.assertIsNone(lim.evaluate(t + 299))
        self.assertIn("step 2", lim.evaluate(t + 305))
        self.assertEqual(lim.fps, 2.0)

    def test_busy_burst_resets_the_calm_timer(self):
        lim = DutyLimiter(window=60, limit=0.40, recover_after=300)
        lim.step = 2
        lim.evaluate(1000.0)
        lim.evaluate(1200.0)
        for k in range(50):
            lim.record(1.0, 1250 + k)
        lim.evaluate(1300.0)                           # busy again -> steps up, timer reset
        self.assertEqual(lim.step, 3)
        self.assertIsNone(lim.evaluate(1310.0))

    def test_evaluates_at_most_every_5s(self):
        lim = DutyLimiter(window=60, limit=0.1)
        for k in range(30):
            lim.record(1.0, k)
        self.assertIsNotNone(lim.evaluate(30.0))
        self.assertIsNone(lim.evaluate(31.0))


class NightTest(unittest.TestCase):
    def test_istanbul_sun_times(self):
        rise, sset = sun_times(date(2026, 10, 30), 41.0, 29.0)
        r, s = rise.astimezone(IST), sset.astimezone(IST)
        self.assertLess(abs((r.hour * 60 + r.minute) - (7 * 60 + 31)), 8)
        self.assertLess(abs((s.hour * 60 + s.minute) - (18 * 60 + 3)), 8)

    def test_day_and_night(self):
        def at(h, m, d=(2026, 10, 30)):
            return datetime(*d, h, m, tzinfo=IST).timestamp()
        self.assertFalse(is_night(at(12, 0), IST))
        self.assertFalse(is_night(at(16, 0), IST))
        self.assertTrue(is_night(at(18, 13), IST))     # IR already on in the Phase 0 frames
        self.assertTrue(is_night(at(23, 0), IST))
        self.assertTrue(is_night(at(5, 0), IST))
        self.assertTrue(is_night(at(7, 30), IST))      # before sunrise + margin
        self.assertFalse(is_night(at(9, 0), IST))

    def test_summer_evening_is_still_day(self):
        t = datetime(2026, 6, 21, 19, 30, tzinfo=IST).timestamp()
        self.assertFalse(is_night(t, IST))
        self.assertTrue(is_night(datetime(2026, 6, 21, 21, 0, tzinfo=IST).timestamp(), IST))

    def test_fixed_hours_mode_wraps_midnight(self):
        def at(h):
            return datetime(2026, 10, 30, h, 0, tzinfo=IST).timestamp()
        self.assertTrue(is_night(at(23), IST, "fixed", start_hour=19, end_hour=6))
        self.assertTrue(is_night(at(3), IST, "fixed", start_hour=19, end_hour=6))
        self.assertFalse(is_night(at(12), IST, "fixed", start_hour=19, end_hour=6))


if __name__ == "__main__":
    unittest.main()
