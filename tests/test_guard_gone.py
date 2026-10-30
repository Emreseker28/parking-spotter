"""CAR_GONE, departure and escalation in the GUARD engine (fake clock, scripted camera)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from parkingbot.config import load_settings
from parkingbot.guard import GuardConfig, GuardEngine
from parkingbot.guard_geometry import Zones
from parkingbot.modes import Mode, ModeController, StateStore
from parkingbot.spot_tracker import SpotTracker
from parkingbot.spotwatch import Outbox

CAR = (100.0, 100.0, 200.0, 180.0)
ZONES = Zones((352, 288), CAR, [(70, 80), (230, 80), (230, 200), (70, 200)])
RNG = np.random.default_rng(1)


def scene(car=True):
    f = np.full((288, 352, 3), (80, 90, 100), np.uint8)
    cv2.randu(f, (70, 80, 90), (95, 105, 115))
    if car:
        cv2.rectangle(f, (100, 100), (200, 180), (200, 200, 205), -1)
        for x in range(105, 195, 22):
            cv2.rectangle(f, (x, 108), (x + 12, 128), (60, 60, 70), -1)
        cv2.rectangle(f, (110, 150), (190, 170), (90, 90, 95), -1)
    return np.clip(f.astype(int) + RNG.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


class Notifier:
    def __init__(self):
        self.sent = []

    async def send_text(self, text, silent=False, **kw):
        self.sent.append({"text": text, "silent": silent, **kw})

    async def send_photo(self, jpeg, caption, **kw):
        self.sent.append({"text": caption, **kw})

    async def send_media_group(self, jpegs, caption, **kw):
        self.sent.append({"text": caption, "n": len(jpegs), **kw})


class Cap:
    def __init__(self):
        self.fail = False

    async def grab(self, ch, sub, expected_size=None):
        if self.fail:
            raise RuntimeError("camera down")
        return b"jpeg"


class GDet:
    def ensure(self):
        pass

    def detect(self, *a, **k):
        return []


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings()
        self.clk = Clock()
        store = StateStore(Path(self._tmp.name) / "s.db")
        self.ctl = ModeController(store, self.clk, guard_available=True)
        self.ctl.verify()
        self.n = Notifier()
        self.cfg = GuardConfig(followup_main=False, presence_check_s=1.0, gone_confirm_gap_s=1.0,
                               gone_recheck_s=15.0, departure_s=90.0, gone_reply_s=120.0)
        self.eng = GuardEngine(self.settings, self.ctl, Cap(), None, Outbox(self.n, self.clk),
                               clock=self.clk, cfg=self.cfg, guard_detector=GDet())
        self.eng.zones = ZONES
        self.eng._init_state(self.settings.spots[0])
        self.ctl.auto_enter_guard(self.settings.spots[0].name)
        self.car_visible = True
        self.eng._pick_car = lambda jpeg, spot: (100.0, 100.0, 200.0, 180.0) if self.car_visible else None
        self.departures = []

        async def on_dep(reason):
            self.departures.append(reason)
        self.eng.on_departure = on_dep
        orig = asyncio.sleep
        self._orig_sleep = orig

        async def fast(d, *a, **k):          # advance the fake clock instead of waiting
            self.clk.t += d
            await orig(0)
        asyncio.sleep = fast

    def tearDown(self):
        asyncio.sleep = self._orig_sleep
        self._tmp.cleanup()

    def texts(self):
        return [s["text"] for s in self.n.sent]

    async def feed(self, car, seconds, dt=1.0):
        n = int(seconds / dt)
        for _ in range(n):
            self.clk.t += dt
            self.eng.on_frame(scene(car), self.clk.t)
            await self._orig_sleep(0)

    async def settle(self, rounds=60):
        for _ in range(rounds):
            await self._orig_sleep(0)
            if self.eng._gone_task is None:
                return


class DetectionTest(Base):
    def test_a_parked_car_never_raises_anything(self):
        async def go():
            await self.feed(True, 120)
            await self.settle()
        asyncio.run(go())
        self.assertEqual(self.n.sent, [])
        self.assertIsNone(self.eng.gone_state)

    def test_vanished_car_raises_car_gone_with_buttons_and_evidence(self):
        async def go():
            await self.feed(True, 20)
            self.car_visible = False
            await self.feed(False, 8)
            await asyncio.wait_for(self._wait_alert(), 5)
        asyncio.run(go())
        a = [s for s in self.n.sent if "ARAÇ GİTTİ" in s["text"]]
        self.assertEqual(len(a), 1)
        self.assertIn("ALARM", a[0]["text"])
        self.assertGreaterEqual(a[0].get("n", 1), 1)                  # photos from the RAM ring
        # Telegram albums cannot carry buttons: the [Me]/[Not me] arrive in a follow-up message
        b = [s for s in self.n.sent if s.get("buttons")]
        self.assertEqual(len(b), 1)
        self.assertEqual([x[0] for x in b[0]["buttons"][0]], ["Bendim", "Ben değilim"])
        self.assertEqual(self.eng.active_alert.kind, "gone")
        self.assertEqual(self.eng.alerts["ALERT"], 1)
        self.assertEqual(len(self.eng.latencies), 1)

    async def _wait_alert(self):
        while not self.eng.gone_alerted:
            await self._orig_sleep(0)
            self.clk.t += 0.5

    def test_zone_changed_but_main_stream_still_sees_the_car_is_a_false_alarm(self):
        async def go():
            await self.feed(True, 20)
            await self.feed(False, 8)              # e.g. exposure shift / person in front
            await self.settle()
        asyncio.run(go())                          # _pick_car still returns the car
        self.assertEqual([t for t in self.texts() if "ARAÇ GİTTİ" in t], [])
        self.assertIsNone(self.eng.gone_state)
        self.assertFalse(self.eng.gone_alerted)

    def test_unreadable_camera_gives_no_verdict_and_no_alarm(self):
        self.eng.capture.fail = True

        async def go():
            await self.feed(True, 20)
            await self.feed(False, 8)
            for _ in range(10):
                await self._orig_sleep(0)
            self.eng._gone_task and self.eng._gone_task.cancel()
        asyncio.run(go())
        self.assertEqual([t for t in self.texts() if "ARAÇ GİTTİ" in t], [])


class DepartureTest(Base):
    def test_paused_owner_absent_90s_hands_over_to_spot_watch_without_an_alarm(self):
        self.ctl.pause(3600)

        async def go():
            await self.feed(True, 20)
            self.car_visible = False
            await self.feed(False, 8)
            for _ in range(500):                  # real threads are involved: poll in real time
                await self._orig_sleep(0.01)
                if self.departures:
                    break
        asyncio.run(go())
        self.assertEqual(self.departures, ["pause"])
        self.assertEqual([t for t in self.texts() if "ARAÇ GİTTİ" in t], [])
        self.assertGreaterEqual(self.clk.t - 1_000_000.0, 28 + 90 - 15)   # not before ~90 s absent

    def test_paused_but_back_before_90s_cancels_the_departure(self):
        self.ctl.pause(3600)

        async def go():
            await self.feed(True, 20)
            self.car_visible = False
            await self.feed(False, 8)
            for _ in range(5):
                await self._orig_sleep(0)
            self.car_visible = True                # car (or its detection) is back
            await self.settle()
        asyncio.run(go())
        self.assertEqual(self.departures, [])

    def test_owner_departure_button_hands_over_and_clears_the_alarm(self):
        async def go():
            await self.feed(True, 20)
            self.car_visible = False
            await self.feed(False, 8)
            await asyncio.wait_for(DetectionTest._wait_alert(self), 5)
            ok = await self.eng.owner_departure()
            await self._orig_sleep(0)
            return ok
        self.assertTrue(asyncio.run(go()))
        self.assertEqual(self.departures, ["owner"])
        self.assertIsNone(self.eng.active_alert)


class EscalationTest(Base):
    def alarm(self):
        async def go():
            await self.feed(True, 20)
            self.car_visible = False
            await self.feed(False, 8)
            await asyncio.wait_for(DetectionTest._wait_alert(self), 5)
        asyncio.run(go())

    def run_repeat(self, seconds):
        async def go():
            t = asyncio.get_running_loop().create_task(self.eng._repeat_loop())
            for _ in range(seconds):
                await asyncio.sleep(1.0)
            t.cancel()
        asyncio.run(go())

    def test_unanswered_alarm_repeats_louder_and_faster_after_the_reply_window(self):
        self.alarm()
        n0 = len(self.n.sent)
        self.run_repeat(200)
        rep = [s for s in self.n.sent[n0:] if "tekrar" in s["text"]]
        self.assertGreaterEqual(len(rep), 5)
        self.assertTrue(any("ACİL" in s["text"] for s in rep))
        self.assertTrue(all(s["buttons"] for s in rep))               # [Me]/[Not me] stay on repeats
        self.assertFalse(any("ACİL" in s["text"] for s in rep[:2]))   # louder only after ~2 min

    def test_not_me_escalates_immediately(self):
        self.alarm()
        n0 = len(self.n.sent)
        self.assertTrue(self.eng.escalate())
        self.run_repeat(20)
        rep = [s for s in self.n.sent[n0:] if "tekrar" in s["text"]]
        self.assertTrue(rep and "ACİL" in rep[0]["text"])

    def test_ack_stops_the_repeats(self):
        self.alarm()
        self.assertTrue(self.eng.ack())
        n0 = len(self.n.sent)
        self.run_repeat(200)
        self.assertEqual([s for s in self.n.sent[n0:] if "tekrar" in s["text"]], [])

    def test_escalate_without_a_gone_alarm_does_nothing(self):
        self.assertFalse(self.eng.escalate())


class TrackerQuietFlagTest(unittest.TestCase):
    def test_first_free_alert_is_swallowed_once_after_the_owner_leaves(self):
        t = SpotTracker("a")
        t.observe(True, 0)
        t.on_enter()
        t.quiet_next_free = True
        t.observe(False, 10)
        self.assertEqual(t.observe(False, 20), [])                    # confirmed free, silently
        self.assertTrue(t.alerted)
        self.assertFalse(t.quiet_next_free)
        t.observe(True, 30)                                           # someone parks
        t.observe(False, 40)
        self.assertEqual([e.kind for e in t.observe(False, 50)], ["free_alert"])   # normal again

    def test_arrival_flag_set_only_on_a_confirmed_free_to_occupied_flip(self):
        t = SpotTracker("a")
        t.observe(True, 0)
        self.assertFalse(t.just_arrived)                              # baseline
        t.observe(False, 10); t.observe(False, 20)                    # confirmed free
        t.observe(True, 30)
        self.assertTrue(t.just_arrived)


if __name__ == "__main__":
    unittest.main()
