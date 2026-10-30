"""GuardEngine driven by a scripted detector and a fake clock."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parkingbot.config import load_settings
from parkingbot.guard import GuardConfig, GuardEngine, format_event
from parkingbot.guard_geometry import Zones
from parkingbot.guard_rules import ALERT, INFO, WARNING, GuardEvent
from parkingbot.guard_tracker import Det
from parkingbot.modes import Mode, ModeController, StateStore
from parkingbot.spotwatch import Outbox

CAR = (100.0, 100.0, 200.0, 180.0)
ZONES = Zones((352, 288), CAR, [(70, 80), (230, 80), (230, 200), (70, 200)])
BESIDE = Det(0, "person", 0.8, (78, 120, 93, 170))           # in the perimeter, not on the car
ON_CAR = Det(0, "person", 0.8, (150, 110, 180, 170))         # overlaps the car box
FAR = Det(0, "person", 0.8, (300, 100, 320, 150))            # pavement, outside the perimeter


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


class FakeNotifier:
    def __init__(self):
        self.sent = []          # (kind, text, silent, ack, n_images)

    async def send_text(self, text, silent=False, **kw):
        self.sent.append(("text", text, silent, kw.get("ack", False), 0))

    async def send_photo(self, jpeg, caption, **kw):
        self.sent.append(("photo", caption, False, kw.get("ack", False), 1))

    async def send_media_group(self, jpegs, caption, **kw):
        self.sent.append(("group", caption, False, False, len(jpegs)))


class FakeGDet:
    def __init__(self):
        self.dets = []
        self.calls = 0

    def ensure(self):
        pass

    def detect(self, frame, window, conf=None):
        self.calls += 1
        return list(self.dets)


def blank():
    return np.full((288, 352, 3), 90, np.uint8)


class EngineBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings()
        self.clk = Clock()
        self.store = StateStore(Path(self._tmp.name) / "s.db")
        self.ctl = ModeController(self.store, self.clk, guard_available=True)
        self.ctl.verify()
        self.notifier = FakeNotifier()
        self.outbox = Outbox(self.notifier, self.clk)
        self.gdet = FakeGDet()
        self.cfg = GuardConfig(followup_main=False)
        self.eng = GuardEngine(self.settings, self.ctl, None, None, self.outbox, clock=self.clk,
                               cfg=self.cfg, guard_detector=self.gdet)
        self.eng.zones = ZONES
        self.eng._init_state(self.settings.spots[0])
        self.ctl.set_manual(Mode.GUARD, self.settings.spots[0].name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_(self, coro):
        return asyncio.run(coro)

    def feed(self, dets, start, end, dt=0.3, motion=True):
        """Process one frame every dt seconds with the scripted detections."""
        self.gdet.dets = dets
        out = []

        async def go():
            t = start
            while t <= end + 1e-9:
                self.clk.t = 1_000_000.0 + t
                if motion:
                    self.eng.last_motion_ts = self.clk.t
                out.extend(await self.eng.process(blank(), self.clk.t))
                t = round(t + dt, 3)
        self.run_(go())
        return out

    def texts(self):
        return [s[1] for s in self.notifier.sent]


class RuleToAlertTest(EngineBase):
    def test_loiter_sends_a_turkish_warning_with_severity_and_evidence(self):
        ev = self.feed([BESIDE], 0, 14)
        self.assertEqual([e.rule for e in ev], ["LOITER"])
        kind, text, silent, ack, n = self.notifier.sent[0]
        self.assertIn("UYARI", text)
        self.assertIn("OYALANMA", text)
        self.assertIn(self.settings.spots[0].name, text)
        self.assertFalse(silent)
        self.assertEqual(self.eng.alerts[WARNING], 1)

    def test_contact_is_an_alarm(self):
        self.feed([ON_CAR], 0, 6)
        text = self.notifier.sent[0][1]
        self.assertIn("ALARM", text)
        self.assertIn("TEMAS", text)
        self.assertEqual(self.eng.alerts[ALERT], 1)

    def test_pedestrian_far_on_the_street_causes_nothing(self):
        self.feed([FAR], 0, 40)
        self.assertEqual(self.notifier.sent, [])

    def test_rule_latency_is_recorded_and_small(self):
        self.feed([BESIDE], 0, 14)
        self.assertEqual(len(self.eng.latencies), 1)
        self.assertLess(self.eng.latencies[0], 0.5)       # processing adds ~one frame interval
        self.assertIn("medyan", self.eng.status_text())

    def test_cooldown_blocks_a_second_alert_of_the_same_rule(self):
        self.feed([BESIDE], 0, 14)                        # LOITER #1
        self.feed([], 14.3, 20.0)                         # person leaves; tracks age out
        self.feed([BESIDE], 24, 40)                       # comes back: a new LOITER within 60 s
        self.assertEqual(sum("OYALANMA" in t for t in self.texts()), 1)
        self.assertEqual(self.eng.alerts[WARNING], 2)     # counted, but not re-sent

    def test_pause_suppresses_alerts_but_keeps_tracking(self):
        self.ctl.pause(600)
        self.feed([ON_CAR], 0, 6)
        self.assertEqual(self.notifier.sent, [])
        self.assertEqual(self.eng.suppressed, 1)
        self.assertTrue(self.eng.tracker.has_tracks())

    def test_night_uses_shorter_loiter_threshold(self):
        self.eng._night, self.eng._night_checked = True, 1_000_000.0 + 1e6   # force night
        ev = self.feed([BESIDE], 0, 8)
        self.assertEqual([e.rule for e in ev], ["LOITER"])


class LoadControlTest(EngineBase):
    def test_idle_scene_never_runs_yolo(self):
        self.feed([], 0, 30, motion=False)
        self.assertEqual(self.gdet.calls, 0)

    def test_yolo_rate_is_capped_by_the_limiter_fps(self):
        self.feed([], 0, 10, dt=0.05)                     # 20 fps offered, 4 fps allowed
        self.assertLessEqual(self.gdet.calls, 4 * 10 + 2)
        self.assertGreater(self.eng.skipped, 0)

    def test_limiter_step_is_logged_and_lowers_the_rate(self):
        for k in range(60):
            self.eng.limiter.record(1.0, 1_000_000.0 + k)  # 100 % busy
        self.clk.t = 1_000_060.0
        self.eng.last_motion_ts = self.clk.t
        self.run_(self.eng.process(blank(), self.clk.t))
        self.assertEqual(self.eng.limiter.step, 1)
        self.assertTrue(any("limiter" in t and "BİLGİ" in t for t in self.texts()))
        self.assertTrue(self.notifier.sent[-1][2])        # INFO is silent
        self.assertEqual(self.eng.gate.scale, 1.5)

    def test_tail_keeps_detecting_then_goes_idle(self):
        self.eng.last_motion_ts = 1_000_000.0
        self.assertTrue(self.eng._active(1_000_009.0))
        self.assertFalse(self.eng._active(1_000_011.0))

    def test_live_track_keeps_it_active_so_a_still_person_is_not_lost(self):
        self.feed([BESIDE], 0, 3)
        self.assertTrue(self.eng.tracker.has_tracks())
        self.assertTrue(self.eng._active(1_000_000.0 + 60))    # no motion for 60 s, track alive


class FramePathTest(EngineBase):
    def test_latest_frame_wins_and_drops_are_counted(self):
        f1, f2, f3 = blank(), blank(), blank()
        self.eng.on_frame(f1, 10.0)
        self.eng.on_frame(f2, 10.1)
        self.eng.on_frame(f3, 10.2)
        self.assertIs(self.eng._slot[1], f3)
        self.assertEqual(self.eng.dropped, 2)
        self.assertEqual(self.eng.frames, 3)

    def test_evidence_buffer_is_a_bounded_ram_ring(self):
        for i in range(100):
            self.eng.on_frame(blank(), 10.0 + i * 0.15)       # 15 s of frames
        span = self.eng.evidence[-1][0] - self.eng.evidence[0][0]
        self.assertLessEqual(span, self.cfg.evidence_s + 0.01)
        self.assertLess(len(self.eng.evidence), 60)

    def test_on_frame_gate_sets_motion_and_ignores_global_change(self):
        self.eng.on_frame(blank(), 10.0)
        moving = blank(); moving[110:170, 120:150] = 235
        self.eng.on_frame(moving, 10.2)
        self.assertGreaterEqual(self.eng.last_motion_ts, 10.2)
        self.eng.last_motion_ts = 0.0
        self.eng.on_frame(np.full((288, 352, 3), 200, np.uint8), 20.0)   # headlight sweep
        self.assertEqual(self.eng.last_motion_ts, 0.0)
        self.assertEqual(self.eng.global_rejects, 1)

    def test_evidence_group_has_key_frame_plus_up_to_three_previous(self):
        for i in range(30):
            self.eng.on_frame(blank(), 100.0 + i * 0.15)
        jpegs = self.eng._build_evidence(blank(), 104.5, [])
        self.assertEqual(len(jpegs), 4)
        self.assertTrue(all(j[:2] == b"\xff\xd8" for j in jpegs))     # real JPEGs, in RAM


class AckAndRepeatTest(EngineBase):
    def test_alert_repeats_every_30s_until_ack(self):
        self.feed([ON_CAR], 0, 6)
        a = self.eng.active_alert
        self.assertIsNotNone(a)
        self.assertTrue(a.repeats == 0)
        n0 = len(self.notifier.sent)

        async def loop_for(seconds):
            task = asyncio.get_running_loop().create_task(self.eng._repeat_loop())
            for _ in range(seconds):
                self.clk.t += 1.0
                await asyncio.sleep(0)
                await asyncio.sleep(0.001)
            task.cancel()
        # fast-forward the fake clock: sleep(1.0) in the loop is patched to be instant
        orig = asyncio.sleep

        async def fast(d, *a, **k):
            await orig(0)
        asyncio.sleep = fast
        try:
            self.run_(loop_for(65))
        finally:
            asyncio.sleep = orig
        repeats = [s for s in self.notifier.sent[n0:] if "tekrar" in s[1]]
        self.assertEqual(len(repeats), 2)                          # at +30 s and +60 s
        self.assertTrue(all(s[3] for s in repeats))                # carry the ack button
        self.assertTrue(self.eng.ack())
        self.assertFalse(self.eng.ack())                           # nothing left to ack

    def test_ack_without_alert(self):
        self.assertFalse(self.eng.ack())


class FormatTest(unittest.TestCase):
    def test_every_severity_has_a_turkish_label(self):
        for sev, word in ((INFO, "BİLGİ"), (WARNING, "UYARI"), (ALERT, "ALARM")):
            t = format_event(GuardEvent("LOITER", sev, 0.0, 1, 12.0, 1), "Spot")
            self.assertIn(word, t)
        self.assertIn("2 kişi", format_event(GuardEvent("MULTI", WARNING, 0.0, None, 3.0, 2), "Spot"))


class ReaderTest(EngineBase):
    def test_reconnects_with_backoff_after_a_dropped_stream(self):
        w, h = self.cfg.sub_size
        frame_bytes = bytes(w * h * 3)
        spawned = []

        class Proc:
            returncode = None

            def __init__(self, n):
                self.n = n
                self.stdout = self

            async def readexactly(self, nbytes):
                if self.n <= 0:
                    raise asyncio.IncompleteReadError(b"", nbytes)
                self.n -= 1
                return frame_bytes

            def kill(self):
                self.returncode = -9

            async def wait(self):
                return 0

        async def factory(channel):
            spawned.append(channel)
            return Proc(3)

        self.eng._source_factory = factory
        self.eng.spot = self.settings.spots[0]
        orig = asyncio.sleep
        sleeps = []

        async def fake_sleep(d, *a, **k):
            sleeps.append(d)
            await orig(0)
        asyncio.sleep = fake_sleep

        async def go():
            t = asyncio.get_running_loop().create_task(self.eng._source_loop())
            for _ in range(200):
                await orig(0)
                if len(spawned) >= 3:
                    break
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        try:
            self.run_(go())
        finally:
            asyncio.sleep = orig
        self.assertGreaterEqual(len(spawned), 3)
        self.assertGreaterEqual(self.eng.reconnects, 2)
        self.assertGreaterEqual(self.eng.frames, 6)
        self.assertEqual(sleeps[:2], [1.0, 1.0])     # backoff resets after frames flow again


class ArmTest(EngineBase):
    def _arm(self, boxes_per_read):
        calls = iter(boxes_per_read)

        class Cap:
            async def grab(self, ch, sub, expected_size=None):
                return b"jpeg"
        self.eng.capture = Cap()
        self.eng._pick_car = lambda jpeg, spot: next(calls, None)
        self.cfg.arm_gap_s = 0.0
        return self.run_(self.eng.arm(0))

    def test_arm_builds_zones_from_median_and_starts_tasks(self):
        async def go():
            calls = iter([(285, 425, 510, 815)] * 4 + [None])

            class Cap:
                async def grab(self, ch, sub, expected_size=None):
                    return b"jpeg"
            self.eng.capture = Cap()
            self.eng._pick_car = lambda jpeg, spot: next(calls, None)
            self.cfg.arm_gap_s = 0.0
            # no real ffmpeg / XVR: stub the source and event tasks
            async def idle(*a, **k):
                await asyncio.sleep(3600)
            self.eng._source_loop = idle
            self.eng._repeat_loop = idle
            self.cfg.gate = "motion"
            ok, msg = await self.eng.arm(0)
            armed = (self.eng.armed, len(self.eng._tasks))
            await self.eng.disarm("test")
            return ok, msg, armed
        ok, msg, armed = self.run_(go())
        self.assertTrue(ok, msg)
        self.assertIn("GUARD aktif", msg)
        self.assertEqual(armed, (True, 3))
        self.assertFalse(self.eng.armed)
        self.assertEqual(len(self.eng.evidence), 0)

    def test_arm_fails_when_the_car_is_not_seen(self):
        ok, msg = self._arm([None, None, (285, 425, 510, 815), None, None])
        self.assertFalse(ok)
        self.assertIn("net görünmüyor", msg)
        self.assertIn("1/5", msg)
        self.assertEqual(self.eng._tasks, [])

    def test_arm_fails_cleanly_when_model_is_missing(self):
        def boom():
            raise FileNotFoundError("GUARD model not found: /x")
        self.gdet.ensure = boom
        ok, msg = self._arm([(285, 425, 510, 815)] * 5)
        self.assertFalse(ok)
        self.assertIn("yüklenemedi", msg)


if __name__ == "__main__":
    unittest.main()
