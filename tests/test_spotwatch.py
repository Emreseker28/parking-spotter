"""SpotWatch engine driven by fakes: cadence, debounce, gates, failures, pause, outbox."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from parkingbot.capture import CaptureError
from parkingbot.config import load_settings
from parkingbot.modes import Mode, ModeController, StateStore
from parkingbot.spotwatch import Outbox, SpotWatch, WatchConfig
from parkingbot.verifier import VerifyResult


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeNotifier:
    def __init__(self):
        self.sent: list[tuple[str, str, bool]] = []     # (kind, text, silent)
        self.fail = False

    async def send_text(self, text, silent=False):
        if self.fail:
            raise RuntimeError("telegram down")
        self.sent.append(("text", text, silent))

    async def send_photo(self, jpeg, caption):
        if self.fail:
            raise RuntimeError("telegram down")
        self.sent.append(("photo", caption, False))

    def texts(self):
        return [t for _, t, _ in self.sent]


class FakeCamera:
    """Scripted per-spot occupancy; ch grabs can be made to fail."""

    def __init__(self, settings):
        self.settings = settings
        self.occupied = {s.name: True for s in settings.spots}
        self.fail_channels: set[int] = set()
        self.reads: list[int] = []

    async def __call__(self, ch):
        self.reads.append(ch)
        if ch in self.fail_channels:
            raise CaptureError("ffmpeg timed out")
        return [SimpleNamespace(spot=s, occupied=self.occupied[s.name], coverage=0.0, frame=None)
                for s in self.settings.spots if s.channel == ch]


class FakeStability:
    def __init__(self):
        self.override = False
        self.calls = []

    def check(self, r, confirmed):
        self.calls.append((r.occupied, getattr(r, "coverage", None), confirmed))
        return self.override and not r.occupied


class FakeDetector:
    def __init__(self):
        self.t2_occupied = False
        self.t2_calls = 0

    def check_spot_tier2(self, frame, spot):
        self.t2_calls += 1
        return SimpleNamespace(occupied=self.t2_occupied, coverage=0.8)


class FakeVerifier:
    def __init__(self):
        self.enabled = False
        self.verdict = VerifyResult(ok=True, occupied=False, confidence=0.9, reason="")
        self.calls = 0

    async def verify_spot(self, img, spot):
        self.calls += 1
        return self.verdict


class Harness:
    def __init__(self, **cfg):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.settings = load_settings()
        self.settings.raw["verify"] = {"gate": True, "fail_open": True}
        self.store = StateStore(Path(self.tmp.name) / "s.db")
        self.controller = ModeController(self.store, self.clock)
        self.cam = FakeCamera(self.settings)
        self.stab, self.det, self.ver = FakeStability(), FakeDetector(), FakeVerifier()
        self.notifier = FakeNotifier()
        self.mem = 5000.0
        self.cfg = WatchConfig(send_snapshot=False, **cfg)
        self.build()

    def build(self):
        self.sw = SpotWatch(self.settings, self.controller, self.store, None, self.det,
                            self.ver, self.stab, self.notifier, lambda: self.mem,
                            clock=self.clock, reader=self.cam, cfg=self.cfg)

    def cycle(self, advance=0.0):
        self.clock.t += advance
        return asyncio.run(self.sw.run_cycle())

    def close(self):
        self.tmp.cleanup()


class SpotWatchTest(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.a, self.b = [s.name for s in self.h.settings.spots]

    def tearDown(self):
        self.h.close()

    def test_baseline_occupied_is_silent_and_verifies_mode(self):
        d = self.h.cycle()
        self.assertEqual(self.h.controller.mode, Mode.SPOT_WATCH)
        self.assertEqual(self.h.notifier.sent, [])
        self.assertEqual(d, 60.0)

    def test_spot_freed_alerts_after_confirm_recheck_only(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False
        d = h.cycle(60)
        self.assertEqual(h.notifier.sent, [])                 # one free read: no alert yet
        self.assertEqual(d, 10.0)                             # quick confirm re-check
        h.cycle(10)
        self.assertEqual(len(h.notifier.sent), 1)
        self.assertIn(self.a, h.notifier.texts()[0])
        self.assertIn("BOŞ", h.notifier.texts()[0])
        self.assertEqual(h.sw.last_cycle.kind, "confirm")
        # the confirm cycle only re-read the channel that had the pending spot
        self.assertEqual(h.cam.reads[-1], next(s.channel for s in h.settings.spots if s.name == self.a))
        self.assertEqual(len(h.cam.reads), 2 + 2 + 1)    # 2 full cycles x 2 channels + 1 confirm

    def test_alert_latency_within_75s(self):
        h = self.h
        d = h.cycle(); h.cam.occupied[self.b] = False   # worst case: freed just after a cycle
        t_free = h.clock.t
        d = h.cycle(d)                                  # next full cycle sees it ...
        h.cycle(d)                                      # ... and the confirm re-check alerts
        self.assertTrue(h.notifier.sent)
        self.assertLessEqual(h.clock.t - t_free, 75)

    def test_car_passing_through_does_not_alert(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60)
        h.cam.occupied[self.a] = True                         # gone again at the re-check
        h.cycle(10)
        h.cycle(50); h.cycle(60)
        self.assertEqual(h.notifier.sent, [])

    def test_already_free_on_start_alerts_right_now(self):
        h = self.h
        h.cam.occupied[self.a] = False
        h.cycle(); h.cycle(10)
        self.assertEqual(len(h.notifier.sent), 1)
        self.assertIn("şu anda", h.notifier.texts()[0])

    def test_both_spots_independent(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False; h.cam.occupied[self.b] = False
        h.cycle(60); h.cycle(10)
        names = " ".join(h.notifier.texts())
        self.assertIn(self.a, names); self.assertIn(self.b, names)
        self.assertEqual(len(h.notifier.sent), 2)

    def test_reminders_then_taken_again(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        for _ in range(5):
            h.cycle(600)
        reminders = [t for t in h.notifier.texts() if "hatırlatma" in t]
        self.assertEqual(len(reminders), 3)                   # capped
        h.cam.occupied[self.a] = True
        h.cycle(60)
        self.assertIn("tekrar DOLU", h.notifier.texts()[-1])

    # ------------------------------------------------------------ gates
    def test_stability_gate_blocks_false_free(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False; h.stab.override = True
        h.cycle(60); h.cycle(10); h.cycle(60)
        self.assertEqual(h.notifier.sent, [])

    def test_tier2_gate_blocks_false_free(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False; h.det.t2_occupied = True
        h.cycle(60); h.cycle(10)
        self.assertEqual(h.notifier.sent, [])
        self.assertGreater(h.det.t2_calls, 0)

    def test_tier2_override_seeds_the_stability_reference(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False; h.det.t2_occupied = True
        h.cycle(60)
        # the stability gate saw an occupied/strong-coverage result from tier-2
        self.assertIn((True, 0.8, "occupied"), h.stab.calls)

    def test_tier2_not_run_when_already_confirmed_free(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        calls = h.det.t2_calls
        h.cycle(60)
        self.assertEqual(h.det.t2_calls, calls)               # cheap steady state

    def test_ram_guard_defers_never_alerts(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False; h.mem = 100.0
        for _ in range(4):
            h.cycle(60)
        self.assertEqual(h.notifier.sent, [])
        self.assertEqual(h.det.t2_calls, 0)

    def test_vlm_veto_suppresses_first_alert(self):
        h = self.h
        h.ver.enabled = True
        h.ver.verdict = VerifyResult(ok=True, occupied=True, confidence=0.9, reason="car")
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        self.assertEqual(h.notifier.sent, [])
        self.assertEqual(h.sw.trackers[self.a].confirmed, "occupied")

    def test_vlm_unreachable_fail_open_tags_alert(self):
        h = self.h
        h.ver.enabled = True
        h.ver.verdict = VerifyResult(ok=False, occupied=False, confidence=0.0, reason="timeout")
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        self.assertIn("doğrulanamadı", h.notifier.texts()[0])

    def test_vlm_fail_closed_suppresses(self):
        h = self.h
        h.settings.raw["verify"]["fail_open"] = False
        h.ver.enabled = True
        h.ver.verdict = VerifyResult(ok=False, occupied=False, confidence=0.0, reason="timeout")
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        self.assertEqual(h.notifier.sent, [])

    # --------------------------------------------------------- failures
    def test_camera_unplugged_one_message_never_a_false_free(self):
        h = self.h
        h.cycle()                                              # baseline occupied
        h.cam.fail_channels = {1, 3}
        h.cam.occupied = {n: False for n in h.cam.occupied}   # would read free if it could
        for _ in range(8):
            h.cycle(60)
        texts = h.notifier.texts()
        self.assertEqual(sum("ulaşılamıyor" in t for t in texts), 1)   # ONE message, both cameras
        self.assertTrue(all(h.sw._label(c) in texts[0] for c in (1, 3)))
        self.assertFalse(any("BOŞ" in t for t in texts))
        for t in h.sw.trackers.values():
            self.assertEqual(t.confirmed, "occupied")

    def test_failures_back_off_and_recovery_is_announced(self):
        h = self.h
        h.cycle()
        h.cam.fail_channels = {1}
        for _ in range(3):
            h.cycle(60)
        health = h.sw.health[1]
        self.assertTrue(health.unreachable_sent)
        self.assertGreater(health.next_try, h.clock.t)        # backing off
        reads_before = h.cam.reads.count(1)
        h.cycle(60)                                            # inside the backoff window
        self.assertEqual(h.cam.reads.count(1), reads_before)
        h.cam.fail_channels = set()
        h.cycle(600)
        self.assertEqual(h.sw.health[1].failures, 0)
        self.assertTrue(any("tekrar erişilebilir" in t for t in h.notifier.texts()))

    def test_startup_with_camera_down_stays_unknown(self):
        h = self.h
        h.cam.fail_channels = {1, 3}
        h.cycle()
        self.assertEqual(h.controller.mode, Mode.UNKNOWN)
        self.assertEqual(h.notifier.sent, [])

    # ----------------------------------------------------- modes / pause
    def test_pause_mutes_then_alerts_when_resumed(self):
        h = self.h
        h.cycle(); h.controller.pause(600)
        h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10); h.cycle(60)
        self.assertEqual(h.notifier.sent, [])
        self.assertEqual(h.sw.trackers[self.a].confirmed, "free")   # still tracked
        h.cycle(600)                                           # pause expires
        texts = h.notifier.texts()
        self.assertTrue(any("Duraklatma bitti" in t for t in texts))
        self.assertTrue(any("BOŞ" in t and self.a in t for t in texts))

    def test_restart_does_not_realert_when_state_is_fresh(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        self.assertEqual(len(h.notifier.sent), 1)
        h.controller = ModeController(h.store, h.clock)       # "restart"
        h.build()
        h.cycle(30); h.cycle(60)
        self.assertEqual(len(h.notifier.sent), 1)             # no duplicate

    def test_stale_state_is_ignored_after_restart(self):
        h = self.h
        h.cycle(); h.cam.occupied[self.a] = False
        h.cycle(60); h.cycle(10)
        h.clock.t += 13 * 3600
        h.controller = ModeController(h.store, h.clock)
        h.build()
        self.assertIsNone(h.sw.trackers[self.a].confirmed)

    def test_guard_mode_leaves_cameras_alone(self):
        h = self.h
        h.cycle()
        h.controller.guard_available = True
        h.controller.set_manual(Mode.GUARD, self.a)
        reads = len(h.cam.reads)
        h.cycle(60)
        self.assertEqual(len(h.cam.reads), reads)

    def test_status_text_mentions_mode_spots_and_health(self):
        h = self.h
        h.cycle()
        txt = h.sw.status_text()
        for needle in ("SPOT_WATCH", self.a, self.b, "RTSP", "CPU"):
            self.assertIn(needle, txt)


class OutboxTest(unittest.TestCase):
    def run_(self, coro):
        return asyncio.run(coro)

    def test_failed_send_is_queued_and_retried(self):
        clock, n = Clock(), FakeNotifier()
        ob = Outbox(n, clock)
        n.fail = True
        self.assertFalse(self.run_(ob.send("hi")))
        self.assertEqual(len(ob.queue), 1)
        n.fail = False
        self.run_(ob.flush())
        self.assertEqual(n.texts(), ["hi"])
        self.assertEqual(len(ob.queue), 0)

    def test_images_dropped_after_ttl_text_kept(self):
        clock, n = Clock(), FakeNotifier()
        ob = Outbox(n, clock, image_ttl=300, msg_ttl=1800)
        n.fail = True
        self.run_(ob.send("alert", jpeg=b"\xff\xd8jpegbytes"))
        clock.t += 400
        n.fail = False
        self.run_(ob.flush())
        self.assertEqual(n.sent, [("text", "alert", False)])   # sent without the image

    def test_old_messages_dropped(self):
        clock, n = Clock(), FakeNotifier()
        ob = Outbox(n, clock, msg_ttl=1800)
        n.fail = True
        self.run_(ob.send("old"))
        clock.t += 1900
        n.fail = False
        self.run_(ob.flush())
        self.assertEqual(n.sent, [])

    def test_photo_failure_falls_back_to_text(self):
        clock, n = Clock(), FakeNotifier()

        async def bad_photo(jpeg, caption):
            raise RuntimeError("bad image")
        n.send_photo = bad_photo
        ob = Outbox(n, clock)
        self.assertTrue(self.run_(ob.send("x", jpeg=b"1")))
        self.assertEqual(n.texts(), ["x"])


if __name__ == "__main__":
    unittest.main()
