"""AutoSwitch: enrollment, arrival evaluation, ask / auto / undo / default-GUARD flows."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from parkingbot.autoswitch import AutoConfig, AutoSwitch
from parkingbot.car_store import CarStore
from parkingbot.config import load_settings
from parkingbot.matcher import MatchConfig, Matcher
from parkingbot.modes import Mode, ModeController, StateStore
from parkingbot.spotwatch import Outbox

BOX = (100, 150, 520, 370)          # inside the 'ev önü' polygon (ch3, 960x576)


def frame(kind, seed=0, gray=False):
    rng = np.random.default_rng(seed)
    img = np.full((576, 960, 3), (90, 70, 50), np.uint8)       # colourful background = daylight
    car = img[150:370, 100:520]
    if kind == "owner":
        car[:] = (205, 200, 195)
        for x in range(0, 420, 60):
            cv2.rectangle(car, (x, 40), (x + 25, 110), (120, 120, 125), -1)
        cv2.rectangle(car, (30, 150), (390, 180), (90, 90, 95), -1)
    else:
        car[:] = (40, 45, 150)
        for i in range(10):
            cv2.circle(car, (40 + i * 38, 60 + (i % 3) * 40), 14, (200, 220, 240), -1)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    if gray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.merge([g, g, g])
    return img


class FakeCapture:
    def __init__(self):
        self.script = []              # list of (kind, gray) or None per grab; last repeats
        self.i = 0
        self.fail = False

    async def grab(self, ch, sub, expected_size=None):
        if self.fail:
            raise RuntimeError("no camera")
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item


class FakeDetector:
    """decode() just returns the frame we smuggled in as 'jpeg'."""
    box = BOX
    visible = True

    def decode(self, jpeg):
        return jpeg

    def detect_array(self, img):
        return []

    def check_spot(self, img, spot, full):
        det = SimpleNamespace(box=self.box)
        return SimpleNamespace(occupied=self.visible, coverage=0.8, detections=[det] if self.visible else [])

    def check_spot_tier2(self, img, spot):
        return None


class FakeGuard:
    def __init__(self):
        self.armed, self.disarmed, self.ok = [], [], True

    async def arm(self, idx):
        self.armed.append(idx)
        return (True, "ok") if self.ok else (False, "araç net görünmüyor")

    async def disarm(self, reason=""):
        self.disarmed.append(reason)


class FakeNotifier:
    def __init__(self):
        self.sent = []

    async def send_text(self, text, silent=False, **kw):
        self.sent.append({"text": text, "silent": silent, **kw})

    async def send_photo(self, jpeg, caption, **kw):
        self.sent.append({"text": caption, **kw})

    async def send_media_group(self, jpegs, caption, **kw):
        self.sent.append({"text": caption, **kw})


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings()
        self.store = StateStore(Path(self._tmp.name) / "s.db")
        self.ctl = ModeController(self.store, guard_available=True)
        self.ctl.verify()
        self.ctl.set_auto(True)
        self.spot_idx = next(i for i, s in enumerate(self.settings.spots) if s.channel == 3)
        self.spot = self.settings.spots[self.spot_idx]
        self.cs = CarStore(self.store.db)
        self.matcher = Matcher(self.cs, MatchConfig())
        self.cap, self.det, self.guard, self.notifier = FakeCapture(), FakeDetector(), FakeGuard(), FakeNotifier()
        self.cfg = AutoConfig(settle_s=0, read_gap_s=0, enroll_gap_s=0, ask_timeout_s=0.2, undo_window_s=5)
        self.auto = AutoSwitch(self.settings, self.ctl, self.matcher, self.det, self.cap,
                               self.guard, Outbox(self.notifier), cfg=self.cfg)

    def tearDown(self):
        self._tmp.cleanup()

    def run_(self, coro):
        return asyncio.run(coro)

    def enroll_owner(self, n=8, gray=False):
        for i in range(n):
            self.cs.add(self.spot.name, "night" if gray else "day",
                        __import__("parkingbot.appearance", fromlist=["x"]).features(
                            frame("owner", i, gray)[150:370, 100:520], BOX, (576, 960)), 1)

    async def arrive(self, kind, gray=False, wait=0.6):
        self.cap.script = [frame(kind, 7, gray)]
        self.cap.i = 0
        await self.auto.on_arrival(self.spot.name)
        await asyncio.sleep(wait)


class EnrollTest(Base):
    def test_enroll_stores_vectors_for_the_spot_and_condition(self):
        self.cap.script = [frame("owner", 1)]
        ok, msg = self.run_(self.auto.enroll(self.spot_idx))
        self.assertTrue(ok, msg)
        c = self.cs.counts(self.spot.name)
        self.assertEqual(c["day"]["pos"], 15)
        self.assertIn("görüntü yok", msg)
        self.assertEqual(self.cs.counts("başka")["day"]["pos"], 0)

    def test_enroll_at_night_is_tagged_night_and_rain_is_tagged_rain(self):
        self.cap.script = [frame("owner", 1, gray=True)]
        self.run_(self.auto.enroll(self.spot_idx))
        self.assertEqual(self.cs.counts(self.spot.name)["night"]["pos"], 15)
        self.cap.script = [frame("owner", 2)]
        self.run_(self.auto.enroll(self.spot_idx, rain=True))
        self.assertEqual(self.cs.counts(self.spot.name)["rain"]["pos"], 15)

    def test_failed_enrollment_stores_nothing(self):
        self.det.visible = False                                      # car not in the spot
        self.cap.script = [frame("owner", 1)]
        ok, msg = self.run_(self.auto.enroll(self.spot_idx))
        self.assertFalse(ok)
        self.assertIn("Hiçbir şey kaydedilmedi", msg)
        self.assertEqual(self.cs.refs(self.spot.name), [])

    def test_implausible_box_is_not_enrolled(self):
        self.det.box = (300, 100, 340, 500)                           # a thin sliver
        self.cap.script = [frame("owner", 1)]
        ok, _ = self.run_(self.auto.enroll(self.spot_idx))
        self.assertFalse(ok)
        self.assertEqual(self.cs.refs(self.spot.name), [])


class ArrivalTest(Base):
    def test_does_nothing_when_auto_is_off_muted_or_not_watching(self):
        self.enroll_owner()

        async def go():
            self.ctl.set_auto(False)
            await self.arrive("owner")
            self.ctl.set_auto(True)
            self.ctl.pause(600)
            await self.arrive("owner")
            self.ctl.resume()
            self.ctl.auto_enter_guard("x")
            await self.arrive("owner")
        self.run_(go())
        self.assertEqual((self.guard.armed, self.notifier.sent), ([], []))

    def test_recognised_car_switches_to_guard_keeps_auto_and_offers_undo(self):
        self.enroll_owner()
        self.run_(self.arrive("owner"))
        self.assertEqual(self.guard.armed, [self.spot_idx])
        self.assertEqual(self.ctl.mode, Mode.GUARD)
        self.assertTrue(self.ctl.auto)                                # automatic stays automatic
        m = self.notifier.sent[-1]
        self.assertIn("tanıdım", m["text"])
        self.assertEqual(m["buttons"][0][0][0], "Bu benim arabam değil")
        self.assertEqual(self.auto.outcomes["auto"], 1)

    def test_another_car_is_ignored_silently(self):
        self.enroll_owner()
        self.run_(self.arrive("other"))
        self.assertEqual((self.guard.armed, self.notifier.sent), ([], []))
        self.assertEqual(self.ctl.mode, Mode.SPOT_WATCH)
        self.assertEqual(self.auto.outcomes["other"], 1)

    def test_without_enough_references_it_asks_instead_of_switching(self):
        self.enroll_owner(n=2)
        self.run_(self.arrive("owner", wait=0.05))
        self.assertEqual(self.guard.armed, [])
        m = self.notifier.sent[-1]
        self.assertIn("Bu senin araban mı", m["text"])
        labels = [b[0] for b in m["buttons"][0]]
        self.assertEqual(labels, ["Evet, benim", "Hayır"])
        self.assertEqual(len(self.auto.pending), 1)

    def test_car_that_leaves_before_settling_aborts(self):
        self.enroll_owner()
        self.det.visible = False
        self.run_(self.arrive("owner"))
        self.assertEqual((self.guard.armed, self.notifier.sent), ([], []))
        self.assertEqual(self.auto.outcomes["aborted"], 1)

    def test_owner_taking_over_manually_cancels_the_decision(self):
        self.enroll_owner()
        self.cfg.settle_s = 0.2

        async def go():
            self.cap.script = [frame("owner", 7)]
            await self.auto.on_arrival(self.spot.name)
            self.ctl.set_manual(Mode.SPOT_WATCH)                       # /watch -> auto off
            await asyncio.sleep(0.6)
        self.run_(go())
        self.assertEqual(self.guard.armed, [])

    def test_one_evaluation_per_spot_at_a_time(self):
        self.enroll_owner()
        self.cfg.settle_s = 0.3

        async def go():
            self.cap.script = [frame("owner", 7)]
            await self.auto.on_arrival(self.spot.name)
            await self.auto.on_arrival(self.spot.name)
            await asyncio.sleep(0.8)
        self.run_(go())
        self.assertEqual(self.guard.armed, [self.spot_idx])


class AnswerTest(Base):
    def ask(self):
        self.enroll_owner(n=2)

        async def go():
            await self.arrive("owner", wait=0.05)
        self.run_(go())
        return next(iter(self.auto.pending))

    def test_yes_learns_a_positive_and_arms_guard(self):
        pid = self.ask()
        before = self.cs.counts(self.spot.name)["day"]["pos"]
        msg = self.run_(self.auto.answer(pid, True))
        self.assertIn("GUARD aktif", msg)
        self.assertEqual(self.cs.counts(self.spot.name)["day"]["pos"], before + 1)
        self.assertEqual(self.ctl.mode, Mode.GUARD)
        self.assertTrue(self.ctl.auto)

    def test_no_learns_a_negative_and_stays_in_spot_watch(self):
        pid = self.ask()
        msg = self.run_(self.auto.answer(pid, False))
        self.assertIn("başka araç", msg)
        self.assertEqual(self.cs.counts(self.spot.name)["day"]["neg"], 1)
        self.assertEqual(self.guard.armed, [])
        self.assertEqual(self.ctl.mode, Mode.SPOT_WATCH)

    def test_stale_or_unknown_answer_is_harmless(self):
        self.assertIn("geçerli değil", self.run_(self.auto.answer("nope", True)))

    def test_unanswered_question_defaults_to_guard_with_a_notice(self):
        self.enroll_owner(n=2)

        async def go():
            await self.arrive("owner", wait=0.05)
            await asyncio.sleep(0.5)                                   # ask_timeout_s = 0.2
        self.run_(go())
        self.assertEqual(self.guard.armed, [self.spot_idx])
        self.assertEqual(self.ctl.mode, Mode.GUARD)
        self.assertIn("varsayılan", self.notifier.sent[-1]["text"])
        self.assertEqual(self.auto.outcomes["default_guard"], 1)
        self.assertEqual(self.auto.pending, {})

    def test_answering_cancels_the_default_timer(self):
        pid = self.ask()

        async def go():
            await self.auto.answer(pid, False)
            await asyncio.sleep(0.5)
        self.run_(go())
        self.assertEqual(self.guard.armed, [])                         # timer did not fire

    def test_undo_after_an_automatic_switch_learns_a_negative_and_returns_to_watch(self):
        self.enroll_owner()
        self.run_(self.arrive("owner"))
        pid = next(iter(self.auto.pending))
        msg = self.run_(self.auto.answer(pid, False))
        self.assertIn("SPOT_WATCH", msg)
        self.assertEqual(self.ctl.mode, Mode.SPOT_WATCH)
        self.assertTrue(self.ctl.auto)
        self.assertEqual(self.guard.disarmed, ["undo"])
        self.assertEqual(self.cs.counts(self.spot.name)["day"]["neg"], 1)

    def test_failed_arming_on_yes_is_reported_not_swallowed(self):
        pid = self.ask()
        self.guard.ok = False
        msg = self.run_(self.auto.answer(pid, True))
        self.assertIn("kurulamadı", msg)
        self.assertEqual(self.ctl.mode, Mode.SPOT_WATCH)


class StatusTest(Base):
    def test_status_lists_references_thresholds_and_outcomes(self):
        self.enroll_owner(n=4)
        t = self.auto.status_text()
        for needle in ("Otomatik mod: AÇIK", "referans", "day 4+", "Sonuçlar"):
            self.assertIn(needle, t)


if __name__ == "__main__":
    unittest.main()
