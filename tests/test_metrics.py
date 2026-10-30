"""Prometheus metrics: instrumentation, scrape-time state, no secrets, never fatal."""

from __future__ import annotations

import asyncio
import socket
import tempfile
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from prometheus_client import generate_latest

from parkingbot.bot import ParkingBot
from parkingbot.config import load_settings
from parkingbot.guard import GuardConfig, GuardEngine
from parkingbot.guard_geometry import Zones
from parkingbot.guard_rules import ALERT, GuardEvent
from parkingbot.guard_tracker import Track
from parkingbot.metrics import NULL, BotMetrics, start_server
from parkingbot.modes import Mode
from parkingbot.spotwatch import Outbox


class FakeJobQueue:
    def run_once(self, cb, when=0, **kw):
        return MagicMock()


def scrape(m: BotMetrics) -> str:
    return generate_latest(m.registry).decode()


def value(text: str, name: str, labels: str = "") -> float | None:
    key = f"{name}{labels} "
    for line in text.splitlines():
        if line.startswith(key):
            return float(line.split()[-1])
    return None


class InstrumentationTest(unittest.TestCase):
    def test_null_metrics_swallow_every_call(self):
        NULL.alert("ALERT", "CONTACT")
        NULL.anything_at_all(1, 2, x=3)

    def test_counters_and_histograms_appear_in_the_scrape(self):
        m = BotMetrics()
        m.alert("WARNING", "LOITER"); m.alert("WARNING", "LOITER"); m.alert("ALERT", "CONTACT")
        m.rule_latency(0.8); m.rule_latency(2.5)
        m.gate("tier2"); m.gate("cross"); m.reconnect(); m.frame("dropped", 3)
        m.limiter_change("up"); m.grab_failure(3); m.send_failure(); m.spot_alert("free_alert")
        t = scrape(m)
        self.assertEqual(value(t, "parking_guard_alerts_total", '{rule="LOITER",severity="WARNING"}'), 2.0)
        self.assertEqual(value(t, "parking_guard_alerts_total", '{rule="CONTACT",severity="ALERT"}'), 1.0)
        self.assertEqual(value(t, "parking_guard_rule_latency_seconds_count"), 2.0)
        self.assertEqual(value(t, "parking_guard_rule_latency_seconds_bucket", '{le="1.0"}'), 1.0)
        self.assertEqual(value(t, "parking_spotwatch_gate_overrides_total", '{gate="tier2"}'), 1.0)
        self.assertEqual(value(t, "parking_guard_frames_total", '{outcome="dropped"}'), 3.0)
        self.assertEqual(value(t, "parking_camera_grab_failures_total", '{channel="3"}'), 1.0)
        self.assertEqual(value(t, "parking_telegram_send_failures_total"), 1.0)
        self.assertIsNotNone(value(t, "process_resident_memory_bytes"))

    def test_outbox_counts_failed_sends(self):
        m = BotMetrics()

        class Down:
            async def send_text(self, *a, **k):
                raise RuntimeError("down")
        ob = Outbox(Down(), metrics=m)
        asyncio.run(ob.send("x"))
        self.assertEqual(value(scrape(m), "parking_telegram_send_failures_total"), 1.0)


class StateScrapeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        s = load_settings()
        s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True, "state_db": str(Path(self._tmp.name) / "s.db")}
        s.raw["guard"] = {**s.raw.get("guard", {}), "enabled": True}
        self.settings = s
        self.bot = ParkingBot(s)
        self.bot.init_spotwatch(MagicMock(), FakeJobQueue())
        self.bot.controller.verify()
        self.bot.metrics.attach(self.bot)

    def tearDown(self):
        self._tmp.cleanup()

    def test_mode_spots_cameras_and_references(self):
        sp = self.settings.spots[0].name
        self.bot.auto.matcher.store.add(sp, "night", [0.0] * 4, 1)
        self.bot.auto.matcher.store.add(sp, "night", [0.0] * 4, 0)
        t = scrape(self.bot.metrics)
        self.assertEqual(value(t, "parking_mode", '{mode="SPOT_WATCH"}'), 1.0)
        self.assertEqual(value(t, "parking_mode", '{mode="GUARD"}'), 0.0)
        self.assertEqual(value(t, "parking_auto_enabled"), 0.0)
        self.assertEqual(value(t, "parking_spot_state", f'{{spot="{sp}"}}'), -1.0)      # nothing read yet
        self.assertEqual(value(t, "parking_car_references", f'{{condition="night",label="owner",spot="{sp}"}}'), 1.0)
        self.assertEqual(value(t, "parking_car_references", f'{{condition="night",label="other",spot="{sp}"}}'), 1.0)
        self.assertIsNotNone(value(t, "parking_camera_consecutive_failures", '{channel="1"}'))
        self.assertIsNotNone(value(t, "parking_host_mem_available_bytes"))
        self.assertIsNotNone(value(t, "parking_cpu_seconds_total"))
        self.assertEqual(value(t, "parking_guard_armed", f'{{spot="{sp}"}}'), 0.0)
        self.assertIsNone(value(t, "parking_guard_frame_age_seconds"))                  # only while armed

    def test_spot_state_follows_the_tracker(self):
        sw = self.bot.spotwatch
        sp = self.settings.spots[0].name
        sw.trackers[sp].observe(True, 0)
        self.assertEqual(value(scrape(self.bot.metrics), "parking_spot_state", f'{{spot="{sp}"}}'), 1.0)
        sw.trackers[sp].observe(False, 1); sw.trackers[sp].observe(False, 2)
        t = scrape(self.bot.metrics)
        self.assertEqual(value(t, "parking_spot_state", f'{{spot="{sp}"}}'), 0.0)
        self.assertEqual(value(t, "parking_spot_alerted", f'{{spot="{sp}"}}'), 1.0)

    def test_guard_gauges_appear_only_while_armed(self):
        g = self.bot.guard
        g.zones = Zones((352, 288), (100.0, 100.0, 200.0, 180.0), [(70, 80), (230, 80), (230, 200), (70, 200)])
        g._init_state(self.settings.spots[0])
        g.on_frame(np.full((288, 352, 3), 90, np.uint8), __import__("time").time())
        t = scrape(self.bot.metrics)
        self.assertEqual(value(t, "parking_guard_armed", f'{{spot="{self.settings.spots[0].name}"}}'), 1.0)
        self.assertLess(value(t, "parking_guard_frame_age_seconds"), 5.0)
        self.assertEqual(value(t, "parking_guard_limiter_step"), 0.0)
        self.assertEqual(value(t, "parking_guard_evidence_frames"), 1.0)
        self.assertEqual(value(t, "parking_guard_car_gone_alarm"), 0.0)

    def test_one_failing_section_does_not_break_the_scrape(self):
        self.bot.guard = MagicMock()
        self.bot.guard.armed = True
        self.bot.guard.spot = None                    # makes the guard section blow up
        self.bot.guard.last_frame_ts = "boom"
        t = scrape(self.bot.metrics)
        self.assertEqual(value(t, "parking_mode", '{mode="SPOT_WATCH"}'), 1.0)             # others fine

    def test_scrape_contains_no_secrets_and_is_plain_text(self):
        t = scrape(self.bot.metrics)
        sec = self.settings.secrets
        for secret in (sec.bot_token, sec.rtsp_pass, str(sec.allowed_chat_id), sec.rtsp_user + ":"):
            self.assertNotIn(secret, t)
        t.encode("ascii", "ignore")
        self.assertNotIn("rtsp://", t)
        self.assertNotIn("\xff\xd8", t)               # no JPEG bytes

    def test_auto_switch_outcomes_are_exported(self):
        self.bot.auto.outcomes["auto"] = 2
        t = scrape(self.bot.metrics)
        self.assertEqual(value(t, "parking_auto_switch_outcomes_total", '{outcome="auto"}'), 2.0)


class EngineMetricsTest(unittest.TestCase):
    def test_guard_counts_frames_dropped_alerts_and_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = load_settings()
            from parkingbot.modes import ModeController, StateStore
            ctl = ModeController(StateStore(Path(tmp) / "s.db"), guard_available=True); ctl.verify()
            sent = []

            class N:
                async def send_text(self, text, silent=False, **kw): sent.append(text)
                async def send_photo(self, *a, **k): sent.append("p")
                async def send_media_group(self, *a, **k): sent.append("g")
            m = BotMetrics()
            eng = GuardEngine(s, ctl, None, None, Outbox(N()), cfg=GuardConfig(followup_main=False),
                              guard_detector=MagicMock(), metrics=m)
            eng.zones = Zones((352, 288), (100.0, 100.0, 200.0, 180.0), [(70, 80), (230, 80), (230, 200), (70, 200)])
            eng._init_state(s.spots[0])
            f = np.full((288, 352, 3), 90, np.uint8)
            eng.on_frame(f, 10.0); eng.on_frame(f, 10.1); eng.on_frame(f, 10.2)       # 2 dropped
            ev = GuardEvent("CONTACT", ALERT, eng._clock() - 0.5, 1, 3.0, 1)
            trk = Track(1, 0, "person", (150, 110, 180, 170), 0.0, 3.0)
            asyncio.run(eng._dispatch(ev, f, 10.2, [trk]))
            t = scrape(m)
            self.assertEqual(value(t, "parking_guard_frames_total", '{outcome="received"}'), 3.0)
            self.assertEqual(value(t, "parking_guard_frames_total", '{outcome="dropped"}'), 2.0)
            self.assertEqual(value(t, "parking_guard_alerts_total", '{rule="CONTACT",severity="ALERT"}'), 1.0)
            self.assertEqual(value(t, "parking_guard_rule_latency_seconds_count"), 1.0)
            ctl.pause(60)
            asyncio.run(eng._dispatch(GuardEvent("MULTI", "WARNING", eng._clock(), None, 2.0, 2), f, 10.3, [trk]))
            self.assertEqual(value(scrape(m), "parking_guard_alerts_suppressed_total", '{reason="paused"}'), 1.0)


class ServerTest(unittest.TestCase):
    def _free_port(self):
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            return sk.getsockname()[1]

    def _bot(self, tmp, **metrics):
        s = load_settings()
        s.raw["metrics"] = metrics
        s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True, "state_db": str(Path(tmp) / "s.db")}
        b = ParkingBot(s)
        b.init_spotwatch(MagicMock(), FakeJobQueue())
        b.controller.verify()
        return b, s

    def test_disabled_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, s = self._bot(tmp, enabled=False)
            self.assertIsNone(start_server(b, s))

    def test_serves_metrics_over_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = self._free_port()
            b, s = self._bot(tmp, enabled=True, bind="127.0.0.1", port=port)
            srv = start_server(b, s)
            try:
                self.assertIsNotNone(srv)
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
                self.assertIn("parking_mode", body)
                self.assertIn('parking_mode{mode="SPOT_WATCH"} 1.0', body)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_port_in_use_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as busy:
            busy.bind(("127.0.0.1", 0)); busy.listen()
            b, s = self._bot(tmp, enabled=True, bind="127.0.0.1", port=busy.getsockname()[1])
            self.assertIsNone(start_server(b, s))


class HeartbeatTest(unittest.TestCase):
    def _bot(self, tmp):
        s = load_settings()
        s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True, "state_db": str(Path(tmp) / "s.db")}
        b = ParkingBot(s)
        b.init_spotwatch(MagicMock(), FakeJobQueue())
        return b

    def test_pings_only_while_the_watch_loop_is_fresh(self):
        import os, time
        calls = []

        class Client:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get(self, url, params=None): calls.append((url, params))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HEARTBEAT_URL": "http://kuma/push/abc"}), \
                patch("httpx.AsyncClient", Client):
            b = self._bot(tmp)
            asyncio.run(b.heartbeat_job(None))
            self.assertEqual(calls, [])                                   # no cycle yet: stay silent
            b.spotwatch.last_cycle.ts = time.time()
            asyncio.run(b.heartbeat_job(None))
            self.assertEqual(calls, [("http://kuma/push/abc", {"status": "up", "msg": "OK"})])
            b.spotwatch.last_cycle.ts = time.time() - 3600                # stalled
            asyncio.run(b.heartbeat_job(None))
            self.assertEqual(len(calls), 1)

    def test_no_url_means_no_call(self):
        import os
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HEARTBEAT_URL": ""}):
            b = self._bot(tmp)
            b.spotwatch.last_cycle.ts = __import__("time").time()
            asyncio.run(b.heartbeat_job(None))                            # must not raise


if __name__ == "__main__":
    unittest.main()
