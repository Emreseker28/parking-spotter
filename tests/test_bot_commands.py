"""Spot Watch Telegram commands + application wiring, with mocked Telegram objects."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from parkingbot.bot import ParkingBot, build_application, parse_duration
from parkingbot.config import load_settings
from parkingbot.modes import Mode


class FakeJobQueue:
    def __init__(self):
        self.jobs = []

    def run_once(self, cb, when=0, **kw):
        job = MagicMock()
        self.jobs.append((cb, when, job))
        return job


def make_update(chat_id):
    sent = []

    async def reply_text(text, **kw):
        sent.append(text)

    msg = SimpleNamespace(reply_text=reply_text)
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), effective_message=msg,
                           callback_query=None), sent


class CommandsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings()
        self.settings.raw["spotwatch"] = {**self.settings.raw.get("spotwatch", {}),
                                          "enabled": True,
                                          "state_db": str(Path(self._tmp.name) / "s.db")}
        self.bot = ParkingBot(self.settings)
        self.jq = FakeJobQueue()
        self.bot.init_spotwatch(MagicMock(), self.jq)
        self.chat = self.settings.secrets.allowed_chat_id

    def tearDown(self):
        self._tmp.cleanup()

    def call(self, fn, *args, chat=None):
        upd, sent = make_update(self.chat if chat is None else chat)
        asyncio.run(fn(upd, SimpleNamespace(args=list(args))))
        return sent

    def test_unauthorized_chat_is_rejected(self):
        sent = self.call(self.bot.cmd_status, chat=self.chat + 1)
        self.assertEqual(sent, ["⛔ Not authorized."])
        self.assertEqual(self.call(self.bot.cmd_watch, chat=self.chat + 1), ["⛔ Not authorized."])

    def test_status_shows_mode(self):
        self.assertIn("UNKNOWN", self.call(self.bot.cmd_status)[0])

    def test_watch_sets_manual_and_kicks_a_single_job(self):
        self.bot.controller.verify()
        self.bot.controller.set_auto(True)
        self.call(self.bot.cmd_watch)
        self.assertFalse(self.bot.controller.auto)
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)
        self.call(self.bot.cmd_watch)
        # the previously scheduled job was removed before scheduling the next
        self.jq.jobs[-2][2].schedule_removal.assert_called()

    def test_guard_is_not_available_yet(self):
        self.bot.controller.verify()
        sent = self.call(self.bot.cmd_guard)
        self.assertIn("henüz yok", sent[0])
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)

    def test_pause_and_resume(self):
        self.bot.controller.verify()
        self.assertIn("10 dk", self.call(self.bot.cmd_pause, "10m")[0])
        self.assertEqual(self.bot.controller.mode, Mode.PAUSED)
        self.assertIn("Devam", self.call(self.bot.cmd_resume)[0])
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)
        self.assertIn("Duraklatılmış bir şey yok", self.call(self.bot.cmd_resume)[0])

    def test_pause_bad_argument_and_before_verify(self):
        self.assertIn("Şu an duraklatılamaz", self.call(self.bot.cmd_pause, "10m")[0])
        self.bot.controller.verify()
        self.assertIn("Kullanım", self.call(self.bot.cmd_pause, "banana")[0])
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)

    def test_auto_flag_is_saved_with_phase3_note(self):
        self.bot.controller.verify()
        self.assertIn("Faz 3", self.call(self.bot.cmd_auto)[0])
        self.assertTrue(self.bot.controller.auto)

    def test_commands_report_disabled_when_spotwatch_off(self):
        off = ParkingBot(self.settings)
        sent = []
        upd, sent = make_update(self.chat)
        asyncio.run(off.cmd_status(upd, SimpleNamespace(args=[])))
        self.assertIn("kapalı", sent[0])


class DurationTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_duration("10m"), 600)
        self.assertEqual(parse_duration("90s"), 90)
        self.assertEqual(parse_duration("2h"), 7200)
        self.assertEqual(parse_duration("15"), 900)
        self.assertEqual(parse_duration(""), 600)
        self.assertIsNone(parse_duration("0m"))
        self.assertIsNone(parse_duration("48h"))
        self.assertIsNone(parse_duration("abc"))


class WiringTest(unittest.TestCase):
    def test_build_application_replaces_push_loop_with_spotwatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = load_settings()
            s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True,
                                  "state_db": str(Path(tmp) / "s.db")}
            s.raw["push"] = {**s.raw.get("push", {}), "enabled": True}
            app = build_application(s)
            names = {j.name for j in app.job_queue.jobs()}
            self.assertTrue(any("spotwatch_job" in n for n in names), names)
            self.assertFalse(any("push_job" in n for n in names), names)
            cmds = {h.commands for g in app.handlers.values() for h in g if hasattr(h, "commands")}
            flat = set().union(*cmds)
            for c in ("watch", "guard", "auto", "pause", "resume", "status"):
                self.assertIn(c, flat)


if __name__ == "__main__":
    unittest.main()
