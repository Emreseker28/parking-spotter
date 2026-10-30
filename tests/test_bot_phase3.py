"""Phase 3 in the bot: /auto, /enroll, auto-switch answers, CAR_GONE buttons, departure."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from parkingbot.bot import ParkingBot
from parkingbot.config import load_settings
from parkingbot.modes import Mode


class FakeJobQueue:
    def run_once(self, cb, when=0, **kw):
        return MagicMock()


class FakeGuard:
    spot = None

    def __init__(self):
        self.disarmed, self.escalated, self.departed = [], False, False
        self.alert = True

    async def disarm(self, reason=""):
        self.disarmed.append(reason)

    async def owner_departure(self):
        self.departed = True
        return True

    def escalate(self):
        self.escalated = True
        return self.alert

    def status_text(self):
        return "GUARD: fake"


def msg_update(chat):
    sent = []

    async def reply_text(text, reply_markup=None, **kw):
        sent.append((text, reply_markup))
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat),
                           effective_message=SimpleNamespace(reply_text=reply_text),
                           callback_query=None), sent


def query_update(chat, data):
    edits, answers = [], []

    async def edit_text(text, **kw):
        edits.append(text)

    async def answer(text=None, **kw):
        answers.append(text)

    async def reply_text(text, **kw):
        edits.append(text)
    q = SimpleNamespace(data=data, answer=answer,
                        message=SimpleNamespace(edit_text=edit_text, reply_text=reply_text, text="Soru"))
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat), callback_query=q,
                           effective_message=None), edits, answers


class Phase3BotTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        s = load_settings()
        s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True,
                              "state_db": str(Path(self._tmp.name) / "s.db")}
        s.raw["guard"] = {**s.raw.get("guard", {}), "enabled": True}
        self.settings = s
        self.bot = ParkingBot(s)
        self.bot.init_spotwatch(MagicMock(), FakeJobQueue())
        self.chat = s.secrets.allowed_chat_id
        self.guard = FakeGuard()
        self.bot.guard = self.guard
        self.bot.auto.guard = self.guard
        self.bot.controller.verify()
        self.bot.spotwatch.outbox.send = AsyncMock()

    def tearDown(self):
        self._tmp.cleanup()

    def msg(self, fn, *args):
        upd, sent = msg_update(self.chat)
        asyncio.run(fn(upd, SimpleNamespace(args=list(args))))
        return sent

    def btn(self, fn, data, chat=None):
        upd, edits, answers = query_update(self.chat if chat is None else chat, data)
        asyncio.run(fn(upd, SimpleNamespace()))
        return edits, answers

    def test_everything_is_wired(self):
        self.assertIsNotNone(self.bot.auto)
        self.assertEqual(self.bot.spotwatch.on_arrival.__func__, type(self.bot.auto).on_arrival)
        b = ParkingBot(self.settings)
        b.init_spotwatch(MagicMock(), FakeJobQueue())
        self.assertEqual(b.guard.on_departure.__func__, ParkingBot._on_departure)

    def test_auto_on_warns_when_nothing_is_enrolled_and_off_turns_it_off(self):
        text = self.msg(self.bot.cmd_auto)[0][0]
        self.assertTrue(self.bot.controller.auto)
        self.assertIn("/enroll", text)
        self.assertIn("kapatıldı", self.msg(self.bot.cmd_auto, "off")[0][0])
        self.assertFalse(self.bot.controller.auto)

    def test_auto_on_without_warning_once_references_exist(self):
        spot = self.settings.spots[0].name
        self.bot.auto.matcher.store.add(spot, "day", [0.0] * 4, 1)
        self.assertNotIn("Henüz referans yok", self.msg(self.bot.cmd_auto)[0][0])

    def test_manual_guard_or_watch_turns_auto_off(self):
        self.bot.controller.set_auto(True)
        self.bot.controller.set_manual(Mode.SPOT_WATCH)
        self.assertFalse(self.bot.controller.auto)

    def test_enroll_asks_which_spot_and_tags_rain(self):
        text, kb = self.msg(self.bot.cmd_enroll)[0]
        self.assertIn("hangi park yerinde", text)
        self.assertEqual([r[0].callback_data for r in kb.inline_keyboard], ["en:0:0", "en:1:0"])
        _, kb2 = self.msg(self.bot.cmd_enroll, "rain")[0]
        self.assertEqual([r[0].callback_data for r in kb2.inline_keyboard], ["en:0:1", "en:1:1"])

    def test_enroll_button_runs_enrollment_and_reports(self):
        self.bot.auto.enroll = AsyncMock(return_value=(True, "✅ 15 örnek kaydedildi"))
        edits, _ = self.btn(self.bot.on_enroll_button, "en:1:0")
        self.bot.auto.enroll.assert_awaited_once_with(1, False)
        self.assertIn("kaydedildi", edits[-1])
        self.bot.auto.enroll = AsyncMock(return_value=(False, "araç görünmüyor"))
        edits, _ = self.btn(self.bot.on_enroll_button, "en:0:1")
        self.bot.auto.enroll.assert_awaited_once_with(0, True)
        self.assertTrue(edits[-1].startswith("⚠️"))

    def test_enroll_button_ignores_bad_data_and_strangers(self):
        self.bot.auto.enroll = AsyncMock()
        self.btn(self.bot.on_enroll_button, "en:9:0")
        self.btn(self.bot.on_enroll_button, "en:x")
        self.btn(self.bot.on_enroll_button, "en:0:0", chat=self.chat + 1)
        self.bot.auto.enroll.assert_not_awaited()

    def test_auto_button_forwards_yes_and_no(self):
        self.bot.auto.answer = AsyncMock(return_value="✅ tamam")
        edits, _ = self.btn(self.bot.on_auto_button, "am:yes:abc123")
        self.bot.auto.answer.assert_awaited_with("abc123", True)
        self.assertIn("tamam", edits[-1])
        self.btn(self.bot.on_auto_button, "am:no:abc123")
        self.bot.auto.answer.assert_awaited_with("abc123", False)
        n = self.bot.auto.answer.await_count
        self.btn(self.bot.on_auto_button, "am:maybe:abc123")            # unknown verb ignored
        self.assertEqual(self.bot.auto.answer.await_count, n)

    def test_gone_buttons(self):
        _, ans = self.btn(self.bot.on_gone_button, "cg:me")
        self.assertTrue(self.guard.departed)
        _, ans = self.btn(self.bot.on_gone_button, "cg:notme")
        self.assertTrue(self.guard.escalated)
        self.assertIn("sık", ans[-1])

    def test_departure_returns_to_watch_keeps_auto_and_stays_quiet_about_the_free_spot(self):
        name = self.settings.spots[0].name
        self.bot.controller.guard_available = True
        self.bot.controller.auto_enter_guard(name)
        self.bot.controller.set_auto(True)
        asyncio.run(self.bot._on_departure("owner"))
        self.assertEqual(self.guard.disarmed, ["departure"])
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)
        self.assertTrue(self.bot.controller.auto)
        self.assertTrue(self.bot.spotwatch.trackers[name].quiet_next_free)
        self.assertIn("İyi yolculuklar", self.bot.spotwatch.outbox.send.call_args[0][0])
        asyncio.run(self.bot._on_departure("pause"))
        self.assertIn("ayrıldı", self.bot.spotwatch.outbox.send.call_args[0][0])

    def test_status_includes_auto_section(self):
        self.assertIn("Otomatik mod", self.msg(self.bot.cmd_status)[0][0])


if __name__ == "__main__":
    unittest.main()
