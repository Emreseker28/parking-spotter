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
        self.settings.raw["guard"] = {**self.settings.raw.get("guard", {}), "enabled": False}
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

    def test_guard_reports_disabled_when_not_configured(self):
        self.bot.controller.verify()
        self.assertIsNone(self.bot.guard)
        self.assertIn("kapalı", self.call(self.bot.cmd_guard)[0])

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


class FakeGuard:
    def __init__(self, ok=True):
        self.ok, self.armed_idx, self.disarmed = ok, None, []
        self.acked = False

    async def arm(self, idx):
        self.armed_idx = idx
        return (True, "Spot: araç bulundu, GUARD aktif.") if self.ok else (False, "araç net görünmüyor")

    async def disarm(self, reason=""):
        self.disarmed.append(reason)

    def ack(self):
        return not self.acked and not setattr(self, "acked", True)

    def status_text(self):
        return "GUARD: fake"


def make_query_update(chat_id, data):
    edits = []

    async def edit_text(text, **kw):
        edits.append(text)

    async def answer(*a, **k):
        pass

    q = SimpleNamespace(data=data, answer=answer, message=SimpleNamespace(edit_text=edit_text))
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), callback_query=q,
                           effective_message=None), edits


class GuardCommandsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings()
        self.settings.raw["spotwatch"] = {**self.settings.raw.get("spotwatch", {}), "enabled": True,
                                          "state_db": str(Path(self._tmp.name) / "s.db")}
        self.settings.raw["guard"] = {**self.settings.raw.get("guard", {}), "enabled": True}
        self.bot = ParkingBot(self.settings)
        self.bot.init_spotwatch(MagicMock(), FakeJobQueue())
        self.assertIsNotNone(self.bot.guard)                  # real engine is built when enabled
        self.bot.guard = FakeGuard()
        self.chat = self.settings.secrets.allowed_chat_id
        self.bot.controller.verify()

    def tearDown(self):
        self._tmp.cleanup()

    def msg(self, fn, *args):
        upd, sent = make_update(self.chat)
        asyncio.run(fn(upd, SimpleNamespace(args=list(args))))
        return sent

    def button(self, data, chat=None):
        upd, edits = make_query_update(self.chat if chat is None else chat, data)
        asyncio.run(self.bot.on_guard_button(upd, SimpleNamespace()))
        return edits

    def test_guard_command_asks_which_spot(self):
        upd, sent = make_update(self.chat)
        captured = {}

        async def reply_text(text, reply_markup=None, **kw):
            captured["text"], captured["kb"] = text, reply_markup
        upd.effective_message.reply_text = reply_text
        asyncio.run(self.bot.cmd_guard(upd, SimpleNamespace(args=[])))
        self.assertIn("hangi park yerinde", captured["text"])
        rows = captured["kb"].inline_keyboard
        self.assertEqual([r[0].text for r in rows], [sp.name for sp in self.settings.spots])
        self.assertEqual([r[0].callback_data for r in rows], ["guard:arm:0", "guard:arm:1"])
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)   # nothing armed until chosen

    def test_choosing_a_spot_arms_and_enters_guard(self):
        edits = self.button("guard:arm:1")
        self.assertEqual(self.bot.guard.armed_idx, 1)
        self.assertEqual(self.bot.controller.mode, Mode.GUARD)
        self.assertEqual(self.bot.controller.guard_spot, self.settings.spots[1].name)
        self.assertIn("GUARD aktif", edits[-1])
        self.assertIn("diske yazılmaz", edits[-1])

    def test_failed_arming_stays_in_spot_watch(self):
        self.bot.guard.ok = False
        edits = self.button("guard:arm:0")
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)
        self.assertIn("net görünmüyor", edits[-1])

    def test_unauthorized_chat_cannot_arm(self):
        self.button("guard:arm:0", chat=self.chat + 1)
        self.assertIsNone(self.bot.guard.armed_idx)

    def test_bad_callback_data_is_ignored(self):
        self.button("guard:arm:9")
        self.button("guard:arm:x")
        self.assertIsNone(self.bot.guard.armed_idx)

    def test_watch_disarms_guard(self):
        self.button("guard:arm:0")
        self.msg(self.bot.cmd_watch)
        self.assertEqual(self.bot.guard.disarmed, ["/watch"])
        self.assertEqual(self.bot.controller.mode, Mode.SPOT_WATCH)
        self.assertIsNone(self.bot.controller.guard_spot)

    def test_ack_command_and_button(self):
        self.assertIn("onaylandı", self.msg(self.bot.cmd_ack)[0])
        self.assertIn("yok", self.msg(self.bot.cmd_ack)[0])
        self.bot.guard.acked = False
        answers = []

        async def answer(text=None, **k):
            answers.append(text)
        upd = SimpleNamespace(effective_chat=SimpleNamespace(id=self.chat),
                              callback_query=SimpleNamespace(data="gk:ack", answer=answer))
        asyncio.run(self.bot.on_ack_button(upd, SimpleNamespace()))
        self.assertIn("Onaylandı", answers[0])

    def test_status_includes_guard_section(self):
        self.assertIn("GUARD: fake", self.msg(self.bot.cmd_status)[0])

    def test_pause_works_from_guard_and_keeps_guard(self):
        self.button("guard:arm:0")
        self.msg(self.bot.cmd_pause, "5m")
        self.assertEqual(self.bot.controller.mode, Mode.PAUSED)
        self.assertTrue(self.bot.controller.guarding)
        self.msg(self.bot.cmd_resume)
        self.assertEqual(self.bot.controller.mode, Mode.GUARD)


class ResumeAfterRestartTest(unittest.TestCase):
    def _bot(self, tmp, ok):
        s = load_settings()
        s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True, "state_db": str(Path(tmp) / "s.db")}
        s.raw["guard"] = {**s.raw.get("guard", {}), "enabled": True}
        b = ParkingBot(s)
        b.init_spotwatch(MagicMock(), FakeJobQueue())
        b.guard = FakeGuard(ok)
        return b, s

    def test_guard_is_rearmed_only_when_the_car_is_still_there(self):
        for ok, expect in ((True, Mode.GUARD), (False, Mode.SPOT_WATCH)):
            with tempfile.TemporaryDirectory() as tmp:
                b, s = self._bot(tmp, True)
                b.controller.verify(); b.controller.set_manual(Mode.GUARD, s.spots[1].name)
                b2, _ = self._bot(tmp, ok)                      # "restart"
                self.assertEqual(b2.controller.verify(), Mode.SPOT_WATCH)   # never GUARD blindly
                self.assertEqual(b2.controller.resume_guard, s.spots[1].name)
                asyncio.run(b2._maybe_resume_guard())
                self.assertEqual(b2.controller.mode, expect)
                self.assertEqual(b2.guard.armed_idx, 1)
                self.assertIsNone(b2.controller.resume_guard)


class ReschedulingTest(unittest.TestCase):
    """Regression: the running job removing itself raised JobLookupError in
    production (the fired job is already gone) and killed the loop after 1 cycle."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = load_settings()
        self.settings.raw["spotwatch"] = {**self.settings.raw.get("spotwatch", {}),
                                          "enabled": True,
                                          "state_db": str(Path(self._tmp.name) / "s.db")}
        self.bot = ParkingBot(self.settings)
        self.jq = FakeJobQueue()
        self.bot.init_spotwatch(MagicMock(), self.jq)

    def tearDown(self):
        self._tmp.cleanup()

    def test_job_reschedules_even_if_removing_the_stale_handle_raises(self):
        stale = MagicMock()
        stale.schedule_removal.side_effect = RuntimeError("No job by the id of x was found")
        self.bot._sw_job = stale
        self.bot._schedule_spotwatch(5)           # must not raise
        self.assertEqual(len(self.jq.jobs), 1)

    def test_spotwatch_job_always_reschedules(self):
        async def cycle():
            return 42.0
        self.bot.spotwatch.run_cycle = cycle
        fired = MagicMock()
        fired.schedule_removal.side_effect = RuntimeError("already fired")
        self.bot._sw_job = fired                  # the handle of the job now running
        asyncio.run(self.bot.spotwatch_job(SimpleNamespace()))
        fired.schedule_removal.assert_not_called()   # the running job is not cancelled
        self.assertEqual([w for _, w, _ in self.jq.jobs], [42.0])

    def test_crashing_cycle_still_reschedules(self):
        async def boom():
            raise RuntimeError("cycle crashed")
        self.bot.spotwatch.run_cycle = boom
        asyncio.run(self.bot.spotwatch_job(SimpleNamespace()))
        self.assertEqual([w for _, w, _ in self.jq.jobs], [self.bot.spotwatch.cfg.interval])


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


class BotFatherCommandsTest(unittest.TestCase):
    """Every command the bot handles must be pushed to Telegram (setMyCommands) at startup."""

    def _app(self):
        import re  # noqa: F401
        tmp = tempfile.mkdtemp()
        s = load_settings()
        s.raw["spotwatch"] = {**s.raw.get("spotwatch", {}), "enabled": True,
                              "state_db": str(Path(tmp) / "s.db")}
        return build_application(s)

    def test_every_handled_command_is_registered_with_telegram(self):
        from parkingbot.bot import BOT_COMMANDS
        app = self._app()
        handled = {c for g in app.handlers.values() for h in g if hasattr(h, "commands") for c in h.commands}
        self.assertEqual(handled - {c for c, _ in BOT_COMMANDS}, set(),
                         "command handled but missing from BOT_COMMANDS")
        self.assertEqual({c for c, _ in BOT_COMMANDS} - handled, set(),
                         "command listed in BOT_COMMANDS but not handled")

    def test_commands_respect_telegram_limits(self):
        import re
        from parkingbot.bot import BOT_COMMANDS
        for cmd, desc in BOT_COMMANDS:
            self.assertRegex(cmd, r"^[a-z0-9_]{1,32}$")
            self.assertTrue(3 <= len(desc) <= 256, cmd)
        self.assertLessEqual(len(BOT_COMMANDS), 100)

    def test_post_init_pushes_the_full_list(self):
        from unittest.mock import AsyncMock
        from parkingbot.bot import BOT_COMMANDS
        app = self._app()
        self.assertIsNotNone(app.post_init)
        app.bot = MagicMock()
        app.bot.set_my_commands = AsyncMock()
        asyncio.run(app.post_init(app))
        sent = [c.command for c in app.bot.set_my_commands.call_args[0][0]]
        self.assertEqual(sent, [c for c, _ in BOT_COMMANDS])
        for new in ("watch", "guard", "ack", "auto", "pause", "resume", "status"):
            self.assertIn(new, sent)


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
