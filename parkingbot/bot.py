"""Telegram bot: /cameras viewer, /parking on-demand check, and the push loop.

Single long-polling instance only (duplicate pollers => Telegram 409). All
captures funnel through one serialized CaptureWorker.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from .capture import CaptureError, CaptureWorker
from .config import ROOT, Settings, Spot
from .detection import Detector, SpotResult
from .feedback import (
    CB_PREFIX,
    append_feedback,
    build_accuracy_report,
    parse_feedback_callback,
    prune_frames,
    save_spot_crop,
)
from .jsonl import append_jsonl
from .guard import GuardEngine
from .modes import Mode, ModeController, ModeUnavailable, StateStore
from .push_state import load_push_state, save_push_state
from .spotwatch import SpotWatch
from .stability import StabilityGate
from .verifier import VerifyResult, Verifier

log = logging.getLogger(__name__)

CAM_CB_PREFIX = "cam:"
ALL_CB = "cam:all"


def restricted(func):
    """Allow only ALLOWED_CHAT_ID; the bot streams views of the house."""

    @functools.wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        if chat is None or chat.id != self.settings.secrets.allowed_chat_id:
            log.warning("Rejected access from chat_id=%s", chat.id if chat else "?")
            if update.callback_query:
                await update.callback_query.answer("Not authorized.", show_alert=True)
            elif update.effective_message:
                await update.effective_message.reply_text("⛔ Not authorized.")
            return
        return await func(self, update, context)

    return wrapper


ACK_CB = "gk:ack"
GUARD_CB_PREFIX = "guard:arm:"


def _ack_markup(ack: bool):
    return InlineKeyboardMarkup([[InlineKeyboardButton("✅ Gördüm", callback_data=ACK_CB)]]) if ack else None


class TelegramNotifier:
    """Sends Spot Watch / Guard messages to the owner's chat. Photos go straight from
    RAM bytes; nothing is staged on disk."""

    def __init__(self, bot, chat_id: int):
        self._bot = bot
        self._chat_id = chat_id

    async def send_text(self, text: str, silent: bool = False, ack: bool = False) -> None:
        await self._bot.send_message(chat_id=self._chat_id, text=text,
                                     disable_notification=silent, reply_markup=_ack_markup(ack))

    async def send_photo(self, jpeg: bytes, caption: str, ack: bool = False) -> None:
        await self._bot.send_photo(chat_id=self._chat_id, photo=jpeg, caption=caption,
                                   reply_markup=_ack_markup(ack))

    async def send_media_group(self, jpegs: list[bytes], caption: str) -> None:
        media = [InputMediaPhoto(j, caption=caption if i == 0 else None) for i, j in enumerate(jpegs[:10])]
        await self._bot.send_media_group(chat_id=self._chat_id, media=media)


_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)


def parse_duration(text: str, default: float = 600.0) -> float | None:
    """'10m' / '90s' / '1h' / '15' (minutes) -> seconds; '' -> default; bad -> None."""
    if not text.strip():
        return default
    m = _DUR_RE.match(text)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower() or "m"
    secs = n * {"s": 1, "m": 60, "h": 3600}[unit]
    return secs if 0 < secs <= 24 * 3600 else None


def apply_timezone(settings: Settings):
    """Pin the process to the configured zone (Istanbul) so every localtime user
    (night hours, feedback hour) agrees with the camera clock, not container UTC.
    Returns the ZoneInfo (or None if unavailable)."""
    name = str(settings.spotwatch.get("timezone", "Europe/Istanbul"))
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001
        log.warning("timezone %r unavailable (%s); leaving process TZ unchanged", name, exc)
        return None
    os.environ["TZ"] = name
    if hasattr(time, "tzset"):
        time.tzset()
    return tz


class ParkingBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.capture = CaptureWorker(settings)
        self.detector = Detector(settings)
        self.verifier = Verifier(settings)
        self.stability = StabilityGate(settings)
        # Per-spot debounce/transition state for the push loop. Persisted after
        # every poll (atomic write) and restored on startup when fresh, so a
        # restart mid-cycle keeps the confirmed state (Phase 0 task 3).
        raw_state = str(settings.push.get("state_path", "data/push_state.json"))
        sp = Path(raw_state)
        self._push_state_path = sp if sp.is_absolute() else (ROOT / sp)
        self._push_state_max_age = float(settings.push.get("state_max_age_hours", 24)) * 3600
        self._push_state: dict[str, dict] = self._restore_push_state()

        # Human feedback (Phase 1). Relative paths resolve against the repo root
        # (/app/data in Docker -> the /mnt/storage mount).
        fcfg = settings.feedback
        self._feedback_enabled = bool(fcfg.get("enabled", False))
        raw_fb = str(fcfg.get("log_path", "data/feedback.jsonl"))
        pfb = Path(raw_fb)
        self._feedback_log = pfb if pfb.is_absolute() else (ROOT / pfb)
        raw_frames = str(fcfg.get("frames_dir", "data/frames"))
        pf = Path(raw_frames)
        self._frames_dir = pf if pf.is_absolute() else (ROOT / pf)
        self._feedback_max_frames = int(fcfg.get("max_frames", 2000))
        self._feedback_quality = int(fcfg.get("jpeg_quality", 85))
        self._crop_pad = float(settings.detection.get("crop_pad", 0.15))
        # "frame_id:spot_idx" -> {coverage, context, occupied} for callback lookups
        self._frame_index: dict[str, dict] = {}

        # Tier-2 detector + RAM guard (Phase 3).
        t2 = settings.detection_tier2
        self._tier2_on = bool(t2.get("enabled", False))
        self._tier2_min_mb = float(t2.get("min_available_mb", 2500))
        # Accuracy log (bot-vs-VLM). Relative paths resolve against the repo root;
        # never /mnt/models (read-only in Docker).
        raw_log = str(settings.verify.get("log_path", "data/parking-accuracy.jsonl"))
        p = Path(raw_log)
        self._audit_log = p if p.is_absolute() else (ROOT / p)

        # Spot Watch (set up in init_spotwatch once the Telegram bot exists).
        self.spotwatch: SpotWatch | None = None
        self.guard: GuardEngine | None = None
        self.controller: ModeController | None = None
        self._sw_job = None

    def init_spotwatch(self, tg_bot, job_queue, tz=None) -> None:
        raw = Path(str(self.settings.spotwatch.get("state_db", "state/spotter.db")))
        db = raw if raw.is_absolute() else (ROOT / raw)
        store = StateStore(db)
        self.controller = ModeController(store)
        self.spotwatch = SpotWatch(
            self.settings, self.controller, store, self.capture, self.detector,
            self.verifier, self.stability,
            TelegramNotifier(tg_bot, self.settings.secrets.allowed_chat_id),
            mem_available=self._mem_available_mb, tz=tz,
        )
        self._job_queue = job_queue
        self.guard = None
        if bool(self.settings.guard.get("enabled", False)):
            self.guard = GuardEngine(self.settings, self.controller, self.capture, self.detector,
                                     self.spotwatch.outbox, tz=tz)
            self.controller.guard_available = True
        log.info("Spot Watch enabled: every %.0fs, state db %s, guard=%s",
                 self.spotwatch.cfg.interval, db, self.guard is not None)

    def _restore_push_state(self) -> dict[str, dict]:
        state = load_push_state(self._push_state_path, self._push_state_max_age)
        if state is None:
            return {}
        log.info("restored push state from %s", self._push_state_path)
        return state

    @staticmethod
    def _mem_available_mb() -> float:
        """Host MemAvailable in MB (containers share the host kernel, so
        /proc/meminfo reflects the host). Fail-open (inf) if unreadable."""
        try:
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            log.warning("could not read MemAvailable from /proc/meminfo")
        return float("inf")

    def _next_push_interval(self, results_state: dict, now=None) -> float:
        """Adaptive poll interval in seconds (Phase 5).

        confirm_interval while a flip is pending (the reading differs from the
        confirmed state and the debounce hasn't resolved), else the base interval.
        Night hours use a longer interval (less CPU/noise).
        """
        push = self.settings.push
        base = float(push.get("interval_minutes", 10)) * 60
        confirm = float(push.get("confirm_interval_minutes", 2)) * 60
        pending = any(
            st.get("confirmed") is not None and st.get("candidate") != st.get("confirmed")
            for st in results_state.values()
        )
        interval = confirm if pending else base
        night = push.get("night_interval_minutes", 0)
        if night:
            hour = (now or time.localtime()).tm_hour
            start = int(push.get("night_start_hour", 23))
            end = int(push.get("night_end_hour", 6))
            if start <= end:
                in_night = start <= hour < end
            else:  # range wraps midnight
                in_night = hour >= start or hour < end
            if in_night:
                interval = float(night) * 60
        return interval

    # ------------------------------------------------------------- feedback
    def _save_feedback_crop(self, r: SpotResult, spot_idx: int, ts: float,
                            context: str) -> str | None:
        """Save the judged ROI crop (Phase 1). Returns the frame id, else None."""
        if not self._feedback_enabled or r.frame is None:
            return None
        fid = save_spot_crop(r.frame, r.spot, spot_idx, r.occupied, ts,
                            self._frames_dir, self._crop_pad, self._feedback_quality)
        if fid is not None:
            prune_frames(self._frames_dir, self._feedback_max_frames)
        return fid

    def _feedback_buttons(self, entries: list[tuple[SpotResult, int]],
                          context: str) -> InlineKeyboardMarkup | None:
        """One row per spot: [✅ Doğru] [❌ Yanlış]. Saves the crop, registers
        the frame index, and returns the keyboard (None when feedback is off or
        no crop could be saved)."""
        rows = []
        now = time.time()
        for r, spot_idx in entries:
            fid = self._save_feedback_crop(r, spot_idx, now, context)
            if fid is None:
                continue
            verdict = "occupied" if r.occupied else "free"
            self._frame_index[f"{fid}:{spot_idx}"] = {
                "coverage": round(r.coverage, 4), "context": context,
                "occupied": r.occupied,
            }
            rows.append([
                InlineKeyboardButton("✅ Doğru",
                                     callback_data=f"fb:{spot_idx}:{verdict}:{fid}:ok"),
                InlineKeyboardButton("❌ Yanlış",
                                     callback_data=f"fb:{spot_idx}:{verdict}:{fid}:no"),
            ])
        return InlineKeyboardMarkup(rows) if rows else None

    @restricted
    async def on_feedback_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not self._feedback_enabled:
            await query.answer()
            return
        parsed = parse_feedback_callback(query.data or "")
        if parsed is None:
            await query.answer()
            return
        spot_idx, bot_verdict, fid, response = parsed
        if not 0 <= spot_idx < len(self.settings.spots):
            await query.answer()
            return
        spot = self.settings.spots[spot_idx]
        info = self._frame_index.get(f"{fid}:{spot_idx}", {})
        label = "correct" if response == "ok" else "wrong"
        append_feedback(self._feedback_log, {
            "ts": time.time(), "type": "feedback",
            "spot": spot.name, "spot_idx": spot_idx,
            "bot_occupied": bot_verdict == "occupied",
            "coverage": info.get("coverage", 0.0),
            "label": label, "frame_id": fid,
            "hour": time.localtime().tm_hour,
            "context": info.get("context", "unknown"),
        })
        log.info("feedback: %r %s (bot=%s, frame=%s)", spot.name, label, bot_verdict, fid)
        await query.answer("Kaydedildi ✅" if response == "ok" else "Kaydedildi ❌")

    # ------------------------------------------------------------ commands
    @restricted
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.effective_message.reply_text(
            "🚗 *Parking Spotter*\n\n"
            "/cameras — view a camera\n"
            "/parking — check the parking spots now\n"
            "/accuracy — bot vs. AI credibility score\n"
            "/watch /pause 10m /resume /status — Spot Watch",
            parse_mode=ParseMode.MARKDOWN,
        )

    @restricted
    async def cmd_cameras(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        buttons = [
            [InlineKeyboardButton(label, callback_data=f"{CAM_CB_PREFIX}{ch}")]
            for ch, label in self.settings.cameras.items()
        ]
        buttons.append([InlineKeyboardButton("📷 All", callback_data=ALL_CB)])
        await update.effective_message.reply_text(
            "Which camera?", reply_markup=InlineKeyboardMarkup(buttons)
        )

    @restricted
    async def on_camera_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        subtype = int(self.settings.capture.get("viewer_subtype", 1))

        if data == ALL_CB:
            channels = list(self.settings.cameras.keys())
        else:
            channels = [int(data[len(CAM_CB_PREFIX):])]

        for ch in channels:
            label = self.settings.cameras.get(ch, f"Channel {ch}")
            try:
                jpeg = await self.capture.grab(ch, subtype)
            except CaptureError as exc:
                await query.message.reply_text(f"⚠️ {label}: {exc}")
                continue
            await context.bot.send_photo(
                chat_id=query.message.chat_id, photo=jpeg, caption=label
            )

    @restricted
    async def cmd_parking(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = await update.effective_message.reply_text("⏳ Checking spots…")
        try:
            results = await self._check_all_spots()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            log.exception("parking check failed")
            await msg.edit_text(f"⚠️ Check failed: {exc}")
            return
        await msg.edit_text(
            self._format_results(results),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=self._feedback_buttons(
                [(r, i) for i, r in enumerate(results)], "parking"
            ),
        )

    # ----------------------------------------------------- Spot Watch commands
    def _schedule_spotwatch(self, when: float) -> None:
        """Keep exactly one pending Spot Watch job: replace any scheduled one."""
        job, self._sw_job = self._sw_job, None
        if job is not None:
            try:
                job.schedule_removal()
            except Exception as exc:  # noqa: BLE001 - already fired/removed: nothing to cancel
                log.debug("spotwatch job already gone: %s", exc)
        self._sw_job = self._job_queue.run_once(self.spotwatch_job, when=when)

    async def spotwatch_job(self, context: ContextTypes.DEFAULT_TYPE):
        """Run one cycle and ALWAYS reschedule (a crash must never end the loop)."""
        delay = float(self.spotwatch.cfg.interval)
        self._sw_job = None   # this job has fired; there is nothing pending to cancel
        try:
            delay = await self.spotwatch.run_cycle()
            await self._maybe_resume_guard()
        except Exception as exc:  # noqa: BLE001
            log.exception("spotwatch cycle crashed: %s", exc)
        finally:
            self._schedule_spotwatch(delay)

    async def _maybe_resume_guard(self) -> None:
        """After a restart a persisted GUARD is only re-armed if the camera still shows
        the owner's car in that spot (never trusted blindly)."""
        name = self.controller.resume_guard if self.controller else None
        if not name or self.guard is None or self.controller.mode != Mode.SPOT_WATCH:
            return
        self.controller.resume_guard = None
        idx = next((i for i, sp in enumerate(self.settings.spots) if sp.name == name), None)
        if idx is None:
            return
        ok, msg = await self.guard.arm(idx)
        if ok:
            self.controller.set_manual(Mode.GUARD, name)
            await self.spotwatch.outbox.send(f"🛡️ Yeniden başlatma sonrası GUARD devam ediyor. {msg}", silent=True)
        else:
            await self.spotwatch.outbox.send(
                f"ℹ️ BİLGİ · Yeniden başlatma sonrası GUARD kurulamadı, SPOT_WATCH'a dönüldü. {msg}", silent=True)

    async def _sw_reply(self, update: Update, text: str) -> bool:
        if self.spotwatch is None:
            await update.effective_message.reply_text("Spot Watch kapalı (config: spotwatch.enabled).")
            return False
        await update.effective_message.reply_text(text)
        return True

    @restricted
    async def cmd_watch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.spotwatch is None:
            await self._sw_reply(update, "")
            return
        if self.guard is not None:
            await self.guard.disarm("/watch")
        self.controller.set_manual(Mode.SPOT_WATCH)
        self.spotwatch.kick()
        self._schedule_spotwatch(0.5)
        await update.effective_message.reply_text(
            "👀 SPOT_WATCH (manuel). Park yerleri izleniyor; boşalınca haber vereceğim.")

    @restricted
    async def cmd_guard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.spotwatch is None:
            await self._sw_reply(update, "")
            return
        if self.guard is None:
            await update.effective_message.reply_text("🛡️ GUARD kapalı (config: guard.enabled).")
            return
        if self.controller.mode == Mode.UNKNOWN:
            await update.effective_message.reply_text("Mod henüz doğrulanmadı, birkaç saniye sonra tekrar dene.")
            return
        buttons = [[InlineKeyboardButton(sp.name, callback_data=f"{GUARD_CB_PREFIX}{i}")]
                   for i, sp in enumerate(self.settings.spots)]
        await update.effective_message.reply_text(
            "🛡️ Araban hangi park yerinde?", reply_markup=InlineKeyboardMarkup(buttons))

    @restricted
    async def on_guard_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        try:
            idx = int((query.data or "")[len(GUARD_CB_PREFIX):])
            spot = self.settings.spots[idx]
        except (ValueError, IndexError):
            return
        if self.guard is None or self.controller.mode == Mode.UNKNOWN:
            return
        await query.message.edit_text(f"⏳ {spot.name}: araç aranıyor (~10 sn)…")
        ok, msg = await self.guard.arm(idx)
        if ok:
            self.controller.set_manual(Mode.GUARD, spot.name)
            await query.message.edit_text(f"🛡️ {msg}\nGörüntü RAM'de tutulur, diske yazılmaz. /watch ile çık.")
        else:
            await query.message.edit_text(f"⚠️ {msg}")

    @restricted
    async def cmd_ack(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.guard is None:
            await self._sw_reply(update, "")
            return
        msg = "✅ Alarm onaylandı, tekrar durdu." if self.guard.ack() else "Onaylanacak alarm yok."
        await update.effective_message.reply_text(msg)

    @restricted
    async def on_ack_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        done = self.guard is not None and self.guard.ack()
        await query.answer("Onaylandı ✅" if done else "Onaylanacak alarm yok")

    @restricted
    async def cmd_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.spotwatch is None:
            await self._sw_reply(update, "")
            return
        self.controller.set_auto(True)
        await update.effective_message.reply_text(
            "🔁 Otomatik mod işaretlendi, ancak otomatik geçiş (arabanı tanıma) "
            "Faz 3'te geliyor; şimdilik SPOT_WATCH sabit.")

    @restricted
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.spotwatch is None:
            await self._sw_reply(update, "")
            return
        secs = parse_duration(" ".join(context.args or []))
        if secs is None:
            await update.effective_message.reply_text("Kullanım: /pause 10m  (s, m veya h; en çok 24h)")
            return
        try:
            self.controller.pause(secs)
        except ValueError as exc:
            await update.effective_message.reply_text(f"Şu an duraklatılamaz: {exc}")
            return
        await update.effective_message.reply_text(
            f"⏸️ {secs / 60:.0f} dk duraklatıldı (bildirimler kapalı, izleme sürüyor). /resume ile bitir.")

    @restricted
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.spotwatch is None:
            await self._sw_reply(update, "")
            return
        msg = "▶️ Devam ediyor." if self.controller.resume() else "Duraklatılmış bir şey yok."
        self.spotwatch.kick()
        await update.effective_message.reply_text(msg)

    @restricted
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.spotwatch is None:
            await self._sw_reply(update, "")
            return
        text = self.spotwatch.status_text()
        if self.guard is not None:
            text += "\n\n" + self.guard.status_text()
        await update.effective_message.reply_text(text)

    @restricted
    async def cmd_accuracy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Report how often the bot and the VLM agreed (a credibility score)."""
        days = 7
        if context.args:
            try:
                days = max(1, int(context.args[0]))
            except ValueError:
                pass
        await update.effective_message.reply_text(
            self._format_accuracy(days), parse_mode=ParseMode.MARKDOWN
        )

    def _format_accuracy(self, days: int) -> str:
        """Feedback-based report (Phase 1) + VLM-agreement lines (only when verify
        is enabled). Returns a 'no data' line when neither has records."""
        parts = []
        if self._feedback_enabled and self._feedback_log.exists():
            report = build_accuracy_report(days, self._read_jsonl(self._feedback_log))
            if report:
                # Surface tier-2 rescues (logged to the accuracy log).
                cutoff = time.time() - days * 86400
                t2_rescues = sum(
                    1 for rec in self._read_jsonl(self._audit_log)
                    if rec.get("type") == "tier2_override"
                    and float(rec.get("ts", 0)) >= cutoff
                )
                if t2_rescues:
                    report += f"\n• Tier-2 ile düzeltilen yanlış 'BOŞ': *{t2_rescues}*"
                parts.append(report)
        if self.verifier.enabled and self._audit_log.exists():
            vlm = self._format_vlm_accuracy(days)
            if vlm:
                parts.append(vlm)
        if not parts:
            return "📊 *Doğruluk*\nHenüz kayıt yok."
        return "\n\n".join(parts)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        records = []
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            return records
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
        return records

    def _format_vlm_accuracy(self, days: int) -> str:
        """Bot-vs-VLM agreement (only meaningful while the VLM gate is enabled)."""
        cutoff = time.time() - days * 86400
        per_spot: dict[str, list[int]] = {}   # name -> [agree, total]
        checks = 0
        suppressed = 0
        for rec in self._read_jsonl(self._audit_log):
            if float(rec.get("ts", 0)) < cutoff:
                continue
            if rec.get("type") == "suppressed":
                suppressed += 1
                continue
            if rec.get("type") != "audit" or rec.get("agree") is None:
                continue
            checks += 1
            tally = per_spot.setdefault(rec.get("spot", "?"), [0, 0])
            tally[0] += 1 if rec["agree"] else 0
            tally[1] += 1

        if checks == 0:
            return ""
        agree_total = sum(a for a, _ in per_spot.values())
        lines = [f"📊 *Doğruluk* (son {days} gün)"]
        lines.append(f"• Genel uyum: *{agree_total / checks:.0%}*  ({agree_total}/{checks})")
        for name, (a, t) in sorted(per_spot.items()):
            lines.append(f"• {name}: {a / t:.0%}  ({a}/{t})")
        lines.append(f"• Engellenen yanlış 'BOŞ': *{suppressed}*")
        return "\n".join(lines)

    # ------------------------------------------------------------- detection
    async def _check_all_spots(self) -> list[SpotResult]:
        """Grab each needed channel once (mainstream), evaluate every spot on it.

        Per channel: one full-frame inference (shared by its spots); per spot:
        one extra ROI-crop inference combined with the full-frame result.
        """
        import asyncio

        subtype = int(self.settings.capture.get("detect_subtype", 0))
        channels = sorted({spot.channel for spot in self.settings.spots})

        # detection is CPU-bound + blocking -> run in a thread executor.
        loop = asyncio.get_running_loop()
        results: list[SpotResult] = []
        for ch in channels:
            ch_spots = [s for s in self.settings.spots if s.channel == ch]
            # ROI safety check (Phase 4): the detection path expects the mainstream
            # frame size the ROI polygons are defined in. A snapshot frame of a
            # different size is rejected inside grab() and falls back to ffmpeg.
            expected = ch_spots[0].frame_size if ch_spots else None
            jpeg = await self.capture.grab(ch, subtype, expected_size=expected)

            def _detect_channel(jpeg=jpeg, ch_spots=ch_spots):
                img = self.detector.decode(jpeg)
                full_dets = self.detector.detect_array(img)
                return [self.detector.check_spot(img, s, full_dets) for s in ch_spots]

            results.extend(await loop.run_in_executor(None, _detect_channel))
        # preserve config order
        order = {s.name: i for i, s in enumerate(self.settings.spots)}
        results.sort(key=lambda r: order[r.spot.name])
        return results

    @staticmethod
    def _format_results(results: list[SpotResult]) -> str:
        lines = ["🅿️ *Parking*"]
        for r in results:
            state = "🔴 DOLU" if r.occupied else "🟢 BOŞ"
            lines.append(f"• {r.spot.name}: {state}  _(cov {r.coverage:.0%})_")
        return "\n".join(lines)

    # -------------------------------------------------------------- push loop
    async def push_job(self, context: ContextTypes.DEFAULT_TYPE):
        """Wrapper: run the poll and ALWAYS reschedule the next one (run_once
        chain), so an unexpected error can never kill the push loop."""
        try:
            await self._push_job_impl(context)
        except Exception as exc:  # noqa: BLE001 - never let the loop die
            log.exception("push poll crashed: %s", exc)
        finally:
            try:
                next_interval = self._next_push_interval(self._push_state)
            except Exception:  # noqa: BLE001
                next_interval = float(self.settings.push.get("interval_minutes", 10)) * 60
            context.job_queue.run_once(self.push_job, when=next_interval)

    async def _push_job_impl(self, context: ContextTypes.DEFAULT_TYPE):
        push = self.settings.push
        vcfg = self.settings.verify
        debounce = int(push.get("debounce_frames", 2))
        policy = str(push.get("policy", "free_only"))
        chat_id = self.settings.secrets.allowed_chat_id
        verify_on = self.verifier.enabled
        do_audit = verify_on and bool(vcfg.get("audit", True))
        do_gate = verify_on and bool(vcfg.get("gate", True))
        fail_open = bool(vcfg.get("fail_open", True))

        try:
            results = await self._check_all_spots()
        except Exception as exc:  # noqa: BLE001
            log.warning("push poll failed: %s", exc)
            return

        for spot_idx, r in enumerate(results):
            now = "occupied" if r.occupied else "free"
            st = self._push_state.setdefault(
                r.spot.name, {"confirmed": None, "candidate": None, "count": 0}
            )

            # Phase 2 stability gate: refresh the reference on strong occupied
            # readings; override FREE->OCCUPIED when the ROI is unchanged (the
            # night false-"BOŞ": YOLO misses a car that is still there).
            if self.stability.check(r, st["confirmed"]):
                r.occupied = True
                now = "occupied"

            # Debounce: require N consecutive agreeing frames before flipping.
            if now == st["candidate"]:
                st["count"] += 1
            else:
                st["candidate"] = now
                st["count"] = 1

            # Phase 1: save a crop for every state-flip candidate (dataset for the
            # Phase 2 offline replay and the Phase 7 fine-tune).
            if st["confirmed"] is not None and now != st["confirmed"]:
                self._save_feedback_crop(r, spot_idx, time.time(), "flip")

            # AUDIT: every poll, ask the VLM and log agreement (credibility score).
            verdict: VerifyResult | None = None
            if do_audit:
                verdict = await self.verifier.verify_spot(r.frame, r.spot)
                self._append_log({
                    "type": "audit",
                    "spot": r.spot.name,
                    "bot_occupied": r.occupied,
                    "coverage": round(r.coverage, 4),
                    "vlm_ok": verdict.ok,
                    "vlm_occupied": verdict.occupied,
                    "vlm_confidence": round(verdict.confidence, 3),
                    "agree": (r.occupied == verdict.occupied) if verdict.ok else None,
                })

            if st["count"] < debounce:
                continue
            if now == st["confirmed"]:
                continue

            prev, st["confirmed"] = st["confirmed"], now
            if prev is None:
                # first confirmed reading: set baseline, don't alert
                continue

            transition_to_free = now == "free"
            if policy == "free_only" and not transition_to_free:
                continue

            # Phase 3: RAM guard + tier-2 detector (only on a pending free flip).
            # Like every gate here, these can only SUPPRESS a free alert, never
            # create one — a failing gate can't spam.
            if transition_to_free and self._tier2_on:
                mem = self._mem_available_mb()
                if mem < self._tier2_min_mb:
                    log.info(
                        "ram_defer: %r MemAvailable %.0f MB < %.0f MB, deferring free alert",
                        r.spot.name, mem, self._tier2_min_mb,
                    )
                    st["confirmed"] = prev  # revert; retry next poll
                    continue
                t2 = self.detector.check_spot_tier2(r.frame, r.spot)
                if t2 is not None and t2.occupied:
                    log.info(
                        "tier2_override: %r tier-2 sees a car (cov %.2f)",
                        r.spot.name, t2.coverage,
                    )
                    self._append_log({
                        "type": "tier2_override",
                        "spot": r.spot.name,
                        "tier2_coverage": round(t2.coverage, 4),
                    })
                    st["confirmed"] = prev  # revert so we re-check next poll
                    continue

            # GATE: don't cry "BOŞ" unless the VLM agrees the spot is empty. This is
            # what kills the night false positives (YOLO misses a dark car -> FREE).
            unverified = False
            if do_gate and transition_to_free:
                if verdict is None:
                    verdict = await self.verifier.verify_spot(r.frame, r.spot)
                if verdict.ok and verdict.occupied:
                    log.info(
                        "gate: suppressed false BOŞ for %r (VLM sees a car, conf %.2f: %s)",
                        r.spot.name, verdict.confidence, verdict.reason,
                    )
                    self._append_log({
                        "type": "suppressed",
                        "spot": r.spot.name,
                        "vlm_confidence": round(verdict.confidence, 3),
                        "vlm_reason": verdict.reason,
                    })
                    st["confirmed"] = prev  # revert so we re-check next poll
                    continue
                if not verdict.ok:
                    if fail_open:
                        unverified = True
                        log.warning(
                            "gate: VLM unavailable for %r, sending UNVERIFIED alert (%s)",
                            r.spot.name, verdict.reason,
                        )
                    else:
                        log.warning(
                            "gate: VLM unavailable for %r, fail-closed -> suppress (%s)",
                            r.spot.name, verdict.reason,
                        )
                        st["confirmed"] = prev
                        continue

            emoji = "🟢" if transition_to_free else "🔴"
            state = "BOŞ" if transition_to_free else "DOLU"
            prefix = "⚠️ _(doğrulanamadı)_ " if unverified else ""
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"{prefix}{emoji} *{r.spot.name}* şimdi *{state}*",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=self._feedback_buttons([(r, spot_idx)], "alert"),
            )

        # Persist debounce/transition state so a restart mid-cycle survives.
        try:
            save_push_state(self._push_state_path, self._push_state)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save push state: %s", exc)

    # ---------------------------------------------------------- accuracy log
    def _append_log(self, record: dict) -> None:
        """Append one JSON line to the accuracy log. Never raises into the loop."""
        record = {"ts": time.time(), **record}
        try:
            vcfg = self.settings.verify
            append_jsonl(
                self._audit_log,
                record,
                max_bytes=int(vcfg.get("log_max_mb", 10)) * 1024 * 1024,
                keep=int(vcfg.get("log_keep", 3)),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write accuracy log %s: %s", self._audit_log, exc)


# Commands registered with Telegram (BotFather setMyCommands) at startup so the
# command list appears in the chat UI.
BOT_COMMANDS = [
    ("start", "Hoş geldin"),
    ("help", "Yardım"),
    ("cameras", "Kamerayı gör"),
    ("parking", "Park yerlerini kontrol et"),
    ("accuracy", "Bot doğruluk raporu"),
    ("watch", "Park yeri izleme modu"),
    ("guard", "Araba koruma modu"),
    ("ack", "Alarmı onayla"),
    ("auto", "Otomatik mod geçişi"),
    ("pause", "Bildirimleri duraklat, örn. /pause 10m"),
    ("resume", "Duraklatmayı bitir"),
    ("status", "Mod ve sağlık durumu"),
]


def _make_shutdown(bot_ref: "ParkingBot"):
    async def _shutdown(app: Application) -> None:
        if bot_ref.guard is not None:
            await bot_ref.guard.disarm("shutdown")   # kills the ffmpeg reader
    return _shutdown


async def _register_commands(app: Application) -> None:
    from telegram import BotCommand

    await app.bot.set_my_commands([BotCommand(cmd, desc) for cmd, desc in BOT_COMMANDS])


def build_application(settings: Settings) -> Application:
    app = (
        Application.builder()
        .token(settings.secrets.bot_token)
        .post_init(_register_commands)
        .build()
    )
    bot = ParkingBot(settings)
    app.post_shutdown = _make_shutdown(bot)
    tz = apply_timezone(settings)
    watch_on = bool(settings.spotwatch.get("enabled", False))
    if watch_on:
        bot.init_spotwatch(app.bot, app.job_queue, tz)

    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(CommandHandler("help", bot.cmd_start))
    app.add_handler(CommandHandler("cameras", bot.cmd_cameras))
    app.add_handler(CommandHandler("parking", bot.cmd_parking))
    app.add_handler(CommandHandler("accuracy", bot.cmd_accuracy))
    app.add_handler(CommandHandler("watch", bot.cmd_watch))
    app.add_handler(CommandHandler("guard", bot.cmd_guard))
    app.add_handler(CommandHandler("auto", bot.cmd_auto))
    app.add_handler(CommandHandler("pause", bot.cmd_pause))
    app.add_handler(CommandHandler("resume", bot.cmd_resume))
    app.add_handler(CommandHandler("status", bot.cmd_status))
    app.add_handler(CommandHandler("ack", bot.cmd_ack))
    app.add_handler(CallbackQueryHandler(bot.on_guard_button, pattern=f"^{GUARD_CB_PREFIX}"))
    app.add_handler(CallbackQueryHandler(bot.on_ack_button, pattern=f"^{ACK_CB}$"))
    app.add_handler(CallbackQueryHandler(bot.on_camera_button, pattern=f"^{CAM_CB_PREFIX}"))
    app.add_handler(CallbackQueryHandler(bot.on_feedback_button, pattern=f"^{CB_PREFIX[:-1]}"))

    if watch_on:
        # Spot Watch replaces the 10-minute push loop (two alerters would double up).
        bot._schedule_spotwatch(2.0)
        if settings.push.get("enabled", False):
            log.info("push loop disabled: superseded by spotwatch")
    elif settings.push.get("enabled", False):
        interval = float(settings.push.get("interval_minutes", 10)) * 60
        # Self-rescheduling run_once chain: push_job reschedules itself after each
        # run with the adaptive interval (Phase 5).
        app.job_queue.run_once(bot.push_job, when=interval)
        log.info("Push loop enabled: base every %.0fs (adaptive)", interval)

    return app
