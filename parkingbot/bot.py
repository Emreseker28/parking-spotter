"""Telegram bot: /cameras viewer, /parking on-demand check, and the push loop.

Single long-polling instance only (duplicate pollers => Telegram 409). All
captures funnel through one serialized CaptureWorker.
"""

from __future__ import annotations

import functools
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from .capture import CaptureError, CaptureWorker
from .config import Settings, Spot
from .detection import Detector, SpotResult

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


class ParkingBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.capture = CaptureWorker(settings)
        self.detector = Detector(settings)
        # per-spot debounce/transition state for the push loop
        self._push_state: dict[str, dict] = {}

    # ------------------------------------------------------------ commands
    @restricted
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.effective_message.reply_text(
            "🚗 *Parking Spotter*\n\n"
            "/cameras — view a camera\n"
            "/parking — check the parking spots now",
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
        await msg.edit_text(self._format_results(results), parse_mode=ParseMode.MARKDOWN)

    # ------------------------------------------------------------- detection
    async def _check_all_spots(self) -> list[SpotResult]:
        """Grab each needed channel once (mainstream), evaluate every spot on it."""
        import asyncio

        subtype = int(self.settings.capture.get("detect_subtype", 0))
        channels = sorted({spot.channel for spot in self.settings.spots})

        # detection is CPU-bound + blocking -> run in a thread executor.
        loop = asyncio.get_running_loop()
        results: list[SpotResult] = []
        for ch in channels:
            jpeg = await self.capture.grab(ch, subtype)
            detections, shape = await loop.run_in_executor(
                None, self.detector.detect, jpeg
            )
            for spot in self.settings.spots:
                if spot.channel != ch:
                    continue
                results.append(self.detector.occupancy(spot, detections, shape))
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
        push = self.settings.push
        debounce = int(push.get("debounce_frames", 2))
        policy = str(push.get("policy", "free_only"))
        chat_id = self.settings.secrets.allowed_chat_id

        try:
            results = await self._check_all_spots()
        except Exception as exc:  # noqa: BLE001
            log.warning("push poll failed: %s", exc)
            return

        for r in results:
            now = "occupied" if r.occupied else "free"
            st = self._push_state.setdefault(
                r.spot.name, {"confirmed": None, "candidate": None, "count": 0}
            )

            # Debounce: require N consecutive agreeing frames before flipping.
            if now == st["candidate"]:
                st["count"] += 1
            else:
                st["candidate"] = now
                st["count"] = 1

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

            emoji = "🟢" if transition_to_free else "🔴"
            state = "BOŞ" if transition_to_free else "DOLU"
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"{emoji} *{r.spot.name}* şimdi *{state}*",
                parse_mode=ParseMode.MARKDOWN,
            )


def build_application(settings: Settings) -> Application:
    app = Application.builder().token(settings.secrets.bot_token).build()
    bot = ParkingBot(settings)

    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(CommandHandler("help", bot.cmd_start))
    app.add_handler(CommandHandler("cameras", bot.cmd_cameras))
    app.add_handler(CommandHandler("parking", bot.cmd_parking))
    app.add_handler(CallbackQueryHandler(bot.on_camera_button, pattern=f"^{CAM_CB_PREFIX}"))

    if settings.push.get("enabled", False):
        interval = float(settings.push.get("interval_minutes", 4)) * 60
        app.job_queue.run_repeating(bot.push_job, interval=interval, first=interval)
        log.info("Push loop enabled: every %.0fs", interval)

    return app
