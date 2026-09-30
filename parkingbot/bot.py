"""Telegram bot: /cameras viewer, /parking on-demand check, and the push loop.

Single long-polling instance only (duplicate pollers => Telegram 409). All
captures funnel through one serialized CaptureWorker.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
from .push_state import load_push_state, save_push_state
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
            "/accuracy — bot vs. AI credibility score",
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
]


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

    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(CommandHandler("help", bot.cmd_start))
    app.add_handler(CommandHandler("cameras", bot.cmd_cameras))
    app.add_handler(CommandHandler("parking", bot.cmd_parking))
    app.add_handler(CommandHandler("accuracy", bot.cmd_accuracy))
    app.add_handler(CallbackQueryHandler(bot.on_camera_button, pattern=f"^{CAM_CB_PREFIX}"))
    app.add_handler(CallbackQueryHandler(bot.on_feedback_button, pattern=f"^{CB_PREFIX[:-1]}"))

    if settings.push.get("enabled", False):
        interval = float(settings.push.get("interval_minutes", 4)) * 60
        app.job_queue.run_repeating(bot.push_job, interval=interval, first=interval)
        log.info("Push loop enabled: every %.0fs", interval)

    return app
