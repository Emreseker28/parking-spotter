"""SPOT_WATCH engine: alert as soon as a watched parking spot becomes free.

Both configured spots are watched independently (the owner's car is sometimes
parked elsewhere on the street, so "free" is always about the spot, never about
the owner's car).

Per cycle (default 60 s): grab each needed channel in memory, run the existing
detector, and push every reading through the existing false-"free" gates before
the per-spot tracker sees it:

    detector  ->  stability gate  ->  tier-2 detector (+ RAM guard)  -> tracker
    tracker says "free, confirmed"  ->  VLM gate (first alert only)  -> alert

A free reading is confirmed by an out-of-cycle re-check ~10 s later. A failed grab
never counts as free. Nothing here writes an image to disk: frames are RAM arrays,
the alert snapshot is JPEG-encoded in memory and handed straight to Telegram.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

import cv2
import numpy as np

from .capture import CaptureError
from .config import Settings
from .modes import Mode, ModeController, StateStore
from .spot_tracker import FREE, SpotTracker, TrackerEvent

log = logging.getLogger(__name__)


class Notifier(Protocol):
    """``ack=True`` asks for an inline "seen" button (GUARD alerts). Implementations
    may ignore it. ``send_media_group`` is only used by GUARD evidence."""

    async def send_text(self, text: str, silent: bool = False, **kw) -> None: ...
    async def send_photo(self, jpeg: bytes, caption: str, **kw) -> None: ...
    async def send_media_group(self, jpegs: list[bytes], caption: str, **kw) -> None: ...


@dataclass
class WatchConfig:
    interval: float = 60.0
    confirm_delay: float = 10.0
    confirm_reads: int = 2
    remind_every: float = 600.0
    remind_max: int = 3
    send_snapshot: bool = True          # D1: snapshot from RAM, never stored
    snapshot_quality: int = 80
    subtype: int = 0
    fail_threshold: int = 3
    backoff_max: float = 480.0
    state_max_age: float = 12 * 3600.0
    min_available_mb: float = 2500.0    # RAM guard for the tier-2 detector

    @classmethod
    def from_settings(cls, settings: Settings) -> "WatchConfig":
        c = settings.spotwatch
        d = cls()
        interval = min(120.0, max(30.0, float(c.get("interval_seconds", d.interval))))
        return cls(
            interval=interval,
            confirm_delay=float(c.get("confirm_delay_seconds", d.confirm_delay)),
            confirm_reads=int(c.get("confirm_reads", d.confirm_reads)),
            remind_every=float(c.get("remind_every_minutes", d.remind_every / 60)) * 60,
            remind_max=int(c.get("remind_max", d.remind_max)),
            send_snapshot=bool(c.get("send_snapshot", d.send_snapshot)),
            snapshot_quality=int(c.get("snapshot_quality", d.snapshot_quality)),
            subtype=int(c.get("subtype", d.subtype)),
            fail_threshold=int(c.get("fail_threshold", d.fail_threshold)),
            backoff_max=float(c.get("backoff_max_seconds", d.backoff_max)),
            state_max_age=float(c.get("state_max_age_hours", d.state_max_age / 3600)) * 3600,
            min_available_mb=float(settings.detection_tier2.get("min_available_mb", d.min_available_mb)),
        )


@dataclass
class ChannelHealth:
    failures: int = 0
    last_ok: float | None = None
    last_error: str = ""
    next_try: float = 0.0
    unreachable_sent: bool = False
    last_latency_ms: float | None = None   # grab + detect, last successful read


class Outbox:
    """Deliver messages; on failure keep them briefly in memory and retry.

    Telegram outage handling (plan section 8): the snapshot is dropped after
    ``image_ttl`` (text still goes out), the whole message after ``msg_ttl``.
    """

    def __init__(self, notifier: Notifier, clock: Callable[[], float] = time.time,
                 image_ttl: float = 300.0, msg_ttl: float = 1800.0, maxlen: int = 20):
        self._n = notifier
        self._clock = clock
        self.image_ttl = image_ttl
        self.msg_ttl = msg_ttl
        self.queue: deque[dict] = deque(maxlen=maxlen)

    async def _deliver(self, item: dict) -> None:
        kw = {"ack": True} if item.get("ack") else {}
        if item.get("jpegs") and len(item["jpegs"]) > 1:
            try:
                await self._n.send_media_group(item["jpegs"], item["text"])
                return
            except Exception as exc:  # noqa: BLE001 - fall back to a single photo
                log.warning("send_media_group failed (%s); trying one photo", exc)
                item = {**item, "jpeg": item["jpegs"][0], "jpegs": None}
        if item.get("jpegs"):
            item = {**item, "jpeg": item["jpegs"][0]}
        if item["jpeg"]:
            try:
                await self._n.send_photo(item["jpeg"], item["text"], **kw)
                return
            except Exception as exc:  # noqa: BLE001 - fall back to text-only
                log.warning("send_photo failed (%s); trying text", exc)
        await self._n.send_text(item["text"], silent=item["silent"], **kw)

    async def send(self, text: str, jpeg: bytes | None = None, silent: bool = False,
                   jpegs: list[bytes] | None = None, ack: bool = False) -> bool:
        item = {"text": text, "jpeg": jpeg, "jpegs": jpegs, "silent": silent, "ack": ack,
                "ts": self._clock()}
        try:
            await self._deliver(item)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram send failed, queued for retry: %s", exc)
            self.queue.append(item)
            return False

    async def flush(self) -> None:
        pending = list(self.queue)
        self.queue.clear()
        for i, item in enumerate(pending):
            age = self._clock() - item["ts"]
            if age > self.msg_ttl:
                log.warning("dropping queued message (%.0fs old)", age)
                continue
            if (item["jpeg"] or item.get("jpegs")) and age > self.image_ttl:
                item["jpeg"] = None
                item["jpegs"] = None
            try:
                await self._deliver(item)
            except Exception as exc:  # noqa: BLE001
                log.warning("retry failed, keeping %d queued: %s", len(pending) - i, exc)
                self.queue.extend(pending[i:])
                return


@dataclass
class CycleInfo:
    ts: float = 0.0
    duration_ms: float = 0.0
    kind: str = ""          # "full" | "confirm"
    channels: list[int] = field(default_factory=list)


class SpotWatch:
    def __init__(self, settings: Settings, controller: ModeController, store: StateStore,
                 capture, detector, verifier, stability, notifier: Notifier,
                 mem_available: Callable[[], float],
                 clock: Callable[[], float] = time.time,
                 reader: Callable[[int], Awaitable[list]] | None = None,
                 cfg: WatchConfig | None = None, tz=None):
        self.settings = settings
        self.cfg = cfg or WatchConfig.from_settings(settings)
        self.controller = controller
        self.store = store
        self.capture = capture
        self.detector = detector
        self.verifier = verifier
        self.stability = stability
        self.outbox = Outbox(notifier, clock)
        self._mem_available = mem_available
        self._clock = clock
        self._reader = reader or self._read_channel
        self._tz = tz
        self._lock = asyncio.Lock()
        self._started = clock()
        self._cpu0 = self._cpu_seconds()
        self.trackers: dict[str, SpotTracker] = {
            s.name: SpotTracker(s.name, self.cfg.confirm_reads,
                                self.cfg.remind_every, self.cfg.remind_max)
            for s in settings.spots
        }
        self.health: dict[int, ChannelHealth] = {
            ch: ChannelHealth() for ch in sorted({s.channel for s in settings.spots})
        }
        self.last_cycle = CycleInfo()
        self._next_full_at = 0.0
        self._restore_trackers()

    # ------------------------------------------------------------ persistence
    def _restore_trackers(self) -> None:
        now = self._clock()
        for name, tr in self.trackers.items():
            data, ts = self.store.get_with_ts(f"spot:{name}")
            if data is None or ts is None or now - ts > self.cfg.state_max_age:
                continue
            tr.restore(data)
            log.info("restored tracker %r: %s", name, tr.confirmed)

    def _save_tracker(self, tr: SpotTracker) -> None:
        try:
            self.store.set(f"spot:{tr.name}", tr.to_dict(), ts=self._clock())
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save tracker %r: %s", tr.name, exc)

    # ------------------------------------------------------------------ cycle
    def pending_confirm(self) -> bool:
        return any(t.needs_confirm for t in self.trackers.values())

    def kick(self) -> None:
        """Make the next cycle a full one (e.g. right after /watch)."""
        self._next_full_at = 0.0

    async def run_cycle(self) -> float:
        """Run one cycle; returns the delay (seconds) until the next one."""
        if self._lock.locked():
            return 1.0
        async with self._lock:
            return await self._run_cycle()

    async def _run_cycle(self) -> float:
        t0 = time.perf_counter()
        now = self._clock()
        if self.controller.tick():
            log.info("pause ended")
            await self.outbox.send("▶️ Duraklatma bitti, izleme devam ediyor.", silent=True)
        await self.outbox.flush()

        if self.controller.mode != Mode.UNKNOWN and not self.controller.watching:
            return self.cfg.interval   # GUARD (Phase 2) owns the cameras

        full = now >= self._next_full_at
        if full:
            wanted = list(self.health)
            self._next_full_at = now + self.cfg.interval
        else:
            wanted = sorted({s.channel for s in self.settings.spots
                             if self.trackers[s.name].needs_confirm})
        due = [ch for ch in wanted if self.health[ch].next_try <= now]

        results = []
        down: list[int] = []        # channels that just became unreachable
        back: list[int] = []        # channels that just recovered
        for ch in due:
            try:
                results.extend(await self._grab_and_detect(ch, back))
            except Exception as exc:  # noqa: BLE001 - any failure is "no reading"
                self._note_failure(ch, exc, down)
        # One message per event, however many cameras are involved (an XVR outage
        # takes every channel down in the same cycle).
        if down:
            names = ", ".join(self._label(c) for c in down)
            await self.outbox.send(
                f"📷⚠️ Kamera ulaşılamıyor: {names}. Park yeri durumu bilinmiyor.")
        if back:
            names = ", ".join(self._label(c) for c in back)
            await self.outbox.send(f"📷 Kamera tekrar erişilebilir: {names}.", silent=True)

        if results and self.controller.mode == Mode.UNKNOWN:
            self.controller.verify()
            self.controller.consume_entered_watch()   # restart: keep restored alerted state
        if self.controller.consume_entered_watch():
            for tr in self.trackers.values():
                tr.on_enter()
            log.info("SPOT_WATCH entered: trackers re-armed")

        for r in results:
            try:
                await self._process(r)
            except Exception:  # noqa: BLE001 - one spot must not kill the cycle
                log.exception("processing spot %r failed", r.spot.name)

        self.last_cycle = CycleInfo(ts=now, duration_ms=(time.perf_counter() - t0) * 1000,
                                    kind="full" if full else "confirm", channels=due)
        log.info("spotwatch cycle kind=%s mode=%s channels=%s ms=%.0f spots=%s",
                 self.last_cycle.kind, self.controller.mode.value, due,
                 self.last_cycle.duration_ms,
                 {n: t.confirmed for n, t in self.trackers.items()})
        return self._next_delay()

    def _next_delay(self) -> float:
        now = self._clock()
        until_full = max(1.0, self._next_full_at - now)
        # a free reading awaiting confirmation gets the quick ~10 s re-check
        delay = min(self.cfg.confirm_delay, until_full) if self.pending_confirm() else until_full
        # every channel backing off -> don't spin faster than the earliest retry
        retries = [h.next_try - now for h in self.health.values() if h.next_try > now]
        if retries and len(retries) == len(self.health):
            delay = max(delay, min(retries))
        return max(1.0, delay)

    # ------------------------------------------------------------- reading
    async def _grab_and_detect(self, ch: int, back: list[int]) -> list:
        t0 = time.perf_counter()
        results = await self._reader(ch)
        h = self.health[ch]
        h.last_latency_ms = (time.perf_counter() - t0) * 1000
        h.last_ok = self._clock()
        if h.unreachable_sent:
            back.append(ch)
        h.failures = 0
        h.unreachable_sent = False
        h.next_try = 0.0
        return results

    async def _read_channel(self, ch: int) -> list:
        spots = [s for s in self.settings.spots if s.channel == ch]
        expected = spots[0].frame_size if spots else None
        jpeg = await self.capture.grab(ch, self.cfg.subtype, expected_size=expected)

        def work():
            img = self.detector.decode(jpeg)
            full = self.detector.detect_array(img)
            return [self.detector.check_spot(img, s, full) for s in spots]

        return await asyncio.to_thread(work)

    def _label(self, ch: int) -> str:
        return self.settings.cameras.get(ch, f"Channel {ch}")

    def _note_failure(self, ch: int, exc: Exception, down: list[int]) -> None:
        """A failed grab is never a 'free' reading; after N in a row the channel is
        reported (once, via ``down``) and backed off."""
        h = self.health[ch]
        h.failures += 1
        h.last_error = str(exc)[:200] if isinstance(exc, CaptureError) else type(exc).__name__
        log.warning("grab failed ch=%s (%d in a row): %s", ch, h.failures, h.last_error)
        if h.failures >= self.cfg.fail_threshold:
            steps = min(h.failures - self.cfg.fail_threshold + 1, 6)
            h.next_try = self._clock() + min(self.cfg.backoff_max, self.cfg.interval * 2 ** steps)
            if not h.unreachable_sent:
                h.unreachable_sent = True
                down.append(ch)

    # ----------------------------------------------------------- processing
    async def _process(self, r) -> None:
        tr = self.trackers[r.spot.name]
        occupied = r.occupied

        # Gate 1: stability. Refreshes its reference on strong occupied readings and
        # overrides FREE -> OCCUPIED when the ROI is unchanged (night YOLO misses).
        if self.stability.check(r, tr.confirmed):
            occupied = True
            log.info("stable_override: %r", r.spot.name)

        # Gate 2: tier-2 detector, only for a free reading that could change state.
        if not occupied and tr.confirmed != FREE:
            mem = self._mem_available()
            if mem < self.cfg.min_available_mb:
                log.info("ram_defer: %r MemAvailable %.0f MB < %.0f MB, skipping reading",
                         r.spot.name, mem, self.cfg.min_available_mb)
                return   # inconclusive: neither free nor occupied
            t2 = await asyncio.to_thread(self.detector.check_spot_tier2, r.frame, r.spot)
            if t2 is not None and t2.occupied:
                occupied = True
                log.info("tier2_override: %r cov=%.2f", r.spot.name, t2.coverage)
                # Hand the strong tier-2 verdict to the stability gate as its
                # reference. Where plain YOLO never sees the car (ch3 at night) the
                # gate would otherwise never get one, and every cycle would pay for
                # tier-2 again; with a reference the next cycle is a cheap compare.
                self.stability.check(t2, tr.confirmed)

        events = tr.observe(occupied, self._clock(), muted=self.controller.muted)
        self._save_tracker(tr)
        for ev in events:
            await self._announce(r, tr, ev)

    async def _announce(self, r, tr: SpotTracker, ev: TrackerEvent) -> None:
        name = tr.name
        if ev.kind == "free_alert":
            unverified = False
            vcfg = self.settings.verify
            if self.verifier.enabled and bool(vcfg.get("gate", True)):
                verdict = await self.verifier.verify_spot(r.frame, r.spot)
                if verdict.ok and verdict.occupied:
                    log.info("vlm_veto: %r (conf %.2f: %s)", name, verdict.confidence, verdict.reason)
                    tr.veto_free()
                    self._save_tracker(tr)
                    return
                if not verdict.ok:
                    if not bool(vcfg.get("fail_open", True)):
                        log.warning("vlm unavailable, fail-closed: %r", name)
                        tr.veto_free()
                        self._save_tracker(tr)
                        return
                    unverified = True
            when = "şu anda" if ev.immediate else "şimdi"
            text = f"{'⚠️ (doğrulanamadı) ' if unverified else ''}🟢 {name} {when} BOŞ"
            jpeg = await self._snapshot(r) if self.cfg.send_snapshot else None
            await self.outbox.send(text, jpeg=jpeg)
            log.info("alert free_alert %r immediate=%s snapshot=%s", name, ev.immediate, bool(jpeg))
        elif ev.kind == "reminder":
            await self.outbox.send(
                f"🟢 {name} hâlâ BOŞ (hatırlatma {ev.n}/{self.cfg.remind_max})", silent=True)
            log.info("alert reminder %r %d", name, ev.n)
        elif ev.kind == "taken":
            await self.outbox.send(f"🔴 {name} tekrar DOLU", silent=True)
            log.info("alert taken %r", name)

    async def _snapshot(self, r) -> bytes | None:
        """JPEG for the alert, encoded in RAM on demand (D1). None on failure."""
        if r.frame is None:
            return None

        def encode():
            img = r.frame.copy()
            pts = np.array(r.spot.polygon, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 220, 0), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.snapshot_quality])
            return buf.tobytes() if ok else None

        try:
            return await asyncio.to_thread(encode)
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot encode failed: %s", exc)
            return None

    # --------------------------------------------------------------- status
    @staticmethod
    def _cpu_seconds() -> float:
        t = os.times()
        return t.user + t.system + t.children_user + t.children_system

    @staticmethod
    def _rss_mb() -> float | None:
        try:
            with open("/proc/self/status", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return None

    def _fmt_time(self, ts: float | None) -> str:
        if not ts:
            return "-"
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts, self._tz or timezone.utc).strftime("%H:%M:%S")

    def status_text(self) -> str:
        c = self.controller
        now = self._clock()
        lines = [f"Mod: {c.mode.value}" + (" (otomatik)" if c.auto else " (manuel)")]
        if c.mode == Mode.PAUSED:
            lines.append(f"Duraklatıldı: {c.pause_remaining / 60:.1f} dk kaldı")
        for s in self.settings.spots:
            t = self.trackers[s.name]
            extra = ""
            if t.confirmed == FREE and t.free_since:
                extra = f", {(now - t.free_since) / 60:.0f} dk'dır"
            lines.append(f"• {s.name}: {t.confirmed or 'bilinmiyor'}{extra}"
                         f"{' (teyit bekliyor)' if t.needs_confirm else ''}")
        lc = self.last_cycle
        lines.append(f"Son kontrol: {self._fmt_time(lc.ts)} ({lc.kind or '-'}, {lc.duration_ms:.0f} ms)")
        for ch, h in self.health.items():
            ok = f"{now - h.last_ok:.0f} sn önce" if h.last_ok else "hiç"
            lat = f", gecikme {h.last_latency_ms:.0f} ms" if h.last_latency_ms else ""
            err = f", HATA x{h.failures}: {h.last_error}" if h.failures else ""
            lines.append(f"RTSP {self._label(ch)}: son başarılı {ok}{lat}{err}")
        wall = max(1.0, now - self._started)
        cpu = (self._cpu_seconds() - self._cpu0) / wall * 100
        rss = self._rss_mb()
        mem = self._mem_available()
        lines.append(f"Bot: CPU ort. %{cpu:.1f} (1 çekirdek), RSS {rss:.0f} MB, "
                     f"host boş RAM {mem:.0f} MB" if rss is not None
                     else f"Bot: CPU ort. %{cpu:.1f}, host boş RAM {mem:.0f} MB")
        if self.outbox.queue:
            lines.append(f"Kuyrukta {len(self.outbox.queue)} gönderilemeyen mesaj")
        return "\n".join(lines)
