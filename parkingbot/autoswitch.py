"""Automatic mode switching ("teach the car"): enrollment + arrival matching.

Flow (plan section 5):
  /enroll         ~15 main-stream reads of the car in a spot -> appearance VECTORS stored
                  per spot and condition (day / night-IR / rain). Crops are never kept.
  arrival         SPOT_WATCH sees a confirmed free -> occupied flip. After the car has
                  settled, a few reads are scored against the references:
                    auto  -> GUARD on that spot, tell the owner, offer an undo button
                    ask   -> "Is this your car?" [Yes] [No]; no answer in 5 min -> GUARD
                             with a notice (decision D5)
                    other -> stay in SPOT_WATCH (someone else's car), silently
  every answer is a labelled sample, so the next match improves.

Only acts while automatic mode is on (/auto), the mode is SPOT_WATCH and alerts are not
paused. Manual /watch or /guard turns automatic mode off (manual always wins).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from . import appearance as ap
from .carfinder import find_car
from .config import Settings, Spot
from .matcher import MatchConfig, Matcher, MatchResult

log = logging.getLogger(__name__)

CB_PREFIX = "am:"


@dataclass
class AutoConfig:
    enabled: bool = True
    settle_s: float = 20.0          # let the car finish parking before judging it
    reads: int = 3
    read_gap_s: float = 5.0
    min_reads_ok: int = 2           # fewer reads with a car in the spot -> it left, abort
    ask_timeout_s: float = 300.0    # D5: unanswered question -> GUARD with a notice
    undo_window_s: float = 600.0
    enroll_samples: int = 15
    enroll_gap_s: float = 2.0
    enroll_min_found: int = 10
    box_area_ratio: tuple = (0.4, 1.2)   # car box / spot-polygon bbox; outside = junk detection
    match: MatchConfig = field(default_factory=MatchConfig)

    @classmethod
    def from_settings(cls, settings: Settings) -> "AutoConfig":
        c = settings.raw.get("auto", {}) or {}
        d = cls()
        m = c.get("match", {}) or {}
        dm = MatchConfig()
        return cls(
            enabled=bool(c.get("enabled", d.enabled)),
            settle_s=float(c.get("settle_seconds", d.settle_s)),
            reads=int(c.get("reads", d.reads)),
            read_gap_s=float(c.get("read_gap_seconds", d.read_gap_s)),
            min_reads_ok=int(c.get("min_reads_ok", d.min_reads_ok)),
            ask_timeout_s=float(c.get("ask_timeout_seconds", d.ask_timeout_s)),
            undo_window_s=float(c.get("undo_window_seconds", d.undo_window_s)),
            enroll_samples=int(c.get("enroll_samples", d.enroll_samples)),
            enroll_gap_s=float(c.get("enroll_gap_seconds", d.enroll_gap_s)),
            enroll_min_found=int(c.get("enroll_min_found", d.enroll_min_found)),
            box_area_ratio=tuple(c.get("box_area_ratio", d.box_area_ratio)),
            match=MatchConfig(
                t_hi=float(m.get("t_hi", dm.t_hi)), t_lo=float(m.get("t_lo", dm.t_lo)),
                top_k=int(m.get("top_k", dm.top_k)), min_refs=int(m.get("min_refs", dm.min_refs)),
                cross_t_lo=float(m.get("cross_condition_t_lo", dm.cross_t_lo)),
                neg_pull=float(m.get("negative_pull", dm.neg_pull)),
                geom_weight=float(m.get("geom_weight", dm.geom_weight)),
                weights=dict(m.get("weights", dm.weights)),
            ),
        )


@dataclass
class Pending:
    id: str
    spot_idx: int
    vec: object
    cond: str
    score: float
    kind: str                    # "ask" | "undo"
    ts: float
    task: asyncio.Task | None = None


class AutoSwitch:
    def __init__(self, settings: Settings, controller, matcher: Matcher, detector, capture,
                 guard, outbox, clock: Callable[[], float] = time.time,
                 cfg: AutoConfig | None = None):
        self.settings = settings
        self.cfg = cfg or AutoConfig.from_settings(settings)
        self.controller = controller
        self.matcher = matcher
        self.detector = detector
        self.capture = capture
        self.guard = guard
        self.outbox = outbox
        self._clock = clock
        self.pending: dict[str, Pending] = {}
        self._busy: set[str] = set()
        self.history: deque[dict] = deque(maxlen=20)
        self.outcomes = {"auto": 0, "ask": 0, "other": 0, "aborted": 0, "default_guard": 0,
                         "confirmed_mine": 0, "confirmed_other": 0, "undo": 0}

    # ------------------------------------------------------------------ reading
    def _spot_idx(self, name: str) -> int | None:
        return next((i for i, s in enumerate(self.settings.spots) if s.name == name), None)

    async def _read_car(self, spot: Spot, rain: bool = False):
        """One main-stream read -> (vector, condition, box) or None if no car is seen."""
        try:
            jpeg = await self.capture.grab(spot.channel, 0, expected_size=spot.frame_size)
        except Exception as exc:  # noqa: BLE001 - a failed grab is just "no read"
            log.warning("auto: grab failed for %r: %s", spot.name, exc)
            return None

        def work():
            img = self.detector.decode(jpeg)
            box = find_car(self.detector, img, spot)
            if box is None:
                return None
            if not self._sane_box(box, spot):
                log.info("auto: rejecting an implausible car box %s for %r", tuple(int(v) for v in box), spot.name)
                return None
            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                return None
            return ap.features(crop, box, img.shape[:2]), ap.condition_of(img, rain), box

        return await asyncio.to_thread(work)

    def _sane_box(self, box, spot: Spot) -> bool:
        """Reject junk detections (a sliver, or two cars merged into one box): the box must
        be a plausible fraction of the spot polygon's bounding box. In 30 h of ch1 crops this
        removed the only 2 bad boxes out of 49."""
        xs = [p[0] for p in spot.polygon]
        ys = [p[1] for p in spot.polygon]
        pa = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        ratio = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / pa
        lo, hi = self.cfg.box_area_ratio
        return lo <= ratio <= hi

    # ---------------------------------------------------------------- enrollment
    async def enroll(self, spot_idx: int, rain: bool = False) -> tuple[bool, str]:
        """Capture ~15 reads of the car in this spot and store their vectors."""
        spot = self.settings.spots[spot_idx]
        got: list[tuple] = []
        for i in range(self.cfg.enroll_samples):
            r = await self._read_car(spot, rain)
            if r is not None:
                got.append(r)
            if i < self.cfg.enroll_samples - 1:
                await asyncio.sleep(self.cfg.enroll_gap_s)
        if len(got) < self.cfg.enroll_min_found:
            # nothing is stored: a half-empty enrollment (car not in the spot?) would poison matching
            return False, (f"{spot.name}: {self.cfg.enroll_samples} okumada araç sadece "
                           f"{len(got)} kez net görüldü; kayıt için en az {self.cfg.enroll_min_found} gerekli. "
                           f"Araç bu yerde mi? Hiçbir şey kaydedilmedi, tekrar dene.")
        stored = []
        for vec, cond, box in got:
            self.matcher.store.add(spot.name, cond, vec, label=1, source="enroll",
                                   meta={"box": [round(v) for v in box]})
            stored.append(cond)
        c = self.matcher.store.counts(spot.name)
        summary = ", ".join(f"{k}: {v['pos']}" for k, v in c.items() if v["pos"])
        return True, (f"✅ {spot.name}: {len(stored)} örnek kaydedildi (yalnızca sayısal vektör, görüntü yok).\n"
                      f"Bu yerdeki toplam referans → {summary}.\n"
                      f"Gece ve yağmurda da tekrar çalıştır (/enroll, /enroll rain); koşul başına ≥10 hedef.")

    # ------------------------------------------------------------------ arrivals
    async def on_arrival(self, spot_name: str) -> None:
        """SPOT_WATCH saw a confirmed free -> occupied flip on ``spot_name``."""
        if not self.cfg.enabled or not self.controller.auto:
            return
        if not self.controller.watching or self.controller.muted:
            return
        idx = self._spot_idx(spot_name)
        if idx is None or spot_name in self._busy:
            return
        self._busy.add(spot_name)
        asyncio.get_running_loop().create_task(self._evaluate(idx))

    async def _evaluate(self, idx: int) -> None:
        spot = self.settings.spots[idx]
        try:
            await asyncio.sleep(self.cfg.settle_s)
            reads = []
            for i in range(self.cfg.reads):
                r = await self._read_car(spot)
                if r is not None:
                    reads.append(r)
                if i < self.cfg.reads - 1:
                    await asyncio.sleep(self.cfg.read_gap_s)
            if len(reads) < self.cfg.min_reads_ok:
                self.outcomes["aborted"] += 1
                self._note(spot.name, "aborted", 0.0, "car not stable in the spot")
                return
            if not self.controller.watching or self.controller.muted or not self.controller.auto:
                return                                  # the owner took over meanwhile
            results = [(self.matcher.score(spot.name, v, c), v, c) for v, c, _ in reads]
            results.sort(key=lambda t: t[0].score)
            res, vec, cond = results[len(results) // 2]       # median read
            await self._act(idx, res, vec, cond)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let an arrival crash the bot
            log.exception("auto: evaluation failed for %r", spot.name)
        finally:
            self._busy.discard(spot.name)

    def _note(self, spot: str, decision: str, score: float, reason: str) -> None:
        self.history.append({"ts": self._clock(), "spot": spot, "decision": decision,
                             "score": round(score, 3), "reason": reason})
        log.info("auto: %r -> %s (score %.3f; %s)", spot, decision, score, reason)

    async def _act(self, idx: int, res: MatchResult, vec, cond: str) -> None:
        spot = self.settings.spots[idx]
        self._note(spot.name, res.decision, res.score, res.reason)
        self.outcomes[res.decision] += 1
        if res.decision == "other":
            return
        pid = secrets.token_hex(3)
        if res.decision == "auto":
            ok, msg = await self.guard.arm(idx)
            if not ok:
                await self.outbox.send(f"ℹ️ BİLGİ · {spot.name}: araç tanındı ama GUARD kurulamadı. {msg}", silent=True)
                return
            self.controller.auto_enter_guard(spot.name)
            self.pending[pid] = Pending(pid, idx, vec, cond, res.score, "undo", self._clock())
            self.pending[pid].task = asyncio.get_running_loop().create_task(self._expire(pid, self.cfg.undo_window_s))
            await self.outbox.send(
                f"🛡️ {spot.name}: arabanı tanıdım (benzerlik %{res.score * 100:.0f}) → GUARD aktif.",
                buttons=[[("Bu benim arabam değil", f"{CB_PREFIX}no:{pid}")]], silent=True)
            return
        # ask
        p = Pending(pid, idx, vec, cond, res.score, "ask", self._clock())
        self.pending[pid] = p
        p.task = asyncio.get_running_loop().create_task(self._ask_timeout(pid))
        note = f" ({res.reason})" if res.reason and res.reason != "ok" else ""
        await self.outbox.send(
            f"🚗 {spot.name}: yeni bir araç park etti. Bu senin araban mı? (benzerlik %{res.score * 100:.0f}{note})\n"
            f"{self.cfg.ask_timeout_s / 60:.0f} dk içinde cevap vermezsen GUARD başlatırım.",
            buttons=[[("Evet, benim", f"{CB_PREFIX}yes:{pid}"), ("Hayır", f"{CB_PREFIX}no:{pid}")]])

    async def _expire(self, pid: str, after: float) -> None:
        await asyncio.sleep(after)
        self.pending.pop(pid, None)

    async def _ask_timeout(self, pid: str) -> None:
        """D5: nobody answered -> GUARD, with a notice."""
        await asyncio.sleep(self.cfg.ask_timeout_s)
        p = self.pending.pop(pid, None)
        if p is None or p.kind != "ask":
            return
        spot = self.settings.spots[p.spot_idx]
        if not self.controller.watching:
            return
        self.outcomes["default_guard"] += 1
        ok, msg = await self.guard.arm(p.spot_idx)
        if ok:
            self.controller.auto_enter_guard(spot.name)
            await self.outbox.send(f"⏱️ {spot.name}: cevap gelmedi, varsayılan olarak GUARD başlatıldı. Değilse /watch.")
        else:
            await self.outbox.send(f"ℹ️ BİLGİ · {spot.name}: cevap gelmedi, GUARD kurulamadı. {msg}", silent=True)
        self._note(spot.name, "default_guard", p.score, "no answer")

    # ------------------------------------------------------------------ answers
    async def answer(self, pid: str, mine: bool) -> str:
        """Inline-button answer. Returns the text to show the owner."""
        p = self.pending.pop(pid, None)
        if p is None:
            return "Bu soru artık geçerli değil."
        if p.task:
            p.task.cancel()
        spot = self.settings.spots[p.spot_idx]
        if p.kind == "undo":
            if mine:
                return "Tamam."
            self.matcher.learn(spot.name, p.vec, p.cond, mine=False, source="undo")
            self.outcomes["undo"] += 1
            await self.guard.disarm("undo")
            self.controller.auto_enter_watch()
            return f"↩️ {spot.name}: başka bir araç olarak kaydettim, SPOT_WATCH'a döndüm."
        self.matcher.learn(spot.name, p.vec, p.cond, mine=mine, source="confirm")
        if not mine:
            self.outcomes["confirmed_other"] += 1
            return f"👍 {spot.name}: başka araç olarak kaydedildi, SPOT_WATCH sürüyor."
        self.outcomes["confirmed_mine"] += 1
        ok, msg = await self.guard.arm(p.spot_idx)
        if not ok:
            return f"✅ Örnek kaydedildi ama GUARD kurulamadı: {msg}"
        self.controller.auto_enter_guard(spot.name)
        return f"✅ {spot.name}: örnek kaydedildi, GUARD aktif."

    # ------------------------------------------------------------------- status
    def status_text(self) -> str:
        c = self.matcher.cfg
        lines = [f"Otomatik mod: {'AÇIK' if self.controller.auto else 'kapalı'} "
                 f"(eşikler: ≥%{c.t_hi * 100:.0f} otomatik, %{c.t_lo * 100:.0f}-%{c.t_hi * 100:.0f} sorar)"]
        for s in self.settings.spots:
            cnt = self.matcher.store.counts(s.name)
            tot = ", ".join(f"{k} {v['pos']}+/{v['neg']}-" for k, v in cnt.items() if v["pos"] or v["neg"])
            lines.append(f"• {s.name} referans: {tot or 'yok (/enroll)'}")
        o = self.outcomes
        lines.append(f"Sonuçlar: otomatik {o['auto']}, soru {o['ask']}, başka araç {o['other']}, "
                     f"iptal {o['aborted']}, varsayılan GUARD {o['default_guard']}, "
                     f"onaylı {o['confirmed_mine']}/{o['confirmed_other']}, geri alma {o['undo']}")
        if self.pending:
            lines.append(f"Bekleyen soru: {len(self.pending)}")
        return "\n".join(lines)
