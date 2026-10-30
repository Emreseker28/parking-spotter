"""GUARD engine: watch the owner's parked car for suspicious activity.

Pipeline (plan section 4.1):

  persistent ffmpeg substream reader  --raw BGR frames-->  on_frame()  (every frame, cheap)
      evidence ring buffer (RAM)  +  stage-1 gates (host motion gate OR XVR event)
  latest-frame-wins one-slot buffer  -->  stage 2: YOLO on a crop window, only while
      active (motion / XVR hint / a live track), rate-capped by the duty-cycle limiter
  -->  tracker  -->  rules (LOITER / CONTACT / MULTI)  -->  dispatcher (Turkish alerts)

Nothing is written to disk: frames live in RAM arrays, the ffmpeg reader writes to a
pipe, evidence is JPEG-encoded in memory only when an alert actually fires.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .carfinder import find_car
from .config import ROOT, Settings, Spot
from .guard_detector import GuardDetector
from .guard_gate import MotionGate, XvrEventGate
from .guard_geometry import Box, Zones, build_zones, iou
from .guard_limiter import DutyLimiter
from .guard_presence import PresenceMonitor
from .guard_rules import ALERT, INFO, WARNING, GuardEvent, RuleConfig, RulesEngine
from .guard_tracker import Det, Tracker
from .metrics import NULL
from .modes import Mode, ModeController
from .night import is_night

log = logging.getLogger(__name__)

SEV_STYLE = {INFO: ("ℹ️", "BİLGİ"), WARNING: ("⚠️", "UYARI"), ALERT: ("🚨", "ALARM")}
RULE_TITLE = {"LOITER": "OYALANMA", "CONTACT": "TEMAS", "MULTI": "BİRDEN FAZLA KİŞİ"}


def format_event(ev: GuardEvent, spot_name: str) -> str:
    emoji, label = SEV_STYLE[ev.severity]
    head = f"{emoji} {label} · {RULE_TITLE.get(ev.rule, ev.rule)}"
    if ev.rule == "LOITER":
        body = f"{spot_name}: aracın çevresinde biri {ev.seconds:.0f} sn'dir bekliyor."
    elif ev.rule == "CONTACT":
        body = f"{spot_name}: biri araca temas ediyor/dayanmış ({ev.seconds:.0f} sn)."
    elif ev.rule == "MULTI":
        body = f"{spot_name}: aracın çevresinde {ev.persons} kişi var."
    else:
        body = spot_name
    return f"{head}\n{body}"


@dataclass
class GuardConfig:
    model_path: str = "models-guard/yolov8n_320.onnx"
    model_threads: int = 1
    conf: float = 0.30
    detect_fps: float = 4.0
    tail_s: float = 10.0                 # keep detecting this long after the last motion
    track_max_age: float = 3.0
    dil_w: float = 0.40
    dil_h: float = 0.25
    dilations: dict = field(default_factory=dict)    # spot name -> (dil_w, dil_h) override
    gate: str = "both"                   # motion | xvr | both (OR)
    xvr_codes: tuple = ("SmartMotionHuman",)   # vehicle events fire every few s on a street
    gate_min_frac: float = 0.004
    gate_global_frac: float = 0.40
    duty_limit: float = 0.40
    duty_window: float = 60.0
    recover_after: float = 300.0
    alert_repeat_s: float = 30.0         # ALERT repeats until /ack (D7) ...
    alert_repeat_max: int = 10           # ... capped so a forgotten alert cannot spam forever
    cooldown_s: float = 60.0             # per-rule minimum gap between alerts
    evidence_s: float = 8.0
    evidence_prev_frames: int = 3
    followup_main: bool = True           # one main-stream snapshot after an alert
    arm_reads: int = 5
    arm_gap_s: float = 1.0
    arm_min_hits: int = 3
    stall_timeout: float = 10.0
    backoff_max: float = 60.0
    sub_size: tuple[int, int] = (352, 288)
    sub_subtype: int = 1
    jpeg_quality: int = 80
    night_mode: str = "sun"
    night_lat: float = 41.0
    night_lon: float = 29.0
    night_margin_min: float = 30.0
    night_start_hour: int = 19
    night_end_hour: int = 6
    presence_check_s: float = 5.0        # car-still-there check cadence (substream, no YOLO)
    gone_sim: float = 0.30               # zone similarity below this = suspicious
    gone_checks: int = 3                 # consecutive low checks (~15 s) before confirming
    gone_confirm_reads: int = 2          # main-stream reads that must ALL miss the car
    gone_confirm_gap_s: float = 3.0
    gone_recheck_s: float = 15.0
    departure_s: float = 90.0            # paused + absent this long = the owner drove off
    gone_reply_s: float = 120.0          # no [Me]/[Not me] reply -> repeat louder
    gone_repeat_escalated_s: float = 15.0
    gone_repeat_max: int = 60
    perimeters: dict = field(default_factory=dict)   # spot name -> polygon (main-stream px)
    rules: RuleConfig = field(default_factory=RuleConfig)

    @classmethod
    def from_settings(cls, settings: Settings) -> "GuardConfig":
        c = settings.guard
        d = cls()
        r, n = c.get("rules", {}), c.get("night", {})
        ss = c.get("sub_size", d.sub_size)
        return cls(
            model_path=str(c.get("model_path", d.model_path)),
            model_threads=int(c.get("model_threads", d.model_threads)),
            conf=float(c.get("conf", d.conf)),
            detect_fps=float(c.get("detect_fps", d.detect_fps)),
            tail_s=float(c.get("tail_seconds", d.tail_s)),
            track_max_age=float(c.get("track_max_age_seconds", d.track_max_age)),
            dil_w=float(c.get("perimeter_dilate_w", d.dil_w)),
            dil_h=float(c.get("perimeter_dilate_h", d.dil_h)),
            dilations={k: (float(v.get("w", c.get("perimeter_dilate_w", d.dil_w))),
                           float(v.get("h", c.get("perimeter_dilate_h", d.dil_h))))
                       for k, v in (c.get("perimeter_margin", {}) or {}).items()},
            gate=str(c.get("gate", d.gate)),
            xvr_codes=tuple(c.get("xvr_codes", d.xvr_codes)),
            gate_min_frac=float(c.get("gate_min_frac", d.gate_min_frac)),
            gate_global_frac=float(c.get("gate_global_frac", d.gate_global_frac)),
            duty_limit=float(c.get("duty_limit", d.duty_limit)),
            duty_window=float(c.get("duty_window_seconds", d.duty_window)),
            recover_after=float(c.get("recover_after_seconds", d.recover_after)),
            alert_repeat_s=float(c.get("alert_repeat_seconds", d.alert_repeat_s)),
            alert_repeat_max=int(c.get("alert_repeat_max", d.alert_repeat_max)),
            cooldown_s=float(c.get("cooldown_seconds", d.cooldown_s)),
            evidence_s=float(c.get("evidence_seconds", d.evidence_s)),
            evidence_prev_frames=int(c.get("evidence_prev_frames", d.evidence_prev_frames)),
            followup_main=bool(c.get("followup_main_snapshot", d.followup_main)),
            arm_reads=int(c.get("arm_reads", d.arm_reads)),
            arm_gap_s=float(c.get("arm_gap_seconds", d.arm_gap_s)),
            arm_min_hits=int(c.get("arm_min_hits", d.arm_min_hits)),
            stall_timeout=float(c.get("stall_timeout_seconds", d.stall_timeout)),
            backoff_max=float(c.get("backoff_max_seconds", d.backoff_max)),
            sub_size=(int(ss[0]), int(ss[1])),
            sub_subtype=int(c.get("sub_subtype", d.sub_subtype)),
            jpeg_quality=int(c.get("jpeg_quality", d.jpeg_quality)),
            night_mode=str(n.get("mode", d.night_mode)),
            night_lat=float(n.get("lat", d.night_lat)),
            night_lon=float(n.get("lon", d.night_lon)),
            night_margin_min=float(n.get("margin_minutes", d.night_margin_min)),
            night_start_hour=int(n.get("start_hour", d.night_start_hour)),
            night_end_hour=int(n.get("end_hour", d.night_end_hour)),
            presence_check_s=float(c.get("presence_check_seconds", d.presence_check_s)),
            gone_sim=float(c.get("gone_similarity", d.gone_sim)),
            gone_checks=int(c.get("gone_checks", d.gone_checks)),
            gone_confirm_reads=int(c.get("gone_confirm_reads", d.gone_confirm_reads)),
            gone_confirm_gap_s=float(c.get("gone_confirm_gap_seconds", d.gone_confirm_gap_s)),
            gone_recheck_s=float(c.get("gone_recheck_seconds", d.gone_recheck_s)),
            departure_s=float(c.get("departure_seconds", d.departure_s)),
            gone_reply_s=float(c.get("gone_reply_seconds", d.gone_reply_s)),
            gone_repeat_escalated_s=float(c.get("gone_repeat_escalated_seconds", d.gone_repeat_escalated_s)),
            gone_repeat_max=int(c.get("gone_repeat_max", d.gone_repeat_max)),
            perimeters=dict(c.get("perimeters", {}) or {}),
            rules=RuleConfig(
                loiter_s=float(r.get("loiter_seconds", 10)),
                contact_s=float(r.get("contact_seconds", 3)),
                multi_s=float(r.get("multi_seconds", 2)),
                contact_overlap=float(r.get("contact_overlap", 0.10)),
                contact_center_margin=float(r.get("contact_center_margin", 0.06)),
                contact_min_hits=int(r.get("contact_min_hits", 3)),
                night_factor=float(r.get("night_factor", 0.6)),
            ),
        )


@dataclass
class ActiveAlert:
    text: str
    last_sent: float
    repeats: int = 0
    acked: bool = False
    buttons: list | None = None          # e.g. CAR_GONE [Me] / [Not me]
    kind: str = "alert"                  # "alert" | "gone"
    started: float = 0.0
    escalated: bool = False              # CAR_GONE unanswered / "not me": repeat louder


class GuardEngine:
    def __init__(self, settings: Settings, controller: ModeController, capture, detector,
                 outbox, clock: Callable[[], float] = time.time, cfg: GuardConfig | None = None,
                 guard_detector: GuardDetector | None = None, tz=None, source_factory=None,
                 metrics=NULL):
        self.settings = settings
        self.metrics = metrics
        self.cfg = cfg or GuardConfig.from_settings(settings)
        self.controller = controller
        self.capture = capture
        self.detector = detector          # existing vehicle Detector, used for ARMING only
        self.outbox = outbox
        self._clock = clock
        self._tz = tz
        mp = Path(self.cfg.model_path)
        self.gdet = guard_detector or GuardDetector(
            mp if mp.is_absolute() else ROOT / mp, self.cfg.model_threads, self.cfg.conf)
        self._source_factory = source_factory or self._spawn_ffmpeg
        self.limiter = DutyLimiter(self.cfg.duty_window, self.cfg.duty_limit,
                                   self.cfg.recover_after, self.cfg.detect_fps)
        self.armed = False
        self.spot: Spot | None = None
        self.zones: Zones | None = None
        self.window = (0, 0, 8, 8)
        self.gate: MotionGate | None = None
        self.xvr: XvrEventGate | None = None
        self.tracker = Tracker(max_age=self.cfg.track_max_age)
        self.rules = RulesEngine(self.cfg.rules)
        self.evidence: deque[tuple[float, np.ndarray]] = deque()
        self._tasks: list[asyncio.Task] = []
        self.presence: PresenceMonitor | None = None
        self.on_departure: Callable[[str], object] | None = None   # set by the bot
        self._gone_task: asyncio.Task | None = None
        self.gone_state: str | None = None        # None | "suspect" | "gone"
        self.gone_alerted = False
        self._slot: tuple[float, np.ndarray] | None = None
        self._evt = asyncio.Event()
        self._reset_stats()

    # ------------------------------------------------------------------ stats
    def _reset_stats(self) -> None:
        self.frames = self.dropped = self.skipped = self.detections_run = 0
        self.reconnects = 0
        self.last_frame_ts = 0.0
        self.last_motion_ts = 0.0
        self.last_detect_ts = 0.0
        self.global_rejects = 0
        self.alerts = {INFO: 0, WARNING: 0, ALERT: 0}
        self.suppressed = 0
        self.latencies: deque[float] = deque(maxlen=50)
        self.last_sent: dict[str, float] = {}
        self.active_alert: ActiveAlert | None = None
        self._night = False
        self._night_checked = 0.0
        self._frame_ts_prev = 0.0

    # --------------------------------------------------------------- arming
    def _pick_car(self, jpeg: bytes, spot: Spot) -> Box | None:
        """Owner's car box (main-stream px) for one arming read."""
        return find_car(self.detector, self.detector.decode(jpeg), spot)

    async def arm(self, spot_idx: int) -> tuple[bool, str]:
        """Find the owner's car in the chosen spot (median of ~5 reads), build the
        zones and start the pipeline. Returns (ok, Turkish message)."""
        if self.armed:
            await self.disarm("yeniden kuruluyor")
        spot = self.settings.spots[spot_idx]
        boxes: list[Box] = []
        for i in range(self.cfg.arm_reads):
            try:
                jpeg = await self.capture.grab(spot.channel, 0, expected_size=spot.frame_size)
                box = await asyncio.to_thread(self._pick_car, jpeg, spot)
            except Exception as exc:  # noqa: BLE001 - a bad read just doesn't count
                log.warning("arm read %d failed: %s", i + 1, exc)
                box = None
            if box:
                boxes.append(box)
            if i < self.cfg.arm_reads - 1:
                await asyncio.sleep(self.cfg.arm_gap_s)
        if len(boxes) < self.cfg.arm_min_hits:
            return False, (f"{spot.name} yerinde araç net görünmüyor "
                           f"({len(boxes)}/{self.cfg.arm_reads} okuma). GUARD başlatılmadı.")
        try:
            await asyncio.to_thread(self.gdet.ensure)
        except Exception as exc:  # noqa: BLE001
            log.error("guard model unavailable: %s", exc)
            return False, f"GUARD modeli yüklenemedi: {exc}"
        main_size = spot.frame_size or (960, 576)
        dil_w, dil_h = self.cfg.dilations.get(spot.name, (self.cfg.dil_w, self.cfg.dil_h))
        self.zones = build_zones(boxes, main_size, self.cfg.sub_size, dil_w, dil_h,
                                 self.cfg.perimeters.get(spot.name))
        self._start(spot)
        c = self.zones.car
        log.info("guard armed on %r: car(sub)=%s perimeter=%s window=%s reads=%d/%d",
                 spot.name, tuple(int(v) for v in c), tuple(int(v) for v in self.zones.perimeter_bbox),
                 self.window, len(boxes), self.cfg.arm_reads)
        return True, f"{spot.name}: araç bulundu, GUARD aktif."

    def _init_state(self, spot: Spot) -> None:
        """Fresh per-arming state (no tasks): zones must already be set."""
        assert self.zones is not None
        self._reset_stats()
        self.spot = spot
        self.window = self.zones.detect_window()
        self.gate = MotionGate(self.zones.perimeter_bbox, min_frac=self.cfg.gate_min_frac,
                               global_frac=self.cfg.gate_global_frac)
        self.tracker.reset()
        self.rules.reset()
        self.evidence.clear()
        self.limiter = DutyLimiter(self.cfg.duty_window, self.cfg.duty_limit,
                                   self.cfg.recover_after, self.cfg.detect_fps)
        self._slot = None
        self._evt.clear()
        sec = self.settings.secrets
        self.xvr = None
        if self.cfg.gate in ("xvr", "both"):
            self.xvr = XvrEventGate(sec.nvr_ip, sec.rtsp_user, sec.rtsp_pass, spot.channel,
                                    clock=self._clock, codes=self.cfg.xvr_codes)
        self.presence = PresenceMonitor(self.zones.car, self.cfg.presence_check_s, self.cfg.gone_sim,
                                        self.cfg.gone_checks)
        self.gone_state, self.gone_alerted, self._gone_task = None, False, None
        self.armed = True

    def _start(self, spot: Spot) -> None:
        self._init_state(spot)
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._source_loop(), name="guard-source"),
                       loop.create_task(self._consume_loop(), name="guard-consume"),
                       loop.create_task(self._repeat_loop(), name="guard-repeat")]
        if self.xvr is not None:
            self._tasks.append(loop.create_task(self.xvr.run(), name="guard-xvr"))

    async def disarm(self, reason: str = "") -> None:
        if not self.armed and not self._tasks:
            return
        self.armed = False
        if self._gone_task is not None:
            self._tasks.append(self._gone_task)
            self._gone_task = None
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks = []
        self.evidence.clear()            # never persisted, dropped on mode exit
        self.tracker.reset()
        self.rules.reset()
        self._slot = None
        self.active_alert = None
        self.gone_state, self.gone_alerted = None, False
        log.info("guard disarmed (%s)", reason)

    # ----------------------------------------------------------------- source
    async def _spawn_ffmpeg(self, channel: int):
        url = self.capture._rtsp_url(channel, self.cfg.sub_subtype)
        w, h = self.cfg.sub_size
        return await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", url, "-an",
            # passthrough: without it ffmpeg duplicates frames up to the stream's 100 fps
            # timebase (14x the real 7 fps), which cost ~40 % of a core in the first live run.
            "-fps_mode", "passthrough",
            "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    async def _source_loop(self) -> None:
        """One persistent reader with exponential-backoff reconnect and a stall
        watchdog (no frame for ``stall_timeout`` s => kill and reconnect)."""
        w, h = self.cfg.sub_size
        nbytes = w * h * 3
        backoff = 1.0
        while True:
            proc = None
            try:
                proc = await self._source_factory(self.spot.channel)
                first = True
                while True:
                    buf = await asyncio.wait_for(proc.stdout.readexactly(nbytes),
                                                 self.cfg.stall_timeout)
                    if first:
                        first, backoff = False, 1.0
                        log.info("guard source: receiving frames")
                    self.on_frame(np.frombuffer(buf, np.uint8).reshape(h, w, 3), self._clock())
            except asyncio.CancelledError:
                raise
            except (asyncio.IncompleteReadError, asyncio.TimeoutError, OSError) as exc:
                self.reconnects += 1
                self.metrics.reconnect()
                log.warning("guard source lost (%s); reconnecting in %.0fs",
                            type(exc).__name__, backoff)
            except Exception:  # noqa: BLE001
                self.reconnects += 1
                self.metrics.reconnect()
                log.exception("guard source crashed; reconnecting in %.0fs", backoff)
            finally:
                if proc is not None and proc.returncode is None:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.cfg.backoff_max)

    # ------------------------------------------------------------ frame path
    def _active(self, ts: float) -> bool:
        if ts - self.last_motion_ts <= self.cfg.tail_s:
            return True
        if self.xvr is not None and self.xvr.hint(ts, self.cfg.tail_s):
            return True
        return self.tracker.has_tracks()

    def on_frame(self, frame: np.ndarray, ts: float) -> None:
        """Every frame, on the event loop: must stay well under a millisecond or so.
        Evidence ring + stage-1 gate + latest-frame-wins hand-off."""
        self.frames += 1
        self.metrics.frame("received")
        self.last_frame_ts = ts
        self.evidence.append((ts, frame))
        while self.evidence and ts - self.evidence[0][0] > self.cfg.evidence_s:
            self.evidence.popleft()
        if self.gate is not None and self.cfg.gate in ("motion", "both"):
            g = self.gate.update(frame, active=self._active(ts))
            if g.global_change:
                self.global_rejects += 1
                self.metrics.global_reject()
            if g.motion:
                self.last_motion_ts = ts
        if self.presence is not None and self.presence.update(frame, ts) and self._gone_task is None:
            try:
                self._gone_task = asyncio.get_running_loop().create_task(self._gone_flow(), name="guard-gone")
            except RuntimeError:         # no running loop (unit tests feeding frames directly)
                pass
        if self._slot is not None:
            self.dropped += 1            # the detector never got to the previous frame
            self.metrics.frame("dropped")
        self._slot = (ts, frame)
        self._evt.set()

    async def _consume_loop(self) -> None:
        while True:
            await self._evt.wait()
            self._evt.clear()
            slot, self._slot = self._slot, None
            if slot is None:
                continue
            try:
                await self.process(slot[1], slot[0])
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a bad frame must not end GUARD
                log.exception("guard process crashed on a frame")

    def _is_night(self, ts: float) -> bool:
        if ts - self._night_checked > 30:
            c = self.cfg
            self._night = is_night(ts, self._tz, c.night_mode, c.night_lat, c.night_lon,
                                   c.night_margin_min, c.night_start_hour, c.night_end_hour)
            self._night_checked = ts
        return self._night

    async def process(self, frame: np.ndarray, ts: float) -> list[GuardEvent]:
        """Stage 2 -> tracker -> rules -> dispatch for one (latest) frame."""
        step_before = self.limiter.step
        msg = self.limiter.evaluate(ts)
        if msg:
            self.metrics.limiter_change("down" if self.limiter.step < step_before else "up")
            log.warning(msg)
            if self.gate is not None:
                self.gate.scale = self.limiter.gate_scale
            await self.outbox.send(f"ℹ️ {SEV_STYLE[INFO][1]} · {msg}", silent=True)
        if not self._active(ts):
            return []
        if ts - self.last_detect_ts < 1.0 / self.limiter.fps:
            self.skipped += 1
            self.metrics.frame("skipped")
            return []
        self.last_detect_ts = ts
        t0 = time.perf_counter()
        dets = await asyncio.to_thread(self.gdet.detect, frame, self.window)
        self.limiter.record(time.perf_counter() - t0, ts)
        self.detections_run += 1
        self.metrics.yolo()
        zones = self.zones
        keep: list[Det] = []
        for d in dets:
            if d.cls == 2 and iou(d.box, zones.car) > 0.4:
                continue                 # that is the owner's own car
            keep.append(d)
        tracks = self.tracker.update(keep, ts)
        persons = [t for t in tracks if t.cls == 0]
        events = self.rules.update(persons, zones, ts, self._is_night(ts), fps=self.limiter.fps)
        for ev in events:
            await self._dispatch(ev, frame, ts, persons)
        return events

    # ------------------------------------------------------------- dispatcher
    def _annotate(self, frame: np.ndarray, persons) -> np.ndarray:
        img = frame.copy()
        z = self.zones
        cv2.rectangle(img, tuple(int(v) for v in z.car[:2]), tuple(int(v) for v in z.car[2:]),
                      (0, 200, 0), 1)
        cv2.polylines(img, [np.array(z.perimeter, np.int32).reshape(-1, 1, 2)], True, (0, 200, 255), 1)
        for t in persons:
            cv2.rectangle(img, tuple(int(v) for v in t.box[:2]), tuple(int(v) for v in t.box[2:]),
                          (0, 0, 255), 2)
        return img

    def _build_evidence(self, frame: np.ndarray, ts: float, persons) -> list[bytes]:
        """Annotated key frame + up to N preceding frames, JPEG-encoded in RAM now
        (no CPU is spent on encoding until an alert actually fires)."""
        imgs = [self._annotate(frame, persons)]
        prev = [f for t, f in self.evidence if t < ts - 0.4]
        if prev:
            n = min(self.cfg.evidence_prev_frames, len(prev))
            step = len(prev) / n
            imgs += [prev[min(len(prev) - 1, int(i * step))] for i in range(n)]
        out = []
        for im in imgs:
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if ok:
                out.append(buf.tobytes())
        return out

    async def _dispatch(self, ev: GuardEvent, frame: np.ndarray, ts: float, persons) -> None:
        sev = ev.severity
        self.alerts[sev] += 1
        text = format_event(ev, self.spot.name)
        now = self._clock()
        if self.controller.muted:
            self.suppressed += 1
            self.metrics.suppressed("paused")
            log.info("guard %s %s suppressed (paused)", sev, ev.rule)
            return
        if now - self.last_sent.get(ev.rule, -1e9) < self.cfg.cooldown_s:
            self.metrics.suppressed("cooldown")
            log.info("guard %s %s skipped (cooldown)", sev, ev.rule)
            return
        self.last_sent[ev.rule] = now
        jpegs = await asyncio.to_thread(self._build_evidence, frame, ts, list(persons))
        await self.outbox.send(text, jpegs=jpegs or None, silent=(sev == INFO),
                               ack=(sev == ALERT and not (self.cfg.followup_main and jpegs)))
        latency = self._clock() - ev.cond_ts
        self.latencies.append(latency)
        self.metrics.alert(sev, ev.rule)
        self.metrics.rule_latency(latency)
        log.info("guard alert %s %s spot=%r held=%.1fs latency=%.2fs frames=%d",
                 sev, ev.rule, self.spot.name, ev.seconds, latency, len(jpegs))
        if sev == ALERT:
            self.active_alert = ActiveAlert(text, self._clock())
        if self.cfg.followup_main:
            asyncio.get_running_loop().create_task(self._followup(sev == ALERT))

    async def _followup(self, with_ack: bool) -> None:
        """One main-stream snapshot after an alert: the substream evidence is only
        352x288, this gives a human something to actually judge."""
        try:
            jpeg = await self.capture.grab(self.spot.channel, 0, expected_size=self.spot.frame_size)
            await self.outbox.send("Yüksek çözünürlüklü görüntü", jpeg=jpeg, silent=True, ack=with_ack)
        except Exception as exc:  # noqa: BLE001 - best effort
            log.warning("guard follow-up snapshot failed: %s", exc)
            if with_ack:
                await self.outbox.send("Görüldü olarak işaretlemek için /ack", silent=True, ack=True)

    # ------------------------------------------------------- CAR_GONE / departure
    async def _main_sees_car(self) -> bool | None:
        """Confirm with main-stream reads: True if any read finds the car, False if every
        read misses it, None if the camera could not be read at all."""
        got_any = False
        for i in range(self.cfg.gone_confirm_reads):
            try:
                jpeg = await self.capture.grab(self.spot.channel, 0, expected_size=self.spot.frame_size)
                box = await asyncio.to_thread(self._pick_car, jpeg, self.spot)
            except Exception as exc:  # noqa: BLE001
                log.warning("gone confirm read failed: %s", exc)
                box = None
            else:
                got_any = True
            if box is not None:
                return True
            if i < self.cfg.gone_confirm_reads - 1:
                await asyncio.sleep(self.cfg.gone_confirm_gap_s)
        return False if got_any else None

    async def _gone_flow(self) -> None:
        """The substream says the car zone changed. Confirm on the main stream; then either
        treat a paused owner's absence as a departure, or raise CAR_GONE."""
        try:
            self.gone_state = "suspect"
            absent_since = None
            while self.armed:
                seen = await self._main_sees_car()
                if seen:
                    log.info("guard: car is still there (false alarm: sim=%.2f); reference refreshed",
                             self.presence.sim or 0.0)
                    if self.slot_frame() is not None:
                        self.presence.reseed(self.slot_frame(), self._clock())
                    if self.gone_alerted:
                        await self.outbox.send(f"ℹ️ {SEV_STYLE[INFO][1]} · {self.spot.name}: araç yerine döndü.", silent=True)
                        self.active_alert = None
                    self.gone_state, self.gone_alerted = None, False
                    return
                if seen is None:                     # cannot see the camera: no verdict, no alarm
                    await asyncio.sleep(self.cfg.gone_recheck_s)
                    continue
                now = self._clock()
                absent_since = absent_since or now
                self.gone_state = "gone"
                if self.controller.muted:            # owner paused before leaving: quiet departure
                    if now - absent_since >= self.cfg.departure_s:
                        self._depart("pause")
                        return
                elif not self.gone_alerted:
                    await self._raise_car_gone(absent_since)
                await asyncio.sleep(self.cfg.gone_recheck_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("guard gone-flow crashed")
        finally:
            self._gone_task = None
            if self.gone_state == "suspect":
                self.gone_state = None

    def slot_frame(self):
        return self.evidence[-1][1] if self.evidence else None

    def _depart(self, reason: str) -> None:
        """Hand over to the bot (separate task: the bot disarms us, which cancels tasks)."""
        self.gone_state = None
        cb = self.on_departure
        if cb is not None:
            asyncio.get_running_loop().create_task(cb(reason))

    async def _raise_car_gone(self, absent_since: float) -> None:
        now = self._clock()
        self.gone_alerted = True
        self.alerts[ALERT] += 1
        text = (f"{SEV_STYLE[ALERT][0]} {SEV_STYLE[ALERT][1]} · ARAÇ GİTTİ\n{self.spot.name}: araç yerinde değil! "
                f"Sen aldıysan \"Bendim\" de.")
        buttons = [[("Bendim", "cg:me"), ("Ben değilim", "cg:notme")]]
        frame = self.slot_frame()
        jpegs = []
        if frame is not None:
            img = frame.copy()
            z = self.zones.car
            cv2.rectangle(img, (int(z[0]), int(z[1])), (int(z[2]), int(z[3])), (0, 0, 255), 2)
            ts0 = self.evidence[-1][0]
            earlier = [f for t, f in self.evidence if t < ts0 - 4.0]
            for im in [img] + ([earlier[0], earlier[len(earlier) // 2]] if len(earlier) >= 2 else []):
                ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
                if ok:
                    jpegs.append(buf.tobytes())
        await self.outbox.send(text, jpegs=jpegs or None, buttons=buttons)
        self.latencies.append(self._clock() - now)
        self.metrics.alert(ALERT, "CAR_GONE")
        self.metrics.rule_latency(self._clock() - now)
        self.active_alert = ActiveAlert(text, self._clock(), buttons=buttons, kind="gone", started=now)
        log.info("guard alert ALERT CAR_GONE spot=%r absent for %.0fs", self.spot.name, now - absent_since)

    async def owner_departure(self) -> bool:
        """[Me] / the owner drove off: stop alarming and hand over to SPOT_WATCH."""
        if not self.armed:
            return False
        self.active_alert = None
        self._depart("owner")
        return True

    def escalate(self) -> bool:
        """[Not me]: repeat louder right away."""
        a = self.active_alert
        if a is None or a.kind != "gone":
            return False
        a.escalated = True
        a.last_sent = 0.0
        return True

    def ack(self) -> bool:
        """/ack: stop an ALERT from repeating. True if there was one."""
        a = self.active_alert
        if a is None or a.acked:
            return False
        a.acked = True
        log.info("guard alert acknowledged")
        return True

    async def _repeat_loop(self) -> None:
        """ALERT repeats every ``alert_repeat_s`` until /ack (D7), capped. An unanswered
        CAR_GONE (no [Me]/[Not me] within ``gone_reply_s``) repeats louder and faster."""
        while True:
            await asyncio.sleep(1.0)
            a = self.active_alert
            if a is None or a.acked or self.controller.muted:
                continue
            now = self._clock()
            if a.kind == "gone" and not a.escalated and now - a.started >= self.cfg.gone_reply_s:
                a.escalated = True
            every = self.cfg.gone_repeat_escalated_s if a.escalated else self.cfg.alert_repeat_s
            if now - a.last_sent < every:
                continue
            cap = self.cfg.gone_repeat_max if a.kind == "gone" else self.cfg.alert_repeat_max
            if a.repeats >= cap:
                self.active_alert = None
                await self.outbox.send("ℹ️ BİLGİ · Alarm tekrarı durduruldu (onaylanmadı). /status", silent=True)
                continue
            a.repeats += 1
            a.last_sent = now
            head = "‼️ ACİL — " if a.escalated else ""
            await self.outbox.send(f"{head}{a.text}\n(tekrar {a.repeats}/{cap})", ack=a.buttons is None,
                                   buttons=a.buttons)

    # ----------------------------------------------------------------- status
    def status_text(self) -> str:
        if not self.armed:
            return "GUARD: kapalı"
        now = self._clock()
        age = now - self.last_frame_ts if self.last_frame_ts else float("inf")
        lat = list(self.latencies)
        lat_s = (f"son {lat[-1]:.1f} sn, medyan {statistics.median(lat):.1f} sn, "
                 f"en çok {max(lat):.1f} sn ({len(lat)} olay)") if lat else "olay yok"
        active = "aktif" if self._active(now) else "boşta"
        lines = [
            f"GUARD: {self.spot.name} ({'DURAKLATILDI, ' if self.controller.muted else ''}{active})",
            f"Kare yaşı: {age:.1f} sn · kare {self.frames}, atlanan {self.dropped + self.skipped}, "
            f"yeniden bağlanma {self.reconnects}",
            f"YOLO: {self.limiter.duty(now):.0%} yük, basamak {self.limiter.step} "
            f"({self.limiter.fps:g} kare/sn), çalışma {self.detections_run}",
            f"Kapı: {self.cfg.gate}; XVR olayı {'bağlı' if self.xvr and self.xvr.connected else 'yok'} "
            f"({self.xvr.count if self.xvr else 0}); global değişim reddi {self.global_rejects}",
            f"Gecikme: {lat_s}",
            f"Uyarılar: bilgi {self.alerts[INFO]}, uyarı {self.alerts[WARNING]}, alarm {self.alerts[ALERT]}"
            f"{', bastırılan ' + str(self.suppressed) if self.suppressed else ''}",
        ]
        if self.presence is not None and self.presence.sim is not None:
            lines.append(f"Araç yerinde mi: benzerlik {self.presence.sim:.2f}"
                         f"{' — ŞÜPHELİ' if self.gone_state else ''}"
                         f"{' (ARAÇ GİTTİ alarmı aktif)' if self.gone_alerted else ''}")
        if self.active_alert and not self.active_alert.acked:
            lines.append("⚠️ Onaylanmamış alarm var → /ack")
        return "\n".join(lines)
