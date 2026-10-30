"""Prometheus metrics (text and numbers only: no image bytes, no secrets, no names beyond spot labels).

Two kinds of series:
- **instrumented** counters / histograms, bumped at the event (alerts by severity, rule latency,
  RTSP reconnects, dropped frames, limiter steps, gate overrides, Telegram send failures);
- **state** gauges read at scrape time by a custom collector from the live objects (mode, spot
  states, frame age, YOLO duty, limiter step, camera health, auto-switch outcomes, references,
  host RAM/swap, container CPU/RSS/throttling).

The engines take an optional ``metrics`` and default to ``NULL`` (no-ops), so nothing depends on
Prometheus being enabled and the unit tests need no registry.
"""

from __future__ import annotations

import logging
import os
import time

from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.process_collector import ProcessCollector

log = logging.getLogger(__name__)

NAMESPACE = "parking"
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30)
CYCLE_BUCKETS = (0.5, 1, 2, 3, 4, 6, 10, 20, 60)


class _Null:
    """No-op stand-in: every instrumentation call does nothing."""

    def __getattr__(self, name):
        return lambda *a, **k: None


NULL = _Null()


class BotMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)             # process_cpu_seconds_total, RSS, fds
        r = self.registry
        self._alerts = Counter(f"{NAMESPACE}_guard_alerts_total", "GUARD alerts raised", ["severity", "rule"], registry=r)
        self._suppressed = Counter(f"{NAMESPACE}_guard_alerts_suppressed_total", "GUARD alerts held back", ["reason"], registry=r)
        self._latency = Histogram(f"{NAMESPACE}_guard_rule_latency_seconds",
                                  "Rule condition met -> Telegram message sent", buckets=LATENCY_BUCKETS, registry=r)
        self._reconnects = Counter(f"{NAMESPACE}_guard_rtsp_reconnects_total", "GUARD substream reader reconnects", registry=r)
        self._frames = Counter(f"{NAMESPACE}_guard_frames_total", "Substream frames", ["outcome"], registry=r)
        self._yolo = Counter(f"{NAMESPACE}_guard_yolo_runs_total", "GUARD detector runs", registry=r)
        self._limiter = Counter(f"{NAMESPACE}_guard_limiter_changes_total", "Duty-cycle limiter step changes",
                                ["direction"], registry=r)
        self._global = Counter(f"{NAMESPACE}_guard_global_change_rejects_total", "Headlight/IR-style whole-scene changes ignored", registry=r)
        self._cycle = Histogram(f"{NAMESPACE}_spotwatch_cycle_seconds", "SPOT_WATCH cycle wall time",
                                buckets=CYCLE_BUCKETS, registry=r)
        self._spot_alerts = Counter(f"{NAMESPACE}_spotwatch_alerts_total", "SPOT_WATCH messages sent", ["kind"], registry=r)
        self._gates = Counter(f"{NAMESPACE}_spotwatch_gate_overrides_total",
                              "A free reading overruled before it could alert", ["gate"], registry=r)
        self._grab_fail = Counter(f"{NAMESPACE}_camera_grab_failures_total", "Failed main-stream grabs", ["channel"], registry=r)
        self._send_fail = Counter(f"{NAMESPACE}_telegram_send_failures_total", "Telegram sends that failed and were queued", registry=r)
        self._started = time.time()

    # ------------------------------------------------------------ instrumentation
    def alert(self, severity: str, rule: str) -> None:
        self._alerts.labels(severity, rule).inc()

    def suppressed(self, reason: str) -> None:
        self._suppressed.labels(reason).inc()

    def rule_latency(self, seconds: float) -> None:
        self._latency.observe(max(0.0, seconds))

    def reconnect(self) -> None:
        self._reconnects.inc()

    def frame(self, outcome: str, n: int = 1) -> None:
        self._frames.labels(outcome).inc(n)

    def yolo(self) -> None:
        self._yolo.inc()

    def limiter_change(self, direction: str) -> None:
        self._limiter.labels(direction).inc()

    def global_reject(self) -> None:
        self._global.inc()

    def cycle(self, seconds: float) -> None:
        self._cycle.observe(seconds)

    def spot_alert(self, kind: str) -> None:
        self._spot_alerts.labels(kind).inc()

    def gate(self, name: str) -> None:
        self._gates.labels(name).inc()

    def grab_failure(self, channel: int) -> None:
        self._grab_fail.labels(str(channel)).inc()

    def send_failure(self) -> None:
        self._send_fail.inc()

    # -------------------------------------------------------------------- state
    def attach(self, bot) -> None:
        """Register the scrape-time collector that reads the live bot objects."""
        self.registry.register(_StateCollector(bot, self._started))


def _read_cgroup() -> dict:
    """Container-level numbers (include the ffmpeg children). cgroup v2; missing files are skipped."""
    out: dict = {}
    try:
        with open("/sys/fs/cgroup/cpu.stat", encoding="ascii") as fh:
            for line in fh:
                k, v = line.split()
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    for name in ("memory.current", "memory.peak", "memory.swap.current"):
        try:
            with open(f"/sys/fs/cgroup/{name}", encoding="ascii") as fh:
                out[name] = int(fh.read().split()[0])
        except (OSError, ValueError, IndexError):
            pass
    return out


def _meminfo() -> dict:
    out = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


class _StateCollector:
    def __init__(self, bot, started: float):
        self.bot = bot
        self.started = started

    def collect(self):
        for section in (self._mode, self._spots, self._cameras, self._guard, self._auto, self._host):
            try:
                yield from section()
            except Exception as exc:  # noqa: BLE001 - one bad section must never break the scrape
                log.warning("metrics section %s failed: %s", section.__name__, exc)

    def _g(self, name, doc, labels=()):
        return GaugeMetricFamily(f"{NAMESPACE}_{name}", doc, labels=list(labels))

    def _mode(self):
        c = self.bot.controller
        if c is None:
            return
        from .modes import Mode
        g = self._g("mode", "Current mode (1 = active)", ["mode"])
        for m in Mode:
            g.add_metric([m.value], 1.0 if c.mode == m else 0.0)
        yield g
        a = self._g("auto_enabled", "Automatic mode switching on (1) / off (0)")
        a.add_metric([], 1.0 if c.auto else 0.0)
        yield a
        p = self._g("pause_remaining_seconds", "Seconds left on a /pause snooze")
        p.add_metric([], c.pause_remaining)
        yield p
        u = self._g("uptime_seconds", "Seconds since the bot process started")
        u.add_metric([], time.time() - self.started)
        yield u

    def _spots(self):
        sw = self.bot.spotwatch
        if sw is None:
            return
        st = self._g("spot_state", "Confirmed spot state: 1 occupied, 0 free, -1 unknown", ["spot"])
        al = self._g("spot_alerted", "A free alert is currently outstanding for the spot", ["spot"])
        for name, t in sw.trackers.items():
            st.add_metric([name], {"occupied": 1.0, "free": 0.0}.get(t.confirmed, -1.0))
            al.add_metric([name], 1.0 if t.alerted else 0.0)
        yield st
        yield al
        lc = sw.last_cycle
        ts = self._g("spotwatch_last_cycle_timestamp_seconds", "Unix time of the last SPOT_WATCH cycle")
        ts.add_metric([], lc.ts)
        yield ts
        co = CounterMetricFamily(f"{NAMESPACE}_spotwatch_cross_overrides", "Cross-camera vetoes of a free reading")
        co.add_metric([], sw.cross_overrides)
        yield co
        q = self._g("telegram_outbox_queued", "Messages waiting for a retry")
        q.add_metric([], len(sw.outbox.queue))
        yield q

    def _cameras(self):
        sw = self.bot.spotwatch
        if sw is None:
            return
        now = time.time()
        age = self._g("camera_last_success_age_seconds", "Seconds since the last good main-stream read", ["channel"])
        fails = self._g("camera_consecutive_failures", "Failed grabs in a row", ["channel"])
        lat = self._g("camera_read_latency_seconds", "Last grab + detect time", ["channel"])
        for ch, h in sw.health.items():
            age.add_metric([str(ch)], (now - h.last_ok) if h.last_ok else -1.0)
            fails.add_metric([str(ch)], h.failures)
            lat.add_metric([str(ch)], (h.last_latency_ms or 0.0) / 1000.0)
        yield age
        yield fails
        yield lat

    def _guard(self):
        g = self.bot.guard
        if g is None:
            return
        armed = self._g("guard_armed", "GUARD armed on the spot (1) / not armed (0)", ["spot"])
        for sp in self.bot.settings.spots:
            armed.add_metric([sp.name], 1.0 if (g.armed and g.spot and g.spot.name == sp.name) else 0.0)
        yield armed
        if not g.armed:
            return
        now = time.time()
        for name, doc, val in (
            ("guard_frame_age_seconds", "Age of the newest substream frame", (now - g.last_frame_ts) if g.last_frame_ts else -1.0),
            ("guard_yolo_duty_ratio", "Share of the last 60 s the detector was busy", g.limiter.duty(now)),
            ("guard_limiter_step", "Duty-cycle limiter step (0 = normal)", g.limiter.step),
            ("guard_detect_fps_limit", "Current YOLO rate cap", g.limiter.fps),
            ("guard_evidence_frames", "Frames in the RAM evidence ring", len(g.evidence)),
            ("guard_tracks", "Live tracks", len(g.tracker.tracks)),
            ("guard_active", "Detector currently active (motion / event / track)", 1.0 if g._active(now) else 0.0),
            ("guard_xvr_events_connected", "XVR event stream attached", 1.0 if (g.xvr and g.xvr.connected) else 0.0),
            ("guard_car_zone_similarity", "Similarity of the car zone to its reference (-1 unknown)",
             g.presence.sim if (g.presence and g.presence.sim is not None) else -1.0),
            ("guard_car_gone_alarm", "ARAÇ GİTTİ alarm is active", 1.0 if g.gone_alerted else 0.0),
            ("guard_unacknowledged_alert", "An ALERT is waiting for /ack",
             1.0 if (g.active_alert and not g.active_alert.acked) else 0.0),
        ):
            m = self._g(name, doc)
            m.add_metric([], float(val))
            yield m
        xe = CounterMetricFamily(f"{NAMESPACE}_guard_xvr_events", "Relevant XVR events seen since arming")
        xe.add_metric([], g.xvr.count if g.xvr else 0)
        yield xe

    def _auto(self):
        a = self.bot.auto
        if a is None:
            return
        oc = CounterMetricFamily(f"{NAMESPACE}_auto_switch_outcomes", "Auto-switch outcomes", labels=["outcome"])
        for k, v in a.outcomes.items():
            oc.add_metric([k], v)
        yield oc
        pend = self._g("auto_pending_questions", "Unanswered 'is this your car?' questions")
        pend.add_metric([], len(a.pending))
        yield pend
        refs = self._g("car_references", "Enrolled appearance vectors", ["spot", "condition", "label"])
        for sp in self.bot.settings.spots:
            for cond, n in a.matcher.store.counts(sp.name).items():
                refs.add_metric([sp.name, cond, "owner"], n["pos"])
                refs.add_metric([sp.name, cond, "other"], n["neg"])
        yield refs

    def _host(self):
        mi = _meminfo()
        if mi:
            m = self._g("host_mem_available_bytes", "Host MemAvailable")
            m.add_metric([], mi.get("MemAvailable", 0))
            yield m
            sw = self._g("host_swap_used_bytes", "Host swap in use")
            sw.add_metric([], mi.get("SwapTotal", 0) - mi.get("SwapFree", 0))
            yield sw
        t = os.times()
        cpu = CounterMetricFamily(f"{NAMESPACE}_cpu_seconds", "CPU seconds used by the bot and its ffmpeg children")
        cpu.add_metric([], t.user + t.system + t.children_user + t.children_system)
        yield cpu
        cg = _read_cgroup()
        for key, name, doc in (("memory.current", "container_memory_bytes", "Container memory in use"),
                               ("memory.peak", "container_memory_peak_bytes", "Container memory peak"),
                               ("memory.swap.current", "container_swap_bytes", "Container swap in use")):
            if key in cg:
                m = self._g(name, doc)
                m.add_metric([], cg[key])
                yield m
        for key, name, doc in (("usage_usec", "container_cpu_usage_seconds", "Container CPU time (incl. ffmpeg)"),
                               ("nr_throttled", "container_cpu_throttled_periods", "CFS periods throttled (must stay 0: cpuset, no quota)")):
            if key in cg:
                m = CounterMetricFamily(f"{NAMESPACE}_{name}", doc)
                m.add_metric([], cg[key] / 1e6 if key == "usage_usec" else cg[key])
                yield m


def start_server(bot, settings):
    """Start the /metrics HTTP endpoint (a daemon thread). Never fatal: returns the server, or
    None (and logs) if it is disabled or the port is taken."""
    cfg = settings.raw.get("metrics", {}) or {}
    if not cfg.get("enabled", False):
        return None
    addr, port = str(cfg.get("bind", "0.0.0.0")), int(cfg.get("port", 9108))
    try:
        bot.metrics.attach(bot)
        server, _thread = start_http_server(port, addr=addr, registry=bot.metrics.registry)
    except OSError as exc:
        log.warning("metrics endpoint not started on %s:%s: %s", addr, port, exc)
        return None
    log.info("metrics endpoint on http://%s:%s/metrics", addr, port)
    return server
