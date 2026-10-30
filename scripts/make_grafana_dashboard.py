"""Generate docs/grafana-parking-spotter.json (importable Grafana dashboard).
Run:  python scripts/make_grafana_dashboard.py"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
_id = 0


def _next():
    global _id
    _id += 1
    return _id


def target(expr, legend="", instant=False):
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": chr(64 + target.n)}
    target.n += 1
    if instant:
        t["instant"] = True
    return t


target.n = 1


def panel(kind, title, targets, x, y, w, h, unit="short", desc="", mappings=None, thresholds=None,
          minv=None, maxv=None, extra=None):
    target.n = 1
    ts = [target(*t) if isinstance(t, tuple) else target(t) for t in targets]
    defaults = {"unit": unit, "mappings": mappings or [],
                "thresholds": thresholds or {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}
    if minv is not None:
        defaults["min"] = minv
    if maxv is not None:
        defaults["max"] = maxv
    p = {"id": _next(), "type": kind, "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": ts,
         "fieldConfig": {"defaults": defaults, "overrides": []}, "options": {}}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "background", "graphMode": "none", "textMode": "auto"}
    if kind == "timeseries":
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
    if extra:
        p.update(extra)
    return p


mode_map = [{"type": "value", "options": {"1": {"text": "ACTIVE", "color": "green"}, "0": {"text": "-", "color": "text"}}}]
spot_map = [{"type": "value", "options": {"1": {"text": "DOLU", "color": "red"}, "0": {"text": "BOŞ", "color": "green"},
                                           "-1": {"text": "bilinmiyor", "color": "yellow"}}}]
onoff = [{"type": "value", "options": {"1": {"text": "AÇIK", "color": "green"}, "0": {"text": "kapalı", "color": "text"}}}]
red_on = [{"type": "value", "options": {"1": {"text": "VAR", "color": "red"}, "0": {"text": "yok", "color": "green"}}}]
panels = []
# row 1: state
panels += [
    panel("stat", "Mode", [('parking_mode == 1', "{{mode}}", True)], 0, 0, 5, 4,
          desc="SPOT_WATCH / GUARD / PAUSED / UNKNOWN", mappings=[]),
    panel("stat", "Spots", [('parking_spot_state', "{{spot}}", True)], 5, 0, 6, 4, mappings=spot_map,
          desc="Confirmed state per watched spot"),
    panel("stat", "Auto-switch", [('parking_auto_enabled', "", True)], 11, 0, 3, 4, mappings=onoff),
    panel("stat", "GUARD armed", [('parking_guard_armed', "{{spot}}", True)], 14, 0, 4, 4, mappings=onoff),
    panel("stat", "ARAÇ GİTTİ alarm", [('parking_guard_car_gone_alarm or vector(0)', "", True)], 18, 0, 3, 4, mappings=red_on),
    panel("stat", "Unacked alert", [('parking_guard_unacknowledged_alert or vector(0)', "", True)], 21, 0, 3, 4, mappings=red_on),
]
# row 2: alerts + latency
panels += [
    panel("timeseries", "GUARD alerts raised", [('sum by (severity, rule) (increase(parking_guard_alerts_total[$__rate_interval]))', "{{severity}} {{rule}}")], 0, 4, 8, 8,
          desc="Alerts by severity and rule (INFO silent, WARNING, ALERT)"),
    panel("timeseries", "Rule latency (condition met → Telegram sent)",
          [('histogram_quantile(0.5, sum by (le) (rate(parking_guard_rule_latency_seconds_bucket[$__rate_interval])))', "p50"),
           ('histogram_quantile(0.95, sum by (le) (rate(parking_guard_rule_latency_seconds_bucket[$__rate_interval])))', "p95")],
          8, 4, 8, 8, unit="s", desc="Target: <= 3 s end to end",
          thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "red", "value": 3}]}),
    panel("timeseries", "Alerts held back", [('sum by (reason) (increase(parking_guard_alerts_suppressed_total[$__rate_interval]))', "{{reason}}")], 16, 4, 8, 8,
          desc="paused = /pause snooze, cooldown = same rule within 60 s"),
]
# row 3: load control
panels += [
    panel("timeseries", "YOLO duty cycle & limiter step",
          [('parking_guard_yolo_duty_ratio', "duty (cap 0.40)"), ('parking_guard_limiter_step / 3', "limiter step / 3")],
          0, 12, 8, 7, unit="percentunit", minv=0, maxv=1,
          desc="Share of the last 60 s the detector was busy; step 0 normal ... 3 = 1 fps"),
    panel("timeseries", "Substream health", [('parking_guard_frame_age_seconds', "frame age (s)"),
                                              ('increase(parking_guard_rtsp_reconnects_total[$__rate_interval])', "reconnects"),
                                              ('increase(parking_guard_frames_total{outcome="dropped"}[$__rate_interval])', "dropped frames")],
          8, 12, 8, 7, desc="Frame age should stay well under 1 s"),
    panel("timeseries", "Car zone similarity", [('parking_guard_car_zone_similarity', "similarity"),
                                                ('increase(parking_guard_global_change_rejects_total[$__rate_interval])', "headlight/IR rejects")],
          16, 12, 8, 7, minv=0, maxv=1, desc="Parked car ~0.7-1.0; empty spot <= 0.1; below 0.30 = suspicious"),
]
# row 4: cameras + XVR
panels += [
    panel("timeseries", "Camera last good read (age)", [('parking_camera_last_success_age_seconds', "ch{{channel}}")], 0, 19, 8, 7, unit="s"),
    panel("timeseries", "Camera failures in a row", [('parking_camera_consecutive_failures', "ch{{channel}}"),
                                                     ('increase(parking_camera_grab_failures_total[$__rate_interval])', "grab failures ch{{channel}}")], 8, 19, 8, 7),
    panel("timeseries", "SPOT_WATCH cycle p95 / XVR events",
          [('histogram_quantile(0.95, sum by (le) (rate(parking_spotwatch_cycle_seconds_bucket[$__rate_interval])))', "cycle p95 (s)"),
           ('parking_guard_xvr_events_connected', "XVR event stream attached")], 16, 19, 8, 7),
]
# row 5: resources
panels += [
    panel("timeseries", "Container CPU (cores)", [('rate(parking_container_cpu_usage_seconds_total[$__rate_interval])', "bot + ffmpeg"),
                                                  ('rate(parking_cpu_seconds_total[$__rate_interval])', "process (incl. children)")],
          0, 26, 8, 7, desc="Budget: SPOT_WATCH < 0.05, GUARD idle < 0.25 of one core"),
    panel("timeseries", "Container memory", [('parking_container_memory_bytes', "in use"), ('parking_container_memory_peak_bytes', "peak"),
                                             ('parking_container_swap_bytes', "swap")], 8, 26, 8, 7, unit="bytes",
          desc="Limit 700 MB, no swap"),
    panel("timeseries", "CFS throttling & host RAM", [('increase(parking_container_cpu_throttled_periods_total[$__rate_interval])', "throttled periods (must be 0)"),
                                                      ('parking_host_mem_available_bytes', "host MemAvailable"),
                                                      ('parking_host_swap_used_bytes', "host swap used")], 16, 26, 8, 7, unit="short"),
]
# row 6: decisions
panels += [
    panel("timeseries", "Auto-switch outcomes", [('sum by (outcome) (increase(parking_auto_switch_outcomes_total[$__rate_interval]))', "{{outcome}}")], 0, 33, 8, 7),
    panel("timeseries", "Gates that stopped a false 'free'", [('sum by (gate) (increase(parking_spotwatch_gate_overrides_total[$__rate_interval]))', "{{gate}}")], 8, 33, 8, 7,
          desc="stable / tier2 / cross / vlm / ram_defer"),
    panel("timeseries", "SPOT_WATCH messages & Telegram failures", [('sum by (kind) (increase(parking_spotwatch_alerts_total[$__rate_interval]))', "{{kind}}"),
                                                                    ('increase(parking_telegram_send_failures_total[$__rate_interval])', "send failures")], 16, 33, 8, 7),
    panel("table", "Enrolled references", [('parking_car_references', "", True)], 0, 40, 12, 6,
          desc="Appearance vectors per spot / condition (owner vs other)"),
    panel("stat", "Uptime", [('parking_uptime_seconds', "", True)], 12, 40, 4, 6, unit="s"),
    panel("stat", "Pending 'is this your car?'", [('parking_auto_pending_questions or vector(0)', "", True)], 16, 40, 4, 6),
    panel("stat", "Telegram outbox queue", [('parking_telegram_outbox_queued or vector(0)', "", True)], 20, 40, 4, 6),
]
dash = {
    "__inputs": [{"name": "DS_PROMETHEUS", "label": "Prometheus", "description": "", "type": "datasource",
                  "pluginId": "prometheus", "pluginName": "Prometheus"}],
    "__requires": [{"type": "grafana", "id": "grafana", "name": "Grafana", "version": "10.0.0"},
                   {"type": "datasource", "id": "prometheus", "name": "Prometheus", "version": "1.0.0"}],
    "title": "Parking Spotter", "uid": "parking-spotter", "tags": ["parking-spotter"], "timezone": "browser",
    "schemaVersion": 38, "version": 1, "refresh": "10s", "time": {"from": "now-6h", "to": "now"},
    "panels": panels, "templating": {"list": []}, "annotations": {"list": []}, "editable": True,
}
out = Path(__file__).resolve().parent.parent / "docs" / "grafana-parking-spotter.json"
out.write_text(json.dumps(dash, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out, "-", len(panels), "panels")
