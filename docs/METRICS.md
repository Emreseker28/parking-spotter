# Prometheus metrics

The bot serves plain-text Prometheus metrics (numbers and spot labels only: no images, no secrets) on
**`0.0.0.0:9108/metrics`** (`metrics:` in `config.yaml`; the container uses `network_mode: host`).

## One-time wiring (needs you, not the bot)

1. **Firewall.** The Prometheus container reaches host services as `host.docker.internal`; the host
   firewall drops that unless allowed (verified: with the endpoint up, a scrape from inside the
   Prometheus container times out). Same pattern as the OCR service on :9099. Prometheus runs on
   `192.168.16.0/20` (network `canessilver_canessilver`) and `172.18.0.0/16` (`monitoring_default`):

   ```
   sudo ufw allow from 192.168.16.0/20 to any port 9108 proto tcp comment 'prometheus -> parking-spotter metrics'
   sudo ufw allow from 172.16.0.0/12  to any port 9108 proto tcp comment 'prometheus -> parking-spotter metrics'
   ```

2. **Scrape job** in `~/monitoring/prometheus.yml` (this stack is not part of this repo):

   ```yaml
     - job_name: 'parking-spotter'
       static_configs:
         - targets: ['host.docker.internal:9108']
           labels:
             service: parking-spotter
   ```
   then reload (lifecycle is enabled): `curl -X POST localhost:9090/-/reload`.

3. **Dashboard.** Grafana -> Dashboards -> New -> Import -> upload `docs/grafana-parking-spotter.json`,
   pick the Prometheus data source. Regenerate with `python scripts/make_grafana_dashboard.py`.

4. **Optional Uptime Kuma heartbeat.** Create a *Push* monitor, put its URL in `.env` as
   `HEARTBEAT_URL=...` (it contains a token, so not in `config.yaml`). The bot calls it every 60 s while
   its watch loop has completed a cycle in the last 5 intervals; a stalled bot goes quiet and the monitor goes red.

Check it: `curl -s localhost:9108/metrics | grep ^parking_` and, in Prometheus, `up{job="parking-spotter"}`.

## What is exported

| Metric | Meaning |
|---|---|
| `parking_mode{mode}` | 1 for the active mode (SPOT_WATCH / GUARD / PAUSED / UNKNOWN) |
| `parking_auto_enabled`, `parking_pause_remaining_seconds`, `parking_uptime_seconds` | |
| `parking_spot_state{spot}` | 1 occupied, 0 free, -1 unknown; `parking_spot_alerted{spot}` |
| `parking_guard_alerts_total{severity,rule}` | alerts raised (INFO / WARNING / ALERT; LOITER, CONTACT, MULTI, CAR_GONE) |
| `parking_guard_alerts_suppressed_total{reason}` | `paused` or `cooldown` |
| `parking_guard_rule_latency_seconds` (histogram) | rule condition met -> Telegram message sent (target <= 3 s) |
| `parking_guard_frame_age_seconds`, `_frames_total{outcome}`, `_rtsp_reconnects_total` | substream health |
| `parking_guard_yolo_duty_ratio`, `_limiter_step`, `_detect_fps_limit`, `_yolo_runs_total`, `_limiter_changes_total{direction}` | load control |
| `parking_guard_car_zone_similarity`, `_car_gone_alarm`, `_unacknowledged_alert`, `_active`, `_tracks`, `_evidence_frames` | GUARD state |
| `parking_guard_xvr_events_connected`, `_xvr_events_total`, `_global_change_rejects_total` | gates |
| `parking_spotwatch_cycle_seconds` (histogram), `_alerts_total{kind}`, `_gate_overrides_total{gate}`, `_cross_overrides_total`, `_last_cycle_timestamp_seconds` | SPOT_WATCH |
| `parking_camera_last_success_age_seconds{channel}`, `_consecutive_failures`, `_read_latency_seconds`, `_grab_failures_total` | cameras |
| `parking_auto_switch_outcomes_total{outcome}`, `parking_auto_pending_questions`, `parking_car_references{spot,condition,label}` | auto-switching |
| `parking_telegram_send_failures_total`, `parking_telegram_outbox_queued` | Telegram health |
| `parking_cpu_seconds_total`, `process_*` | bot + ffmpeg children CPU, RSS |
| `parking_container_cpu_usage_seconds_total`, `_cpu_throttled_periods_total`, `_memory_bytes`, `_memory_peak_bytes`, `_swap_bytes` | cgroup (container) |
| `parking_host_mem_available_bytes`, `parking_host_swap_used_bytes` | host pressure |

## Handy queries / alert ideas

- Bot gone: `up{job="parking-spotter"} == 0`.
- Watch loop stalled: `time() - parking_spotwatch_last_cycle_timestamp_seconds > 300`.
- Substream stalled while armed: `parking_guard_armed == 1 and parking_guard_frame_age_seconds > 30`.
- Slow alerts: `histogram_quantile(0.95, sum by (le) (rate(parking_guard_rule_latency_seconds_bucket[1h]))) > 3`.
- CPU throttling (must stay 0, we use cpuset not a quota): `increase(parking_container_cpu_throttled_periods_total[1h]) > 0`.
- Memory budget: `parking_container_memory_bytes > 650e6`.
- Camera outage: `parking_camera_consecutive_failures >= 3`.
- Soak test: `deriv(parking_container_memory_bytes[6h])` should hover around 0.
