# Parking Spotter — Improvement Plan

> Session anchor for the improvement work. Based on `BOT_OVERVIEW.md` (2026-09-29).
> Read this together with `docs/DECISIONS.md`. Do not start a phase until the
> previous phase's acceptance criteria are met and logged in DECISIONS.md.

---

## 1. Goal

Make the bot's **"spot is free" alerts trustworthy and fast** on the existing
hardware, without re-introducing the resource problems that got the VLM disabled.

Today's weaknesses:

| Problem | Where it comes from |
|---|---|
| Night false "free" (dark top-down car missed by YOLOv8n) | COCO model on grayscale IR frames |
| ~90 min alert delay | 30 min poll x 3-frame debounce, chosen to hide night flicker |
| `/accuracy` is empty while VLM is off | Score is defined only as bot-vs-VLM agreement |
| Debounce state lost on restart | State is in-memory only |
| Accuracy log on a ~97%-full root SSD, never rotated | `./data` volume |
| Slow / fragile capture | New RTSP session + HEVC keyframe wait per grab |

## 2. Constraints

- Homeserver: i3-4170 (2c/4t), 8 GB RAM, **usually ~4 GB free**, no GPU.
- Container limit today: 2 CPU / 1 GB. Threads capped (`OMP_NUM_THREADS=2`).
- The XVR wedges under concurrent RTSP load: **all captures stay serialized**.
- Root SSD nearly full: models and any new data go to `/mnt/models` or `/mnt/storage`.
- The RAM is not the tight resource, CPU and power spikes are. Heavy work must be
  **rare, serialized, and unload immediately**.
- Camera frames currently never leave the LAN. Any cloud step (Phase 6B) is an
  explicit, opt-in exception.

## 3. Working agreement

- Phase-gated: scaffold, working baseline, then features.
- Every new behavior sits behind a config flag, **default off** until its phase
  is accepted. Rollback = flip the flag and `docker compose up -d`.
- Deliver **complete replacement files**, not diffs.
- Each phase ends with a DECISIONS.md entry (what changed, measured result, keep/revert).
- Telegram messages stay in Turkish.

## 4. Target alert pipeline (end state)

```
poll -> capture -> YOLOv8n (full + ROI crop) -> says FREE?
                                                    |
                              debounce candidate reaches threshold
                                                    |
                    stability gate (Phase 2): ROI ~unchanged vs last occupied crop?
                          yes -> override to OCCUPIED, log "stable_override"
                          no  ->
                    RAM guard (Phase 3): MemAvailable >= min_available_mb?
                          no  -> defer to next poll (never alert unverified)
                          yes ->
                    tier-2 detector (Phase 3): yolov8s, lower conf, ROI crop
                          sees car -> override to OCCUPIED, log "tier2_override"
                          no  ->
                    optional VLM (Phase 6): local ollama or Groq via AI Gate
                          sees car -> suppress
                          no  -> SEND "BOŞ" alert (+ feedback buttons)
```

Each gate can only **suppress** a free alert, never create one, so a failing gate
can't spam.

---

## Phase 0 — Baseline and housekeeping

**Why first:** everything later needs stable state, safe disk, and a baseline to compare to.

Tasks
1. Move the writable data dir off the root SSD: compose volume
   `/mnt/storage/parking-spotter/data:/app/data`. Migrate the existing JSONL.
2. Add log rotation for `parking-accuracy.jsonl` (rotate monthly or at ~10 MB, keep N).
3. Persist push/debounce state to `data/push_state.json` (atomic write via temp file +
   rename). On startup, restore it if the file is fresh (configurable max age), else
   set a new baseline as today.
4. Measure and record a baseline: container RSS (`docker stats`), wall time of one
   `/parking` and one push poll, ffmpeg grab time per channel, host `MemAvailable`
   at idle and during a poll.
5. Quick test (no code): does the XVR answer
   `curl --digest -u admin:PASS "http://192.168.1.5/cgi-bin/snapshot.cgi?channel=1" -o t.jpg`?
   Record yes/no and the returned image size in DECISIONS.md (feeds Phase 4).

Files: `docker-compose.yml`, `parkingbot/bot.py`, `config.yaml` (`push.state_path`, `verify.log_path`).

Acceptance
- Restart the container mid-cycle: state survives, no spurious re-baseline.
- Log file rotates; root SSD usage from the bot is ~0.
- Baseline numbers written to DECISIONS.md.

---

## Phase 1 — Human feedback and labeled crops

**Why:** gives a real accuracy metric without the VLM, and builds the dataset for
Phase 7. Also lets you measure every later phase.

Tasks
1. After every alert and every `/parking` result, attach inline buttons
   `✅ Doğru` / `❌ Yanlış` per spot. Callback data carries spot + bot verdict + a
   short frame id.
2. Save the padded ROI crop (JPEG q~85, ROI only, not the full frame) for:
   every state-flip candidate, every alert, and every ❌ press.
   Path: `data/frames/<ts>_<spot>_<verdict>.jpg`. Add a retention cap (max files or GB).
3. Append feedback to `data/feedback.jsonl`: ts, spot, bot verdict, coverage, label,
   frame id, hour of day.
4. Rewrite `/accuracy [days]` to report from `feedback.jsonl`: alert precision
   (correct "BOŞ" / all "BOŞ" alerts), wrong count, per-spot breakdown, and split
   day vs night. Keep the VLM-agreement lines only when `verify.enabled`.

Files: `parkingbot/bot.py`, new `parkingbot/feedback.py`, `config.yaml` (`feedback.enabled`, `feedback.max_frames`).

Acceptance
- Button presses recorded; `/accuracy` shows real numbers from feedback.
- Crops land on `/mnt/storage`, capped.
- Collect **at least 1 week of feedback** before judging later phases.

---

## Phase 2 — Stability ("no motion") gate

**Why:** targets the exact failure (car present but YOLO misses it) with essentially
zero CPU/RAM cost. A car that leaves changes the ROI a lot; a missed dark car doesn't.

Design
- New `parkingbot/stability.py`. Per spot keep the last **confirmed-occupied** ROI crop
  (grayscale, downscaled to ~160 px wide, contrast-normalized, ROI-polygon masked).
- Similarity: edge-based normalized cross-correlation (`cv2.Canny` + `cv2.matchTemplate`,
  `TM_CCOEFF_NORMED`) or blurred absolute-difference. Avoid heavy deps (no scikit-image).
- Rule: when YOLO says FREE **and** confirmed state is OCCUPIED **and** similarity to the
  reference >= threshold, override to OCCUPIED and log `stable_override`.
- Refresh the reference whenever YOLO reports OCCUPIED with high coverage, so slow
  lighting drift (dusk, IR switch) doesn't break comparison. Drop references older than
  `max_ref_age_hours`.
- Spot A note: a moving shadow grazes one edge, so mask to the polygon interior and
  tolerate partial edge changes.

Config
```yaml
stability:
  enabled: false
  similarity_threshold: 0.85   # tune from Phase 1 crops
  max_ref_age_hours: 12
```

Files: new `parkingbot/stability.py`, `parkingbot/bot.py`, `config.yaml`.

Acceptance
- Offline replay on saved crops: threshold chosen so **no real departure is suppressed**
  and the known night misses are caught. Record the chosen threshold and the replay
  numbers in DECISIONS.md.
- Live for 1 week: night false-"BOŞ" alerts (per ❌ feedback) drop versus the baseline week.

---

## Phase 3 — Tier-2 detector with RAM guard

**Why:** use spare RAM for accuracy, but only on the rare pending-alert path.

Tasks
1. Export `yolov8s.onnx` (extend `scripts/export_model.py` with a `--weights` flag) to
   `/mnt/models/`. Consider `yolov8m` only if `s` isn't enough and timing allows.
2. `Detector` gets a second lazily-loaded session (`tier2_model_path`), used only for
   the ROI crop with a lower confidence threshold. **Unload the session after use** so
   it doesn't stay resident.
3. RAM guard: read `/proc/meminfo` `MemAvailable`; if below `min_available_mb`,
   **defer** (keep candidate state, retry next poll). Never fail-open to an unverified
   alert.
4. Log `tier2_override` when it rescues a false free.
5. Re-measure container RSS; raise the compose memory limit (e.g. 1.5 GB) only if
   the measurement requires it.

Config
```yaml
detection_tier2:
  enabled: false
  model_path: "models/yolov8s.onnx"
  conf_threshold: 0.15
  min_available_mb: 2500
```

Files: `parkingbot/detection.py`, `parkingbot/bot.py`, `scripts/export_model.py`, `config.yaml`, `docker-compose.yml` (if memory limit changes).

Acceptance
- Tier-2 runs only on pending free flips (check logs: a few calls/day).
- Peak container memory and CPU during a tier-2 run recorded; host never dips below the guard.
- Measurable drop in ❌ on "BOŞ" alerts over a week, or a documented decision to remove it.

---

## Phase 4 — Faster capture via HTTP snapshot

**Why:** no RTSP session, no HEVC first-keyframe wait, less load on the XVR.
Only proceed if Phase 0's curl test succeeded.

Tasks
1. `capture.py`: add `capture.method: auto | snapshot | rtsp`. Snapshot path uses
   `httpx.DigestAuth` against `/cgi-bin/snapshot.cgi?channel=N`. Keep the global lock.
   `auto` = try snapshot, fall back to ffmpeg on error or an invalid JPEG.
2. **ROI safety check:** ROI polygons are in exact pixel coordinates (ch1 960x1080,
   ch3 960x576). Assert the returned frame matches `spot.frame_size` (or rescale the
   polygon); if not, fall back to ffmpeg and log a warning.
3. Compare 20 paired frames (snapshot vs ffmpeg mainstream): same size, and detection
   results match.

Files: `parkingbot/capture.py`, `config.yaml`.

Acceptance
- Grab time per channel measurably lower than the Phase 0 baseline.
- No change in detection results on paired frames; fallback path tested by blocking HTTP.

---

## Phase 5 — Faster, adaptive polling

**Why:** with real verification in place, the slow poll is no longer needed to hide flicker.
Only start after Phases 2-3 have a week of good feedback.

Tasks
1. Baseline poll interval down (start at 10 min).
2. **Adaptive confirm:** when a reading differs from the confirmed state, switch that
   spot to a short re-poll (e.g. every 2 min) until the debounce resolves either way.
3. Optional `night_hours` with a longer interval if the extra CPU/noise matters.

Config
```yaml
push:
  interval_minutes: 10
  confirm_interval_minutes: 2
  debounce_frames: 3
```

Files: `parkingbot/bot.py`, `config.yaml`.

Acceptance
- Departure-to-alert latency roughly <= 10 min (was ~90).
- Over 7 days: no increase in ❌ "BOŞ" feedback; host load and XVR stability unchanged
  (no capture-queue backlog in logs).

---

## Phase 6 — Optional VLM gate (only if Phases 2-3 leave residual errors)

Decision required (see section 6). Verifier interface stays; add `verify.backend: ollama | aigate`.

**6A. Local retest**
- Pass `keep_alive: 0`, `num_ctx: ~1024`, `num_thread: 2`; set `OLLAMA_NUM_PARALLEL=1`
  and `OLLAMA_MAX_LOADED_MODELS=1`; make sure no other model is loaded in that Ollama.
- Only called on a pending free flip (a few times/day), never every poll.
- Test with `free -h` and `docker stats` open; check that the earlier PSU-related
  shutdown does not recur. Try a smaller VLM (e.g. SmolVLM-500M via llama.cpp) before
  a 3B model.

**6B. Groq via AI Gate**
- Send only the ROI crop, caller ID `parking-spotter`, metadata-only logging (per AI Gate design).
- Blocked on AI Gate existing. Check which vision models Groq currently offers.
- Breaks the "frames never leave the LAN" guarantee for these crops: explicit opt-in.

Acceptance
- Measured latency and peak memory (6A) or cost/latency (6B) recorded.
- Residual false-free rate versus Phase 3 baseline; keep only if it improves it.

---

## Phase 7 — Fine-tune the detector (long-term fix)

Tasks
1. Use the Phase 1 crops: target **200+ labeled images**, including 50+ dark/top-down night cases.
2. Label (CVAT / Label Studio / Roboflow), single class `car`.
3. Fine-tune YOLOv8n on a free GPU (Colab) or a work machine; export ONNX (opset 12, imgsz 640).
4. Validate on held-out frames against the stock model. Ship only if it wins at night
   without regressing daytime.
5. Drop into `/mnt/models`, change `model_path`. Keep the stock model as the rollback.

Acceptance: fewer night misses with the crop pass alone, allowing gates to be relaxed.

---

## 5. Resource budget (verify in Phase 0 — these are rough expectations)

| Item | Expected | Rule |
|---|---|---|
| Bot container (steady) | a few hundred MB | Stay under the compose limit |
| ffmpeg per grab | ~100 MB, seconds | Serialized by the capture lock |
| Tier-2 YOLO session | a few hundred MB, transient | Unload after use |
| Local VLM (6A only) | ~1.5-3 GB, transient | `keep_alive: 0`, one at a time |
| Host headroom | keep >= ~1.5 GB free | RAM guard defers heavy steps |

Do not overlap heavy steps with the OCR pipeline or other big jobs if avoidable.

## 6. Open decisions

1. **VLM route:** none / local retest (6A) / Groq via AI Gate (6B)? Decide after Phase 3 data.
2. **Cadence targets:** is ~10 min acceptable for alert latency, or is faster wanted?
3. **Retention:** how many labeled crops / how many GB to keep on `/mnt/storage`?
4. **Snapshot API:** depends on the Phase 0 curl result.

## 7. File touch map

| File | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| `parkingbot/bot.py` | x | x | x | x | | x | x |
| `parkingbot/detection.py` | | | | x | | | |
| `parkingbot/capture.py` | | | | | x | | |
| `parkingbot/verifier.py` | | | | | | | x |
| `parkingbot/feedback.py` (new) | | x | | | | | |
| `parkingbot/stability.py` (new) | | | x | | | | |
| `scripts/export_model.py` | | | | x | | | |
| `config.yaml` | x | x | x | x | x | x | x |
| `docker-compose.yml` | x | | | maybe | | | |

## 8. Progress tracker

- [ ] Phase 0 — Baseline and housekeeping
- [ ] Phase 1 — Feedback and labeled crops
- [ ] Phase 2 — Stability gate
- [ ] Phase 3 — Tier-2 detector + RAM guard
- [ ] Phase 4 — HTTP snapshot capture
- [ ] Phase 5 — Adaptive polling
- [ ] Phase 6 — Optional VLM gate
- [ ] Phase 7 — Fine-tuned detector
