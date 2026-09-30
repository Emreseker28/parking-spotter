# Decisions Log

Lightweight cross-session memory. Append a short entry per phase / decision.

## 2026-06-30 — Phase 0–3 initial build

- **Scope:** Implemented Phases 0–3 in one pass (scaffold + capture, `/cameras`
  viewer, detection + `/parking`, push loop). Hardware-specific values ship as
  placeholders to fill in on the homeserver.
- **Secrets:** gitignored `.env` (+ committed `.env.example`). Loaded via
  `python-dotenv` locally and `env_file` in Docker. No secrets in `config.yaml`
  or code. Repo is safe to publish.
- **Packaging:** runs in Docker (`Dockerfile` + `docker-compose.yml`). ffmpeg is
  installed in the image; `models/` and `config.yaml` are mounted at runtime so
  weights/ROIs are not baked in.
- **Open decisions resolved:**
  - Viewer scope → **live channels (1,2,3) only**, config-driven.
  - Alert policy → **free-only, once per transition**.
  - Camera labels default: 1=Sokak, 2=Giriş, 3=Bahçe (edit in `config.yaml`).

## 2026-06-30 — Models on /mnt/models

- All AI/ML models live on the dedicated **`/mnt/models`** disk (229 GB, ~empty)
  because the root disk is at 97% (3.8 GB free). This matches the existing
  `ollama` and `fastembed-cache` dirs already there.
- `docker-compose.yml` mounts `/mnt/models` → `/app/models` (ro). Non-Docker
  runs use `PARKING_MODEL_PATH=/mnt/models/yolov8n.onnx`.
- `yolov8n.onnx` was exported on-box via a throwaway ultralytics/torch venv
  placed on `/mnt/models` (pip cache + TMPDIR forced there) so the export never
  touched root, then the venv was deleted. `scripts/export_model.py` writes to
  `/mnt/models/yolov8n.onnx` by default (`MODEL_OUT` overrides).

## 2026-06-30 — Detection tuning against real frames

- Tested live: Spot A (ch1) read DOLU at only 23% (ROI sat too low, catching just
  the car's rear); Spot B reported BOŞ though occupied.
- Root cause for Spot B: the large dark top-down car is **invisible to YOLOv8n on
  the full frame** (best ~0.10). Cropping the ROI out makes it a clear car (~0.75).
  Conversely Spot A's clear near car **only detects on the full frame**, never on
  the crop (over-magnification/letterbox distortion).
- **Fix:** per-spot detection now runs BOTH a full-frame pass (shared per channel)
  and an ROI-crop pass, taking the best coverage (`crop_detect`/`crop_pad`/
  `crop_upscale` in config). conf lowered 0.35 → 0.30. Brightening tried, hurt.
- ROIs re-measured from real frames; both spots now read DOLU (100% / 98%).
- Renamed Spot B "Bahçe" → **"ev önü"** (per user).

### Still TODO on the homeserver
- [ ] Pin the XVR static IP and set `NVR_IP` in `.env` (plan §9).
- [x] Capture real ROI polygons (done 2026-06-30; re-measure if cameras move).
- [x] Verify an EMPTY frame reads BOŞ for both spots — now gated by the VLM verifier.
- [x] Night-frame handling — addressed via the VLM second-opinion gate (below) instead
      of per-time-of-day thresholds (kept as a future option).

## 2026-07-01 — Night false positives → VLM gate + accuracy auditor

- **Symptom:** overnight the bot sent **11 false "BOŞ" (empty)** alerts for two spots
  that were occupied by the owner's cars.
- **Root cause:** *not* passers-by mistaken for empty spaces — the opposite. At night
  YOLOv8n **misses the dark / top-down parked cars**, so ROI coverage drops below
  `roi_overlap_threshold` (0.15) → spot scored FREE → `free_only` fires "BOŞ". The
  detection flickered across the night; each occupied→free flip = one alert (11 flaps).
  `debounce_frames: 2` at a 4-min cadence (~8 min) was too short to smooth it.
- **Fix (this change):**
  1. Cadence `interval_minutes` 4 → **30**, `debounce_frames` 2 → **3**.
  2. **VLM gate** — a local ollama vision model (`llava:7b`, config `verify:`) crops the
     spot ROI and judges occupied/empty. Before any "BOŞ" alert, if the VLM sees a car
     the alert is **suppressed**. `fail_open` sends a tagged "(doğrulanamadı)" alert if
     the VLM is unreachable, so real frees are never lost. New `parkingbot/verifier.py`.
  3. **Accuracy auditor** — every poll logs bot-vs-VLM agreement to
     `data/parking-accuracy.jsonl`; `/accuracy` reports a credibility score + how many
     false "BOŞ" the gate blocked.
- **Model location:** the VLM weights **must** live on `/mnt/models` (root disk ~97%).
  Run ollama with `OLLAMA_MODELS=/mnt/models` and
  `OLLAMA_MODELS=/mnt/models ollama pull llava:7b`. Docker reaches host ollama via
  `host.docker.internal` (`extra_hosts` + `OLLAMA_URL`). The accuracy log goes to a
  writable `./data` volume, **not** the read-only `/mnt/models` mount.
- **Future option:** per-time-of-day thresholds / a night-trained YOLO if the gate isn't
  enough (see plan research notes).

## 2026-09-29 — Improvement plan Phase 0: baseline + housekeeping

**Precondition fix (not in the plan):** the bot was found BROKEN — `yolov8n.onnx`
was gone (`/mnt/models` recreated ~Sep 27-28; model absent everywhere on disk), so
every push poll failed "Model not found" while capture kept working. Restored via
the documented throwaway-venv export (staged in /tmp because `/mnt/models` is
root-owned and there is no passwordless sudo; user ran the `sudo cp` + data-dir
commands). Model validated (ort load check) and the 18:51 poll ran clean.
Also noted: root SSD is at 59% (42G free), not the ~97% the plan assumed;
`/mnt/models` is a plain dir on the root LV, not a separate mount.

**Changes (all behind the working agreement: flags default off / rollback = flip flag):**
1. Data dir moved off root: compose volume `/mnt/storage/parking-spotter/data:/app/data`
   (was `./data`). Migration was a verified no-op (no existing JSONL anywhere).
2. Log rotation: new `parkingbot/jsonl.py::append_jsonl` — size-based (10 MB default,
   keep 3 generations). Accuracy log writes go through it (`verify.log_max_mb`/`log_keep`).
   Unit-tested; live path first exercises when audit/feedback logging is enabled.
3. Push state persistence: new `parkingbot/push_state.py` — atomic temp+rename write,
   freshness-checked load (`push.state_path`, `push.state_max_age_hours: 24`). Bot
   restores on startup, saves after every poll.
4. Baseline measured (below).
5. Snapshot API test: **works** — HTTP 200, image/jpeg, ~10 KB, valid baseline JPEG
   on both ch1 and ch3. **Caveat: 352x288** (substream-class), NOT the mainstream
   960x1080/960x576 the ROI polygons are defined for — Phase 4 must rescale polygons
   or assert-and-fallback.

**Baseline (2026-09-29 ~18:20-19:00 UTC, night, IR):**
| Metric | Value |
|---|---|
| Container RSS (idle, after first poll) | 105.2 MiB / 1 GiB |
| Container RSS (peak during poll) | 114.9 MiB |
| Container CPU (during poll) | ~78% (of 2-core limit) |
| Host MemAvailable (idle) | 3.6-3.7 GiB |
| Host MemAvailable (dip during poll) | ~3.61 GiB (≈90 MiB) |
| ffmpeg grab time | ch1 1.25s, ch3 1.31s |
| Push poll wall time | 3.4s (18:51 poll), 3.72s (instrumented) |
| /parking wall time | same code path as push poll + Telegram I/O |

**Live failure captured for later phases:** at 21:23 local (night IR), ev önü read
BOŞ cov=0.00 while a light SUV was clearly visible inside the ROI (frame +
ROI-overlay saved in `.superpowers/sdd/improvement-plan/baseline-frames/`) — zero
detections on full-frame AND crop pass. Spot A read DOLU cov=0.62 (correct). This is
the plan's target failure, recorded with evidence for Phase 2 (stability gate) and
Phase 7 (fine-tune dataset).

**Acceptance:** restart mid-cycle verified — after `docker restart`, the log shows
"restored push state from /app/data/push_state.json" and the state file (on
/mnt/storage) survived; no spurious re-baseline. Log rotation unit-tested (8 tests).
Root SSD usage from the bot's data: ~0 (all writes via the /mnt/storage mount).

**Decision: keep.** Tests: `python -m unittest discover -s tests` → 20/20.

## 2026-09-29 — Improvement plan Phase 1: feedback + labeled crops

**Changes (behind `feedback.enabled`, now true):**
1. ✅ Doğru / ❌ Yanlış inline buttons on every /parking result and every alert.
   Callback data `fb:<spot_idx>:<bot_verdict>:<frame_id>:<response>` (≤26 chars,
   well under the 64-byte limit). A `frame_id -> coverage/context` index feeds the
   feedback record.
2. Padded ROI crops (JPEG q85, ROI only) saved to `data/frames/<ts>_<spot>_<verdict>.jpg`
   for every state-flip candidate, every alert, and every /parking result (the last is
   a superset of the plan's list — the ❌ button on a /parking result needs a frame to
   attach, and the frame only exists during the call). Retention cap
   `feedback.max_frames` (2000), oldest deleted. Crops land on /mnt/storage via the
   Phase 0 mount.
3. `data/feedback.jsonl`: ts, spot, bot verdict, coverage, label, frame id, local hour,
   context (alert/parking/flip). Written through the rotating append_jsonl.
4. `/accuracy [days]` now reports from feedback.jsonl: BOŞ-alert precision (correct
   "BOŞ" / all "BOŞ" alerts), wrong count, per-spot breakdown, day/night split
   (night = 19:00-06:00 local). VLM-agreement lines only when `verify.enabled`.

**Verification:** 40/40 unit tests (crop naming/geometry, retention, callback
parsing, accuracy report with hand-computed values, callback handler with mocked
Telegram update). Live: container runs the new code, feedback enabled, /accuracy
returns the no-data line until the first press. The button-press path is covered by
the mocked-update test; the first real press in Telegram will confirm in production.

**Acceptance:** buttons + recording + crops + /accuracy all verified by tests and
smoke runs. The 1-week feedback collection for judging later phases has started
(first live data: the 19:48+ push polls and any /parking/alert presses).

**Decision: keep.**

## 2026-09-29 — Improvement plan Phase 2: stability ("no motion") gate

**Mechanism:** per spot, keep the last confirmed-occupied ROI crop (grayscale, 160px
wide, CLAHE-normalized, Canny edges, polygon-interior mask). When YOLO says FREE but
the spot is confirmed OCCUPIED and the current crop matches the reference (edge-NCC
>= threshold), override to OCCUPIED and log `stable_override`. Reference refreshes
whenever YOLO reports OCCUPIED with coverage >= 0.5; references older than 12h are
ignored. A car that leaves changes the ROI (low sim -> no override); a missed dark
car leaves it unchanged (high sim -> override). Cost: one crop + one matchTemplate
per pending FREE verdict — no model inference on the steady path.

**Offline replay (threshold selection, 2026-09-29 night, ev önü):**
- Occupied pairs (SUV present, frames ~1h apart, night IR): sim **0.64–0.78**.
- Simulated departures (SUV inpainted out): sim **0.19–0.26**.
- Chosen threshold: **0.45** (margin on both sides; lower than the plan's 0.85
  suggestion because IR night pairs score lower).
- **Caveat:** production compares a daytime reference (refreshed when YOLO detects
  the car by day) against a night current — the day/night gap is not measured here
  and pushes occupied sims down. The 0.45 threshold hedges for it; the live week
  will give the real number. If night misses aren't caught, lower it (or lower
  min_coverage so marginal night detections refresh the reference to a night crop).

**Verification:** 51/51 unit tests (crop shape, self-similarity = 1.0, departure
low, noisy-same-car high, rule cases, staleness). Deployed with
`stability.enabled: true`. The gate only suppresses free alerts (never creates
ones), so a wrong threshold can only delay a real free alert, not spam.

**Acceptance:** offline replay done (threshold + numbers above). The live week
(night false-"BOŞ" per ❌ feedback vs baseline) is now running.

**Decision: keep.**

## 2026-09-30 — Improvement plan Phase 3: tier-2 detector + RAM guard

**Mechanism:** a second, lazily-loaded yolov8s session (`detection_tier2`) used
ONLY on the rare pending-free path (a confirmed free flip that the stability gate
did not catch), on the ROI crop with a lower conf (0.15). The session is unloaded
after each use. Before it runs, a RAM guard reads `/proc/meminfo` MemAvailable and
**defers** (reverts confirmed, keeps the candidate) when below
`min_available_mb` (2500) — never fails open to an unverified alert. A rescue is
logged as `tier2_override` and counted in /accuracy.

**Export:** `scripts/export_model.py --weights yolov8s.pt` (new flag) →
`/mnt/models/yolov8s.onnx` (44.8 MB, staged in /tmp + `sudo cp`, same root-owned
/mnt/models procedure as the yolov8n restore).

**Measured (2026-09-30 night, ev önü):**
- `check_spot_tier2` on the night frame: **occupied=True cov=0.73** — the tier-2
  detector sees the SUV that yolov8n missed (cov 0.00). The rescue mechanism works.
- Wall time 0.72s (grab + tier-2 inference on the crop).
- Container RSS: idle 48 MiB (fresh) / 105 MiB (yolov8n loaded); **peak 251 MiB**
  during the tier-2 run (transient yolov8s session), back to 48 MiB after unload.
  Production peak ≈ 105 + 203 = **~308 MiB** — well under the 1 GiB limit, so the
  compose memory limit is **not** raised.
- Host MemAvailable dipped to ~3357 MiB during the run — above the 2500 guard.

**Bug found & fixed:** `_ensure_tier2` was missing its `return`, so
`check_spot_tier2` crashed with `'NoneType' object has no attribute 'run'`. Caught
by a live measurement, fixed, and covered by a mocked-session regression test
(the original unit tests never exercised a successful load).

**Verification:** 57/57 unit tests. Deployed with `detection_tier2.enabled: true`.
Tier-2 runs only on pending free flips (a few/day); the live week measures the
❌ drop on "BOŞ" alerts.

**Decision: keep.**

## 2026-09-30 — Improvement plan Phase 4: HTTP snapshot capture

**Plan-premise correction:** the Phase 0 curl test showed the snapshot API works but
returns **352x288** (a fixed low-res thumbnail), NOT the mainstream 960x1080/960x576
the ROI polygons are defined for. So the snapshot cannot be used for detection (the
polygons don't apply to a 352x288 frame). The ROI safety check (task 2) is exactly
what catches this.

**Implementation:** `capture.method: auto|snapshot|rtsp`. Snapshot path uses
`httpx.DigestAuth` against `/cgi-bin/snapshot.cgi?channel=N` (keeps the global
lock). `auto` tries the snapshot first, falls back to ffmpeg on error, invalid
JPEG, or size mismatch. `expected_size=(w,h)` (the detection path, from
`spot.frame_size`) triggers the ROI safety check. A per-channel size cache skips
the snapshot after a known mismatch (avoids wasting an HTTP grab every poll).

**Measured (2026-09-30):**
- Viewer grab: snapshot **0.85s** vs ffmpeg substream **3.54s** — ~4x faster.
- Detection grab: ffmpeg-only **3.50s/2ch** (unchanged — the snapshot mismatches the
  polygon coords and falls back). First auto poll 5.47s (snapshot discovery), then
  the cache skips it.
- Without the cache, the snapshot attempt added **+2.24s** to the detection grab
  (5.63s vs 3.39s) — a 66% regression on the primary path. The cache removes it.

**Acceptance:** viewer grab time measurably lower (0.85s vs 3.54s). Detection results
unchanged (uses ffmpeg). Fallback tested via unit tests (size mismatch + snapshot
error). The plan's "20 paired frames, same size" is adapted: the snapshot (352x288)
is compared against the ffmpeg substream (352x288) for the viewer; detection stays
on ffmpeg.

**Decision: keep** (with the documented premise correction + per-channel cache).

## 2026-09-30 — Improvement plan Phase 5: faster, adaptive polling

**Gate note:** the plan says "Only start after Phases 2-3 have a week of good
feedback." The Phases 2-3 live week is still running (started 2026-09-29 ~19:00).
Implemented and deployed now per the user's drive; the interval is a config flip
(back to 30 min) if the feedback shows problems.

**Changes:**
1. Base poll interval 30 → **10 min**.
2. **Adaptive confirm:** while a spot's reading differs from its confirmed state
   (a flip is pending), the job re-polls every **confirm_interval_minutes (2)**
   until the debounce resolves — 3 × 2 min ≈ 6 min instead of 3 × 30 = 90 min.
3. Optional **night hours** (`night_interval_minutes`, default 0 = off) for a longer
   interval overnight.

**Implementation:** the fixed `run_repeating` became a self-rescheduling `run_once`
chain — `push_job` is a wrapper that runs `_push_job_impl` and ALWAYS reschedules
the next run in a `finally` (adaptive interval via `_next_push_interval`, which
reads the per-spot pending state). The wrapper guarantees the loop never dies on
an unexpected error. `_next_push_interval` is unit-tested (base/confirm/night,
midnight wrap, first-reading-not-pending).

**Acceptance:** departure-to-alert latency ~ base + 2×confirm ≈ 10-16 min (was
~90). The 7-day ❌-feedback and host-load check is running.

**Decision: keep.**
