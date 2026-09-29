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
