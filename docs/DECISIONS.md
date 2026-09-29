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
