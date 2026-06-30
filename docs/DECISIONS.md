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
- [ ] Verify an EMPTY frame reads BOŞ for both spots (no false positives).
- [ ] Night-frame test (plan §9) → Phase 4 tuning (per-time-of-day thresholds).
