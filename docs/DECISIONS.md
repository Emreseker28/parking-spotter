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

### Still TODO on the homeserver (cannot be done off-box)
- [ ] Pin the XVR static IP and set `NVR_IP` in `.env` (plan §9).
- [ ] Capture real ROI polygons via `scripts/grab_frame.py`, paste into
      `config.yaml` (current polygons are placeholders).
- [ ] Run `python scripts/export_model.py` to produce `models/yolov8n.onnx`.
- [ ] Night-frame test (plan §9) → Phase 4 tuning (per-time-of-day thresholds).
