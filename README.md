# Parking Spotter 🚗

A Telegram bot that watches two parking spots through a Dahua XVR recorder and:

- **`/cameras`** — inline keyboard of live cameras; tap one (or *All*) to get a fresh snapshot.
- **`/parking`** — on-demand occupancy check of both spots (free / occupied).
- **Push alerts** — polls periodically and pings you when a spot becomes **free**.

Built around single-frame ffmpeg snapshots (the stream is HEVC/H.265), with
YOLOv8n (ONNX, CPU) for vehicle detection. Designed for a modest homeserver.

## Architecture

```
/cameras   /parking   push loop
    \          |          /
     \         |         /
      Capture Worker (serialized ffmpeg snapshots)
                 |
         Detector (YOLOv8n ONNX + per-spot ROI)   [/parking + push only]
```

All captures are serialized behind one lock — budget XVRs choke on concurrent
RTSP connections.

## Secrets & safety (read before publishing)

- **No secrets are committed.** Real values live in `.env`, which is gitignored.
- Only `.env.example` (a template) is in the repo.
- `config.yaml` and the code contain **no** credentials.
- The bot is restricted to a single Telegram chat (`ALLOWED_CHAT_ID`) — it
  streams views of the house.

| Secret | Where |
|---|---|
| `BOT_TOKEN`, `NVR_IP`, `RTSP_USER`, `RTSP_PASS`, `ALLOWED_CHAT_ID` | `.env` (gitignored) |
| cameras, ROI polygons, tuning | `config.yaml` (committed, non-secret) |
| `yolov8n.onnx` weights | `models/` (gitignored) |

## Setup

### 1. Secrets
```bash
cp .env.example .env
# edit .env with your real BOT_TOKEN, NVR_IP, RTSP creds, ALLOWED_CHAT_ID
```

### 2. Model weights (one-time, any machine with internet)
```bash
pip install ultralytics
python scripts/export_model.py        # -> models/yolov8n.onnx
```
`ultralytics`/`torch` are only needed for this export — **not** at runtime.

### 3. ROI polygons (on the homeserver)
The two spots in `config.yaml` ship with **placeholder** polygons. Capture a real
frame and read off the corner pixel coordinates:
```bash
python scripts/grab_frame.py --channel 1 --subtype 0 --out spotA.jpg
python scripts/grab_frame.py --channel 3 --subtype 0 --out spotB.jpg
```
Open each JPEG, read the polygon corners in pixel coordinates, and paste them
into the matching spot's `polygon:` in `config.yaml`. Polygons are in each
frame's own pixel space (ch1 = 960×1080, ch3 = 960×576).

### 4. Run with Docker (recommended)
```bash
docker compose up -d --build
docker compose logs -f
```
`docker-compose.yml` reads secrets from `.env`, and mounts `models/` and
`config.yaml` read-only so you can edit ROIs / re-export weights without
rebuilding the image.

### Run locally (without Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg          # capture backend
python -m parkingbot
```

## Configuration (`config.yaml`)

- `cameras` — channel → label map shown by `/cameras` (currently live: 1, 2, 3).
- `spots` — the two parking spots with ROI polygons.
- `detection` — model path, thresholds, vehicle classes, ROI overlap threshold.
- `push` — poll interval, debounce frames, alert policy (`free_only`).
- `inference.omp_num_threads` — CPU thread cap.

## Gotchas (from field research)

- **One bot instance only** — duplicate long-pollers cause Telegram 409 conflicts.
- **Pin the XVR's IP** before trusting `NVR_IP` (it has drifted before).
- **URL-encode** special chars in the RTSP password (handled automatically in code).
- **Night detection** of COCO-trained YOLO on grayscale IR frames is weaker —
  validate after dark and tune thresholds (Phase 4).

See `parking-bot-plan.md` for the full research/handoff doc and `docs/DECISIONS.md`
for the running decisions log.
