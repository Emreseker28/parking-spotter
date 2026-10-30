# Parking Spotter — Complete Technical Overview

> An extremely detailed explanation of what the Parking Spotter bot does, how it works,
> and why every design decision was made. Generated from a full investigation of the
> codebase on 2026-09-29.

---

## Table of Contents

1. [What This Bot Does (One-Paragraph Summary)](#1-what-this-bot-does)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [Hardware Environment](#3-hardware-environment)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Source File-by-File Breakdown](#5-source-file-breakdown)
6. [Configuration System](#6-configuration-system)
7. [The Capture Pipeline](#7-the-capture-pipeline)
8. [The Detection Pipeline](#8-the-detection-pipeline)
9. [The VLM Verification System](#9-the-vlm-verification-system)
10. [Telegram Bot Interface](#10-telegram-bot-interface)
11. [The Push Alert Loop](#11-the-push-alert-loop)
12. [Accuracy Auditing & Credibility Score](#12-accuracy-auditing--credibility-score)
13. [Docker Deployment](#13-docker-deployment)
14. [Utility Scripts](#14-utility-scripts)
15. [Security Model](#15-security-model)
16. [Data Flow Diagrams](#16-data-flow-diagrams)
17. [Known Limitations & Gotchas](#17-known-limitations--gotchas)
18. [Configuration Reference](#18-configuration-reference)

---

## 1. What This Bot Does

Parking Spotter is a **Telegram bot** that monitors two parking spots in front of a house
using a Dahua XVR security recorder. It uses **YOLOv8n** (a lightweight object detection
neural network) running on CPU to detect vehicles in camera frames, and a **local vision
language model** (via ollama) as a second-opinion verifier to suppress false "empty" alerts
at night. The bot provides:

- **On-demand camera viewing** — see live snapshots from any camera via Telegram
- **On-demand parking checks** — ask "are the spots free?" and get an instant answer
- **Automatic push alerts** — get notified when a parking spot becomes free
- **A credibility score** — see how often the bot's AI agrees with the VLM verifier

It is designed to run 24/7 on a modest home server (Intel i3-4170, 8 GB RAM, no GPU)
inside a Docker container.

---

## 2. The Problem It Solves

The user has two parking spots in front of their house. They want to know — without
looking out the window — whether a spot is free, so they can park their car there.
The specific challenges are:

1. **The camera recorder uses HEVC/H.265 encoding**, which OpenCV's bundled FFmpeg
   often fails to decode. The solution is to use the `ffmpeg` binary directly via
   subprocess to grab single JPEG frames.

2. **The recorder is a budget Dahua XVR** that cannot handle many simultaneous RTSP
   connections — it has wedged under load before. All captures must be serialized
   behind a single lock.

3. **YOLOv8n is unreliable at night** on grayscale infrared frames. It frequently
   misses dark-colored cars viewed from above, causing false "empty" alerts. The
   fix is a VLM (vision language model) gate that double-checks every "empty" verdict
   before sending an alert.

4. **The home server has limited resources** — a 2-core CPU, 8 GB RAM, no GPU, and
   a nearly-full root disk. All AI models must live on a separate data disk
   (`/mnt/models`), and CPU usage must be capped.

---

## 3. Hardware Environment

### Home Server
| Component | Specification |
|-----------|--------------|
| CPU | Intel i3-4170 (2 cores / 4 threads, Haswell) |
| RAM | 8 GB DDR3 1333 MHz |
| OS Disk | 120 GB SSD (root disk, ~97% full) |
| Data Disk | 2 TB HDD at `/mnt/storage` (media vault) |
| Models Disk | 229 GB dedicated disk at `/mnt/models` |
| Host IP | `192.168.1.50` on `enp2s0` |
| User | `likewhosloss` |
| Container Management | Docker via Portainer |

### Security Recorder ("Ev")
| Property | Value |
|----------|-------|
| Device | Dahua XVR (model DH-XVR1B1..., entry-level hybrid HD-CVI/analog + IP) |
| IP | `192.168.1.5` (must be pinned — it has drifted before) |
| MAC | `6c:1c:71:1d:41:17` |
| RTSP Port | 554 (open, auth required) |
| Auth | `admin` account |
| Codec | HEVC / H.265 |
| Channels | 12 total; only 1–3 confirmed live |

### Camera Channels
| Channel | Resolution | Orientation | Purpose |
|---------|-----------|-------------|---------|
| 1 | 960×1080 | Portrait ("corridor" mode) | Downhill street, front of yellow house. **Parking Spot A.** |
| 2 | 960×1080 | Portrait | Driveway view. Not a target spot. |
| 3 | 960×576 | Landscape | High-angle bay. **Parking Spot B.** |
| 4 | — | — | Offline during research. |
| 5–12 | unknown | — | Not confirmed live. |

### The Two Parking Spots

**Spot A — "Ön (sarı ev)" (Front, yellow house)**
- Channel 1, 960×1080 frame
- A single car-length slot against the near curb in front of the ochre building
- ROI polygon: `[[285, 425], [510, 435], [505, 815], [295, 805]]`
- The parked car renders large at a 3/4 angle — easy detection case
- Mostly sunlit; a moving shadow line grazes one edge

**Spot B — "ev önü" (in front of the house)**
- Channel 3, 960×576 frame
- A defined bay where a dark sedan sits, viewed from a steep top-down angle
- ROI polygon: `[[20, 150], [540, 175], [545, 375], [30, 360]]`
- The car covers ~45% of frame width — trivial for the nano model
- **Critical:** this dark top-down car is invisible to YOLOv8n on the full frame;
  it is only detectable when the ROI is cropped and upscaled

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Telegram Bot (single instance)            │
│                      long-polling, no webhook               │
├──────────────┬──────────────────┬───────────────────────────┤
│  /cameras    │    /parking      │      Push Loop            │
│  (viewer)    │  (on-demand)     │   (periodic every 30min)  │
│              │                  │                           │
│  substream   │   mainstream     │   mainstream              │
│  (subtype=1) │  (subtype=0)     │  (subtype=0)              │
└──────┬───────┴────────┬─────────┴───────────┬───────────────┘
       │                │                     │
       ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Capture Worker (serialized)                     │
│         Single asyncio.Lock — one ffmpeg at a time           │
│         Budget XVRs choke on concurrent RTSP connections     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Detector (YOLOv8n ONNX + per-spot ROI)            │
│     Full-frame pass + ROI-crop pass → best coverage          │
│     Runs in thread executor (CPU-bound blocking work)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│        Verifier (local VLM via ollama) — DISABLED           │
│     GATE: suppress false "empty" alerts                      │
│     AUDIT: log bot-vs-VLM agreement for credibility score    │
│     Currently disabled: hardware cannot sustain it           │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Single-frame snapshots, not video streaming.** The bot never decodes a video
   stream. It grabs one JPEG frame per check via `ffmpeg -frames:v 1`. This keeps
   CPU usage minimal and avoids HEVC decoding issues.

2. **Serialized capture worker.** All RTSP connections go through a single
   `asyncio.Lock`. The viewer, `/parking` command, and push loop can never stack
   connections on the fragile recorder.

3. **ffmpeg binary, not OpenCV VideoCapture.** The stream is HEVC/H.265. OpenCV's
   bundled FFmpeg frequently fails to decode H.265 cleanly. The `ffmpeg` binary
   handles it reliably.

4. **ONNX Runtime, not PyTorch.** The model runs as a `.onnx` file via
   `onnxruntime` on CPU. PyTorch/ultralytics are only needed once to export the
   model — they are not runtime dependencies.

5. **Dual-pass detection.** For each spot, detection runs on BOTH the full frame
   AND a cropped+upscaled ROI. The best coverage wins. This is because some spots
   (clear near cars) only detect on the full frame, while others (dark top-down
   cars) only detect on the crop.

6. **VLM gate (currently disabled).** A local ollama vision model was intended
   to verify every "empty" alert before sending. It was disabled because the
   i3-4170 / 8GB RAM box could not sustain it — the VLM swap-thrashed all cores
   and did not return a verdict in 180 seconds; sustained draw tripped a power
   shutdown.

---

## 5. Source File-by-File Breakdown

### `parkingbot/__init__.py` (3 lines)
Package marker. Defines `__version__ = "0.1.0"`.

### `parkingbot/__main__.py` (36 lines)
**Entry point.** Run via `python -m parkingbot`.

- Configures logging at INFO level, quiets the chatty `httpx` logger
- Loads settings from `config.yaml` + `.env` (fails fast with exit code 2 on error)
- Imports `build_application` from `bot.py` only after config validates (so heavy
  dependencies like `onnxruntime` are not pulled in if config is broken)
- Starts the bot with `app.run_polling(drop_pending_updates=True)` — long-polling,
  not webhooks. `drop_pending_updates=True` prevents replaying a backlog after downtime.

### `parkingbot/config.py` (139 lines)
**Configuration loading.** Two sources: secrets from environment (`.env`),
everything else from `config.yaml`.

**Dataclasses:**
- `Secrets` — `bot_token`, `nvr_ip`, `rtsp_user`, `rtsp_pass`, `allowed_chat_id`
- `Spot` — `name`, `channel`, `polygon` (list of `(x, y)` tuples)
- `Settings` — bundles `Secrets`, `cameras` dict, `spots` list, and the raw YAML dict.
  Provides convenience property accessors (`rtsp`, `capture`, `detection`, `push`,
  `verify`, `inference`) into the raw dict.

**Key behaviors:**
- `load_dotenv(ROOT / ".env")` — loads `.env` if present; no-op in Docker (where
  env vars come from `env_file`). Never overrides already-set process env vars.
- `_require_env()` — raises `ConfigError` if a required env var is missing/empty.
- `_parse_spots()` — validates each spot has a polygon with ≥3 points.
- `ROOT` is computed as the parent of the `parkingbot` package directory, so
  relative paths in config resolve correctly regardless of CWD.

### `parkingbot/capture.py` (112 lines)
**Snapshot capture via ffmpeg subprocess.**

**Class: `CaptureWorker`**
- Holds a single `asyncio.Lock` — all captures are globally serialized.
- `_rtsp_url(channel, subtype)` — builds the RTSP URL with URL-encoded credentials.
  The `quote()` call ensures special characters in the password (`@`, `:`, `#`, `/`)
  don't break the inline `user:pass@host` URL format.
- `_redacted_url()` — same URL but with `***:***` for credentials, used in log messages.
- `grab(channel, subtype)` — the core method:
  1. Acquires the global lock
  2. Creates a temp directory
  3. Builds the ffmpeg command:
     ```
     ffmpeg -y -rtsp_transport tcp -timeout <microseconds> -i <rtsp_url>
            -frames:v 1 -update 1 -q:v 2 <output.jpg>
     ```
  4. Runs it as an async subprocess with `stdout=DEVNULL`, `stderr=PIPE`
  5. Wraps in `asyncio.wait_for(timeout + 10s)` as a wall-clock backstop
  6. On timeout: kills the process and raises `CaptureError`
  7. On non-zero exit or missing output: raises `CaptureError` with last stderr line
  8. Reads and returns the raw JPEG bytes

**Why these ffmpeg flags:**
- `-rtsp_transport tcp` — TCP is more reliable than UDP for the XVR
- `-timeout` — ffmpeg-level socket I/O timeout in microseconds (ffmpeg 6.x renamed
  the old `-stimeout` to `-timeout` for RTSP)
- `-frames:v 1` — grab exactly one frame
- `-update 1` — required by ffmpeg 6.1's image2 muxer for single-image filenames
- `-q:v 2` — high JPEG quality (lower = better)

### `parkingbot/detection.py` (248 lines)
**Vehicle detection and per-spot ROI occupancy logic.**

**Functions:**
- `roi_bbox(polygon, shape, pad)` — computes a padded axis-aligned bounding box of an
  ROI polygon, clamped to frame boundaries. Shared by the detector and the VLM verifier
  so both look at the exact same region.

**Dataclasses:**
- `Detection` — `box` (x1, y1, x2, y2 in original pixels), `score`, `cls`
- `SpotResult` — `spot`, `occupied` (bool), `coverage` (float 0–1), `detections` list,
  `frame` (the source image, shared with the VLM verifier)

**Class: `Detector`**

*Initialization:*
- Resolves model path: `PARKING_MODEL_PATH` env var wins, else `config.yaml`'s
  `model_path`. Relative paths resolve against repo root.
- Sets detection parameters: `input_size=640`, `conf_threshold=0.25`, `nms_iou=0.45`,
  `vehicle_classes=[2, 3, 5, 7]` (COCO: car, motorcycle, bus, truck),
  `roi_overlap_threshold=0.15`
- Crop-detect parameters: `crop_detect=True`, `crop_pad=0.15`, `crop_upscale=2.0`
- Caps CPU threads via `OMP_NUM_THREADS` (default 2)
- ONNX session is lazy-loaded on first use
- ROI mask cache: `(height, width, spot_name) → (mask, area)`

*Model loading (`_ensure_session`):*
- Uses `onnxruntime.InferenceSession` with `CPUExecutionProvider`
- Sets `intra_op_num_threads` and `inter_op_num_threads=1`
- Gets the input tensor name from the session

*Preprocessing (`_letterbox`, `_preprocess`):*
- `_letterbox` — resizes the image keeping aspect ratio, pads to a 640×640 square
  with gray (114) padding. Returns the canvas, scale factor, and padding offsets.
- `_preprocess` — converts BGR→RGB, normalizes to 0–1, transposes to NCHW format,
  adds batch dimension.

*Postprocessing (`_postprocess`):*
- YOLOv8 ONNX output is `(1, 84, 8400)` — transposed to `(8400, 84)`
- Extracts boxes (cx, cy, w, h) and class scores
- Filters by confidence threshold AND vehicle class membership
- Converts from letterboxed coordinates back to original image coordinates
- Runs NMS via `cv2.dnn.NMSBoxes`
- Returns list of `Detection` objects

*Detection methods:*
- `decode(jpeg_bytes)` — decodes JPEG bytes to a BGR numpy array via `cv2.imdecode`
- `detect_array(img)` — runs full detection pipeline on a decoded image
- `detect(jpeg_bytes)` — decode + detect, returns `(detections, (height, width))`

*ROI crop detection (`_crop_detect`):*
- Crops the padded ROI bounding box from the frame
- Upscales by `crop_upscale` (2.0×) using `INTER_CUBIC` interpolation
- Runs detection on the upscaled crop
- Maps detected boxes back to full-frame coordinates

*ROI mask (`_roi_mask`):*
- Creates a binary mask by filling the ROI polygon with `cv2.fillPoly`
- Caches the mask and its area (pixel count) keyed by `(height, width, spot_name)`
- Warns if the mask has zero area (polygon outside frame)

*Coverage calculation (`_coverage`):*
- For each detection, computes how many ROI mask pixels fall inside the detection box
- Returns the maximum coverage ratio (best single detection) as a float 0–1

*Spot checking (`check_spot`):*
- Combines full-frame detections + ROI-crop detections
- Computes coverage against the ROI mask
- A spot is **OCCUPIED** if coverage ≥ `roi_overlap_threshold` (0.15)
- Returns a `SpotResult` with the frame attached (for the VLM verifier)

### `parkingbot/verifier.py` (146 lines)
**Second-opinion verification with a local vision model (ollama).**

**Purpose:** At night, YOLOv8n misses dark/top-down parked cars, causing false "empty"
alerts. The VLM looks at the cropped spot and judges occupied/empty.

**Two roles:**
1. **GATE** — before sending a "BOŞ/empty" alert, ask the VLM. If the VLM sees a car,
   suppress the alert.
2. **AUDIT** — every poll, log bot-vs-VLM agreement for the `/accuracy` credibility score.

**Dataclass: `VerifyResult`** — `ok` (did we get an answer), `occupied` (VLM verdict),
`confidence`, `reason`

**Class: `Verifier`**

*Initialization:*
- `enabled` flag from config (currently `false` — see config.yaml comments)
- `OLLAMA_URL` env var wins over config; defaults to `http://localhost:11434`
- Model name from config (currently `moondream`)
- Timeout, crop padding from config

*The prompt (`_PROMPT`):*
```
You are a parking-spot inspector. This is a cropped image of ONE parking spot at
the front of a house. The image may be daytime, nighttime, or grayscale infrared.
Decide whether a vehicle (car, van, SUV, truck, or motorcycle) is currently parked
in this spot. A vehicle counts as parked even if it is dark, in shadow, low-contrast,
or seen from a steep top-down angle. An EMPTY spot shows bare ground/pavement with no
vehicle. Reply with ONLY a JSON object and nothing else:
{"occupied": true or false, "confidence": a number from 0.0 to 1.0, "reason": "a few words"}
```

*`verify_spot(img, spot)`:*
- Crops the padded ROI (same framing as the detector) → base64 JPEG
- POSTs to `{url}/api/generate` with the prompt, image, `stream=False`, `format=json`,
  `temperature=0`
- Parses the JSON response: extracts `occupied` (bool), `confidence` (float), `reason` (string)
- If JSON parsing fails, attempts to salvage a JSON object embedded in extra prose
- **Never raises** — all exceptions are caught and returned as `VerifyResult(ok=False, ...)`

### `parkingbot/bot.py` (342 lines)
**Telegram bot: commands, viewer, push loop, accuracy reporting.**

**Decorator: `@restricted`**
- Wraps every handler. Checks `update.effective_chat.id` against `ALLOWED_CHAT_ID`.
- Rejects unauthorized users with a warning message and logs the attempt.
- The bot streams views of the house, so only the owner's chat may use it.

**Class: `ParkingBot`**

*Initialization:*
- Creates `CaptureWorker`, `Detector`, `Verifier` instances
- Initializes `_push_state` dict for per-spot debounce/transition tracking
- Resolves the accuracy log path (relative paths resolve against repo root)

**Commands:**

1. **`/start` and `/help`** — Shows a welcome message listing available commands.

2. **`/cameras`** — Displays an inline keyboard with one button per live camera
   (from `config.yaml` `cameras` map) plus an "All" button. On tap:
   - Grabs a substream (subtype=1) frame for each selected channel
   - Sends each as a photo with the channel label as caption
   - On capture failure, sends an error message for that channel

3. **`/parking`** — On-demand occupancy check:
   - Sends a "Checking spots…" placeholder message
   - Calls `_check_all_spots()` which:
     - Determines which channels are needed (from spot configs)
     - For each channel: grabs a mainstream (subtype=0) frame, decodes it, runs
       full-frame detection, then per-spot ROI-crop detection
     - Runs detection in a thread executor (it's CPU-bound and blocking)
     - Returns results in config order
   - Edits the placeholder message with formatted results:
     ```
     🅿️ Parking
     • Ön (sarı ev): 🔴 DOLU  (cov 100%)
     • ev önü: 🔴 DOLU  (cov 98%)
     ```

4. **`/accuracy [days]`** — Reports the bot-vs-VLM credibility score:
   - Reads the accuracy log (`data/parking-accuracy.jsonl`)
   - Filters to the last N days (default 7)
   - Computes per-spot agreement percentage
   - Counts suppressed false "BOŞ" alerts
   - Formats a report:
     ```
     📊 Doğruluk (son 7 gün)
     • Genel uyum: 95%  (19/20)
     • Ön (sarı ev): 100%  (10/10)
     • ev önü: 90%  (9/10)
     • Engellenen yanlış 'BOŞ': 3
     ```

**Push Loop (`push_job`):**
- Runs every `interval_minutes` (currently 30) via `job_queue.run_repeating`
- Calls `_check_all_spots()` (same as `/parking`)
- For each spot result:
  1. **Debounce:** Tracks a `candidate` state and `count`. Only flips `confirmed`
     after `debounce_frames` (3) consecutive agreeing polls. At 30 min/poll, a FREE
     state must persist ~1–1.5 hours before alerting.
  2. **Audit:** If VLM is enabled, calls `verify_spot()` and logs the agreement
     to the accuracy log.
  3. **Transition detection:** Only acts when `confirmed` state changes.
     - First reading sets the baseline (no alert).
     - `free_only` policy: only alerts on occupied→free transitions.
  4. **VLM Gate:** If transitioning to FREE and the VLM is enabled:
     - If VLM says occupied → suppress the alert, log it as "suppressed",
       revert `confirmed` so we re-check next poll.
     - If VLM is unreachable and `fail_open=true` → send alert tagged
       "(doğrulanamadı)" (unverified).
     - If VLM is unreachable and `fail_open=false` → suppress (fail-closed).
  5. **Send alert:** Formats and sends a Telegram message:
     - `🟢 *Ön (sarı ev)* şimdi *BOŞ*` (free)
     - `🔴 *ev önü* şimdi *DOLU*` (occupied, only if policy is `all_changes`)

**Accuracy logging (`_append_log`):**
- Appends one JSON line per record to `data/parking-accuracy.jsonl`
- Each record has a `ts` (Unix timestamp) and `type` ("audit" or "suppressed")
- Audit records include: spot name, bot's occupancy verdict, coverage, VLM verdict,
  VLM confidence, and whether they agreed
- Suppressed records include: spot name, VLM confidence, VLM reason
- Never raises into the push loop — all exceptions are caught and logged

**`build_application(settings)`:**
- Creates the `Application` with the bot token
- Registers all command handlers and the callback query handler
- If push is enabled, schedules `push_job` to repeat at the configured interval
- Returns the configured `Application`

---

## 6. Configuration System

### Two-Layer Configuration

| Layer | File | Contents | Committed? |
|-------|------|----------|------------|
| Secrets | `.env` | Bot token, NVR IP, RTSP creds, chat ID, ollama URL | **No** (gitignored) |
| Non-secrets | `config.yaml` | Cameras, spots, ROIs, detection tuning, push settings, verify settings | **Yes** |

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `BOT_TOKEN` | Yes | Telegram bot token from @BotFather |
| `NVR_IP` | Yes | Dahua XVR IP address (should be static/pinned) |
| `RTSP_USER` | Yes | RTSP username (e.g. `admin`) |
| `RTSP_PASS` | Yes | RTSP password (URL-encoded automatically) |
| `ALLOWED_CHAT_ID` | Yes | Telegram chat ID authorized to use the bot |
| `OLLAMA_URL` | No | ollama endpoint (default: `http://localhost:11434`) |
| `PARKING_MODEL_PATH` | No | Override model path (for non-Docker runs) |

### `config.yaml` Structure

```yaml
cameras:           # Channel → label map for /cameras
  1: "Sokak"
  2: "Giriş"
  3: "Bahçe"

rtsp:              # RTSP connection settings
  port: 554
  path_template: "/cam/realmonitor?channel={channel}&subtype={subtype}"
  transport: tcp
  timeout_seconds: 30

capture:           # Frame grab settings
  viewer_subtype: 1    # substream for /cameras (light)
  detect_subtype: 0    # mainstream for /parking + push (full detail)
  jpeg_quality: 2

detection:         # YOLO + ROI settings
  model_path: "models/yolov8n.onnx"
  input_size: 640
  conf_threshold: 0.25
  nms_iou: 0.45
  vehicle_classes: [2, 3, 5, 7]
  roi_overlap_threshold: 0.15
  crop_detect: true
  crop_pad: 0.15
  crop_upscale: 2.0

spots:             # Parking spot definitions
  - name: "Ön (sarı ev)"
    channel: 1
    frame_size: [960, 1080]
    polygon: [[285, 425], [510, 435], [505, 815], [295, 805]]
  - name: "ev önü"
    channel: 3
    frame_size: [960, 576]
    polygon: [[20, 150], [540, 175], [545, 375], [30, 360]]

push:              # Push alert settings
  enabled: true
  interval_minutes: 30
  debounce_frames: 3
  policy: free_only

verify:            # VLM verification settings
  enabled: false         # DISABLED — hardware cannot sustain it
  backend: ollama
  ollama_url: "http://localhost:11434"
  model: "moondream"
  timeout_seconds: 60
  gate: true
  audit: true
  fail_open: true
  crop_pad: 0.15
  log_path: "data/parking-accuracy.jsonl"

inference:
  omp_num_threads: 2
```

---

## 7. The Capture Pipeline

```
User action (/cameras, /parking, push poll)
        │
        ▼
CaptureWorker.grab(channel, subtype)
        │
        ├── Acquire global asyncio.Lock (serialize all captures)
        │
        ├── Build RTSP URL:
        │   rtsp://<user>:<pass>@<nvr_ip>:554/cam/realmonitor?channel=<ch>&subtype=<sub>
        │   (credentials URL-encoded)
        │
        ├── Build ffmpeg command:
        │   ffmpeg -y -rtsp_transport tcp -timeout <30s_in_us> -i <url>
        │          -frames:v 1 -update 1 -q:v 2 <tmpdir>/frame.jpg
        │
        ├── Run as async subprocess (stdout→DEVNULL, stderr→PIPE)
        │
        ├── Wait with timeout+10s wall-clock guard
        │
        ├── On success: read JPEG bytes, return them
        │
        └── On failure: raise CaptureError with details
```

**Subtypes:**
- `subtype=0` (mainstream): Full resolution, full detail. Used for `/parking` and
  the push loop — anything that feeds into detection.
- `subtype=1` (substream): Lower resolution (e.g. 352×288), lightweight. Used for
  `/cameras` quick viewing — no detection needed, just a visual glance.

---

## 8. The Detection Pipeline

```
JPEG frame bytes
        │
        ▼
Detector.decode(jpeg) → BGR numpy array
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Full-frame pass (shared per channel)               │
│                                                     │
│  _letterbox: resize to 640×640 with aspect ratio    │
│  _preprocess: BGR→RGB, normalize, NCHW              │
│  onnxruntime session.run() → (1, 84, 8400)         │
│  _postprocess: filter by conf + class, NMS,         │
│                map boxes to original coords         │
│  → list[Detection]                                  │
└─────────────────────────────────────────────────────┘
        │
        ▼ (for each spot on this channel)
┌─────────────────────────────────────────────────────┐
│  ROI-crop pass (per spot)                           │
│                                                     │
│  Crop padded ROI bbox from frame                    │
│  Upscale 2.0× with INTER_CUBIC                      │
│  Run same detection pipeline on crop                │
│  Map boxes back to full-frame coordinates           │
│  → list[Detection]                                  │
└─────────────────────────────────────────────────────┘
        │
        ▼
Combine full-frame + crop detections
        │
        ▼
For each detection, compute coverage:
  coverage = (ROI mask pixels inside detection box) / (total ROI mask pixels)
        │
        ▼
best_coverage = max(coverage across all detections)
        │
        ▼
occupied = best_coverage >= roi_overlap_threshold (0.15)
        │
        ▼
SpotResult(spot, occupied, coverage, detections, frame)
```

**Why dual-pass?** Discovered during real-frame testing on 2026-06-30:
- Spot A's clear near car **only detects on the full frame** (over-magnification
  from cropping destroys it)
- Spot B's dark top-down car **only detects on the ROI crop** (invisible on full
  frame at ~0.10 confidence, but ~0.75 on the upscaled crop)
- Running both and taking the best coverage solves both cases

---

## 9. The VLM Verification System

**Status: DISABLED** (as of 2026-07-01)

The VLM verification system was built and tested but disabled because the home
server hardware (i3-4170, 2 cores, 7.5 GiB RAM, no GPU) cannot sustain it:

> moondream swap-thrashed all cores and did not return a verdict in 180s warm;
> sustained draw tripped a power shutdown.

To re-enable: offload ollama to a capable LAN host and set `OLLAMA_URL` to it.

### How It Would Work (When Enabled)

```
Spot result: FREE (YOLO says empty)
        │
        ▼
Verifier.verify_spot(frame, spot)
        │
        ├── Crop padded ROI (same framing as detector)
        ├── Encode as base64 JPEG
        ├── POST to ollama /api/generate:
        │   model: "moondream"
        │   prompt: "You are a parking-spot inspector..."
        │   images: [<base64>]
        │   stream: false
        │   format: json
        │   options: {temperature: 0}
        │
        ├── Parse JSON response:
        │   {"occupied": true/false, "confidence": 0.0-1.0, "reason": "..."}
        │
        └── Return VerifyResult(ok, occupied, confidence, reason)
                │
                ▼
        ┌───────┴───────┐
        │   GATE role   │  Before sending "BOŞ" alert:
        │               │  if VLM says occupied → SUPPRESS
        │               │  if VLM unreachable → fail_open decides
        │               │  if VLM says empty → SEND alert
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │  AUDIT role   │  Every poll (regardless of alert):
        │               │  log bot-vs-VLM agreement to JSONL
        │               │  for /accuracy credibility score
        └───────────────┘
```

### The Prompt

The prompt is carefully crafted to handle the exact failure modes:
- Explicitly mentions nighttime, grayscale infrared, dark cars, shadow, low-contrast,
  and steep top-down angles
- Requests JSON output for reliable parsing
- Temperature 0 for deterministic responses

---

## 10. Telegram Bot Interface

### Commands

| Command | Description | Response |
|---------|-------------|----------|
| `/start` | Welcome message | Lists available commands |
| `/help` | Same as `/start` | Same |
| `/cameras` | Camera viewer | Inline keyboard with camera buttons + "All" |
| `/parking` | Check spots now | "Checking spots…" → formatted results |
| `/accuracy [days]` | Credibility score | Bot-vs-VLM agreement report |

### Inline Keyboard (`/cameras`)

```
┌─────────────┐
│   Sokak     │  ← Channel 1
├─────────────┤
│   Giriş     │  ← Channel 2
├─────────────┤
│   Bahçe     │  ← Channel 3
├─────────────┤
│  📷 All     │  ← All cameras at once
└─────────────┘
```

Tapping a button grabs a substream frame and sends it as a photo.

### Access Control

Every handler is wrapped with `@restricted`. Only the chat ID matching
`ALLOWED_CHAT_ID` can use the bot. Unauthorized attempts are logged and rejected
with "⛔ Not authorized."

### Message Language

The bot's user-facing messages are in **Turkish** (the user's language):
- `BOŞ` = empty/free
- `DOLU` = occupied
- `Doğruluk` = accuracy/credibility
- `doğrulanamadı` = unverified
- `Engellenen yanlış 'BOŞ'` = suppressed false "empty"

---

## 11. The Push Alert Loop

### Timing

- **Poll interval:** Every 30 minutes (configurable via `push.interval_minutes`)
- **Debounce:** 3 consecutive agreeing polls required before flipping state
- **Effective delay:** A FREE state must persist ~1–1.5 hours before alerting
  (30 min × 3 frames = 90 min minimum)

### State Machine (per spot)

```
                    ┌──────────────┐
                    │   unknown    │  (initial state)
                    └──────┬───────┘
                           │ first confirmed reading
                           ▼
                    ┌──────────────┐
              ┌─────│  confirmed   │─────┐
              │     │  (occupied)  │     │
              │     └──────┬───────┘     │
              │            │             │
              │   3× FREE polls          │  3× OCCUPIED polls
              │   (debounced)            │  (debounced)
              │            │             │
              │            ▼             │
              │     ┌──────────────┐     │
              │     │  confirmed   │     │
              └─────│   (free)     │─────┘
                    └──────────────┘
                           │
                           │ occupied→free transition
                           ▼
                    ┌──────────────┐
                    │  VLM GATE    │  (if enabled)
                    │  VLM says    │
                    │  occupied?   │
                    └──────┬───────┘
                      │         │
                   YES│         │NO
                      ▼         ▼
                ┌──────────┐  ┌──────────┐
                │ SUPPRESS │  │  SEND    │
                │ (revert  │  │  ALERT   │
                │  state)  │  │ "BOŞ"    │
                └──────────┘  └──────────┘
```

### Alert Policy

- **`free_only`** (current): Only alert when a spot transitions from occupied to free.
  Stay quiet when a spot becomes occupied.
- **`all_changes`** (alternative): Alert on both directions.

### Alert Format

```
🟢 *Ön (sarı ev)* şimdi *BOŞ*
```
or with unverified tag:
```
⚠️ _(doğrulanamadı)_ 🟢 *Ön (sarı ev)* şimdi *BOŞ*
```

---

## 12. Accuracy Auditing & Credibility Score

### The Accuracy Log

Every push poll appends a JSON line to `data/parking-accuracy.jsonl`:

**Audit record (every poll):**
```json
{
  "ts": 1719878400.123,
  "type": "audit",
  "spot": "Ön (sarı ev)",
  "bot_occupied": true,
  "coverage": 1.0,
  "vlm_ok": true,
  "vlm_occupied": true,
  "vlm_confidence": 0.95,
  "agree": true
}
```

**Suppressed record (when VLM gate blocks a false alert):**
```json
{
  "ts": 1719878400.456,
  "type": "suppressed",
  "spot": "ev önü",
  "vlm_confidence": 0.88,
  "vlm_reason": "dark sedan visible from above"
}
```

### The `/accuracy` Command

Reads the log, filters to the last N days (default 7), and reports:
- Overall agreement percentage (bot vs VLM)
- Per-spot agreement percentage
- Count of suppressed false "BOŞ" alerts

---

## 13. Docker Deployment

### Dockerfile

- Base: `python:3.11-slim`
- Installs: `ffmpeg`, `libgl1`, `libglib2.0-0` (needed by opencv-python-headless)
- Installs Python deps from `requirements.txt`
- Copies `parkingbot/` and `config.yaml` (NOT `.env` or model weights)
- Sets `OMP_NUM_THREADS=2` and `PYTHONUNBUFFERED=1`
- Runs `python -m parkingbot`

### docker-compose.yml

| Setting | Value | Purpose |
|---------|-------|---------|
| `restart` | `unless-stopped` | Auto-restart on crash/reboot |
| `env_file` | `.env` | Inject secrets at runtime |
| `extra_hosts` | `host.docker.internal:host-gateway` | Reach host ollama from container |
| Volume | `/mnt/models:/app/models:ro` | Model weights (read-only) |
| Volume | `./config.yaml:/app/config.yaml:ro` | Live-editable config |
| Volume | `./data:/app/data` | Writable accuracy log |
| CPU limit | 2.0 | Don't starve the host |
| Memory limit | 1 GB | Contain resource usage |
| Log rotation | 10m × 3 files | Prevent log disk fill |

### Resource Constraints

The container is limited to 2 CPUs and 1 GB RAM. The ONNX model runs with
`intra_op_threads=2` and `inter_op_threads=1`. This ensures the bot does not
starve other services on the home server.

---

## 14. Utility Scripts

### `scripts/grab_frame.py`
**Setup helper.** Grabs a single frame from a channel and saves it as a JPEG,
so the user can open it in an image editor and read off the pixel coordinates
of parking-spot corners for the ROI polygons in `config.yaml`.

```bash
python scripts/grab_frame.py --channel 1 --subtype 0 --out spotA.jpg
python scripts/grab_frame.py --channel 3 --subtype 0 --out spotB.jpg
```

### `scripts/export_model.py`
**One-time model export.** Downloads YOLOv8n weights and exports to ONNX format
(opset 12, imgsz 640, simplified). Writes to `/mnt/models/yolov8n.onnx` by default.

```bash
# On the homeserver, using a throwaway venv on the big disk:
export PIP_CACHE_DIR=/mnt/models/.export-cache TMPDIR=/mnt/models/.export-tmp
python3 -m venv /mnt/models/.export-venv
. /mnt/models/.export-venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu ultralytics
python scripts/export_model.py
deactivate && rm -rf /mnt/models/.export-venv /mnt/models/.export-cache /mnt/models/.export-tmp
```

### `scripts/test_verify.py`
**Verification test script.** (Not read in detail, but present in the repo for
testing the VLM verification pipeline.)

---

## 15. Security Model

### Secret Management

- **No secrets are committed to git.** The `.env` file is gitignored.
- Only `.env.example` (a template with placeholder values) is in the repo.
- `config.yaml` and all code contain **no** credentials.
- In Docker, secrets are injected via `env_file` — never baked into the image.
- `load_dotenv()` never overrides already-set process env vars.

### Access Control

- The bot is restricted to a single Telegram chat (`ALLOWED_CHAT_ID`).
- Every handler checks the chat ID before proceeding.
- Unauthorized attempts are logged with the offending chat ID.
- The bot streams views of the house, so this restriction is critical.

### Network Security

- RTSP credentials are URL-encoded to prevent injection via special characters.
- The VLM (ollama) runs locally — camera frames never leave the network.
- The bot uses long-polling (not webhooks), so no inbound firewall rules needed.

### Gitignore Coverage

The `.gitignore` excludes:
- `.env` and `.env.*` (except `.env.example`)
- `models/*.onnx`, `models/*.pt` (model weights)
- `*.jpg`, `*.jpeg`, `*.png` (captured frames)
- `data/` (accuracy logs)
- `__pycache__/`, `.venv/`, `*.log`, `*.jsonl`

---

## 16. Data Flow Diagrams

### `/cameras` Flow

```
User taps camera button
        │
        ▼
CallbackQueryHandler → on_camera_button()
        │
        ├── Parse callback data → channel number
        ├── CaptureWorker.grab(channel, subtype=1)  [substream]
        │       │
        │       └── ffmpeg → RTSP → JPEG bytes
        │
        └── bot.send_photo(chat_id, jpeg, caption=label)
```

### `/parking` Flow

```
User sends /parking
        │
        ▼
cmd_parking()
        │
        ├── Send "Checking spots…" placeholder
        │
        ├── _check_all_spots()
        │       │
        │       ├── For each unique channel:
        │       │       │
        │       │       ├── CaptureWorker.grab(channel, subtype=0)  [mainstream]
        │       │       │       │
        │       │       │       └── ffmpeg → RTSP → JPEG bytes
        │       │       │
        │       │       └── In thread executor:
        │       │               ├── Detector.decode(jpeg) → BGR image
        │       │               ├── Detector.detect_array(img) → full-frame dets
        │       │               └── For each spot on this channel:
        │       │                       └── Detector.check_spot(img, spot, full_dets)
        │       │                               ├── ROI-crop detection
        │       │                               ├── Coverage calculation
        │       │                               └── SpotResult
        │       │
        │       └── Return results in config order
        │
        └── Edit placeholder with formatted results
```

### Push Loop Flow

```
Every 30 minutes (job_queue)
        │
        ▼
push_job()
        │
        ├── _check_all_spots()  [same as /parking]
        │
        └── For each SpotResult:
                │
                ├── Debounce: track candidate state, count consecutive agrees
                │
                ├── Audit (if VLM enabled):
                │       ├── Verifier.verify_spot(frame, spot)
                │       └── Log agreement to JSONL
                │
                ├── If debounced AND state changed:
                │       │
                │       ├── If free_only policy and not →free: skip
                │       │
                │       ├── VLM Gate (if enabled and →free):
                │       │       ├── VLM says occupied → SUPPRESS + revert
                │       │       ├── VLM unreachable + fail_open → send tagged
                │       │       └── VLM unreachable + fail_closed → suppress
                │       │
                │       └── Send alert message
                │
                └── Update confirmed state
```

---

## 17. Known Limitations & Gotchas

### Hardware Limitations

1. **VLM verification is disabled.** The i3-4170 / 8GB RAM box cannot run a local
   vision model sustainably. Night false positives are mitigated only by the 30-min
   poll interval + 3-frame debounce (~1.5h delay), not by the VLM gate.

2. **CPU thread cap.** `OMP_NUM_THREADS=2` limits inference speed. Each YOLOv8n
   inference takes ~0.5–2 seconds on this CPU.

3. **Root disk nearly full.** All models must live on `/mnt/models`. The accuracy
   log goes to `./data` (writable volume), not the read-only models mount.

### Recorder Limitations

4. **One bot instance only.** Duplicate long-pollers cause Telegram 409 conflicts.
   Ensure no old systemd unit is also polling the same token.

5. **IP has drifted before.** The XVR's IP was `192.168.1.250` in an earlier state,
   now `192.168.1.5`. A static IP must be pinned before trusting `NVR_IP`.

6. **RTSP connection cap.** The XVR has wedged under concurrent load. All captures
   are serialized behind one lock, but this means `/cameras` + `/parking` + push
   poll can queue up.

7. **Slow first frame.** The recorder can be slow to produce the first frame. The
   30-second ffmpeg timeout + 10-second wall-clock guard accommodates this.

### Detection Limitations

8. **Night detection is weaker.** COCO-trained YOLO on grayscale IR frames misses
   dark/top-down parked cars. The VLM gate was the intended fix but is disabled.
   Per-time-of-day thresholds remain a future option.

9. **Dual-pass detection is required.** Some spots only detect on the full frame,
   others only on the ROI crop. Running both adds ~1–2 seconds per spot but is
   necessary for reliability.

10. **ROI polygons are camera-specific.** Polygons are in each frame's own pixel
    coordinates (ch1 = 960×1080, ch3 = 960×576). If cameras move, ROIs must be
    re-measured.

### Operational Gotchas

11. **URL-encode RTSP passwords.** Special characters (`@`, `:`, `#`, `/`) in the
    password must be URL-encoded. This is handled automatically in code via
    `urllib.parse.quote()`.

12. **ffmpeg 6.1 requires `-update 1`.** Without it, the image2 muxer rejects
    single-image filenames. This flag is always included.

13. **ffmpeg 6.x renamed `-stimeout` to `-timeout`.** The code uses `-timeout`
    for the RTSP socket I/O timeout.

14. **Accuracy log is append-only.** The JSONL file grows indefinitely. No
    rotation or cleanup is implemented.

15. **No persistent state across restarts.** The push loop's debounce/transition
    state is in-memory only. After a restart, the first reading sets a new baseline
    without alerting.

---

## 18. Configuration Reference

### Detection Tuning

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `conf_threshold` | 0.25 | Minimum detection confidence. Dark cars score low; ROI-overlap gate rejects off-spot vehicles. |
| `nms_iou` | 0.45 | IoU threshold for Non-Maximum Suppression |
| `roi_overlap_threshold` | 0.15 | A spot is OCCUPIED if a vehicle box covers ≥15% of the ROI |
| `crop_detect` | true | Enable dual-pass (full-frame + ROI-crop) detection |
| `crop_pad` | 0.15 | Expand ROI bbox by 15% before cropping |
| `crop_upscale` | 2.0 | Upscale crop 2× before detection |
| `omp_num_threads` | 2 | CPU thread cap for ONNX Runtime |

### Push Tuning

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `interval_minutes` | 30 | Poll cadence (raised from 4 to stop night flicker) |
| `debounce_frames` | 3 | Consecutive agreeing polls before flipping state |
| `policy` | free_only | Only alert on occupied→free transitions |

### Verify Tuning

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| `enabled` | **false** | VLM verification disabled (hardware constraint) |
| `model` | moondream | VLM model name (qwen2.5vl:3b needs ~10GiB, too much) |
| `gate` | true | Suppress FREE alert if VLM says occupied |
| `audit` | true | Log bot-vs-VLM agreement every poll |
| `fail_open` | true | If VLM unreachable, send alert tagged "(doğrulanamadı)" |
| `timeout_seconds` | 60 | HTTP timeout for ollama calls |

---

*This document was generated from a complete investigation of the Parking Spotter
codebase. For the full research/handoff document, see `parking-bot-plan.md`.
For the running decisions log, see `docs/DECISIONS.md`.*

---

## 19. Spot Watch & Guard modes (2026-10-30: Phase 0 probes, Phase 1 SPOT_WATCH, Phase 2 GUARD)

Two new runtime modes are planned: **SPOT_WATCH** (alert when a watched spot frees up, ~60 s cadence,
both ch1 and ch3 spots) and **GUARD** (near-continuous watch for suspicious activity around the
owner's parked car), with later auto-switching via a one-time car "enrollment". Hard rule: no
image or video is ever written to disk by these modes (RAM and pipes only, read-only container).

Phase 0 probe facts that matter for the existing design (details in `docs/PROBE_RESULTS.md`):
substream is CIF 352x288 at <=7 fps (HEVC); XVR event push (`eventManager.cgi`) and playback RTSP
both work; plain YOLOv8n misses the ch3 car at dusk/IR (tier-2 finds it); physical core pairs are
CPUs 0,2 and 1,3. The owner's car is sometimes parked elsewhere on the street, so "no car in either
spot" must mean SPOT_WATCH, not an alarm. Anchor plan and decisions: `docs/DECISIONS.md`.

**Phase 1 (implemented, pending deploy):** `spot_tracker.py` (per-spot debounce/alert/reminder
logic), `modes.py` (SQLite `state/spotter.db`, UNKNOWN/SPOT_WATCH/PAUSED/GUARD controller),
`spotwatch.py` (60 s engine over both spots with the stability -> tier-2 -> VLM gates, camera
failure handling, Telegram outbox, `/status`). Commands: `/watch /guard /auto /pause /resume
/status` (`/guard`, `/auto` are stubs until Phases 2/3). With `spotwatch.enabled: true` the old
10-minute push loop is superseded. The ffmpeg grab now uses a pipe (no temp files) so the
container can run `read_only: true`, pinned with `cpuset: "0,2"`, 700 MB, no swap. Steady-state
cost ~2.8 % of one core. See `docs/DECISIONS.md` (Phase 1 entry) for numbers and gaps.

**Phase 2 (implemented, pending deploy): GUARD.** `/guard` asks which spot the car is in, arms on
the car (median of 5 reads), then watches a perimeter (+40 %/+25 %) on the 352x288 substream.
Idle = decode + a ~1 ms motion gate, no YOLO; YOLO (320 crop) runs only on motion / XVR human
event / live track, rate-capped by a duty-cycle limiter. Rules LOITER / CONTACT / MULTI produce
Turkish alerts with an explicit severity (BİLGİ / UYARI / ALARM); ALARM repeats every 30 s until
`/ack`. Evidence is a RAM ring buffer, JPEG-encoded only when an alert fires. Files:
`guard.py`, `guard_gate.py`, `guard_detector.py`, `guard_tracker.py`, `guard_rules.py`,
`guard_limiter.py`, `guard_geometry.py`, `night.py`; new config section `guard:`; the 320 model is
mounted from `/mnt/storage/parking-spotter/models` at `/app/models-guard`. Person detection on
the real cameras still needs a live walk-through (see the DECISIONS Phase 2 entry).
