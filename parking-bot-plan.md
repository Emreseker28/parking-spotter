# Parking Spot Detection Bot — Project Plan

> Handoff document for Claude Code (claude-cli). This captures the completed research
> phase: confirmed facts, verified commands, architecture, and a phased build path.
> Treat the "Confirmed" sections as ground truth — they were validated on real hardware,
> not assumed. Build iteratively, one phase per session.

---

## 1. Goal

A Telegram bot running on the homeserver that:
1. **`/cameras`** — interactive viewer: shows an inline keyboard of cameras; user taps which channel(s) they want; bot grabs fresh frames and sends the images.
2. **`/parking`** — on-demand occupancy check of two defined parking spots; replies free/occupied per spot.
3. **Push alerts** — polls the two spots periodically and notifies the user when a spot transitions to *free*.

Hardware is modest, so the whole design is built around **single-frame snapshots**, not continuous video decoding.

---

## 2. Environment (Confirmed)

**Homeserver**
- CPU: Intel i3-4170 (2c/4t, Haswell)
- RAM: 8 GB DDR3 1333 MHz
- SSD: 120 GB (OS) · HDD: 2 TB at `/mnt/storage` (media vault)
- Host IP: `192.168.1.50` on `enp2s0`
- User: `likewhosloss` · GitHub: `Emreseker28`
- Stack convention: all services run as Docker containers (Portainer-managed)

**Recorder ("Ev") — Confirmed**
- Device: **Dahua XVR**, model `DH-XVR1B1...` (entry-level hybrid HD-CVI/analog + IP recorder)
- Current IP: `192.168.1.5` · MAC: `6c:1c:71:1d:41:17`
- Device name in router DHCP list and in DMSS app: **"Ev"**
- RTSP: **open on port 554**, auth required (working account: `admin`)
- Codec: **HEVC / H.265**

> ⚠️ The IP has already wandered once (was `192.168.1.250` earlier in a different/unstable
> state). A static IP MUST be pinned before hardcoding URLs — see §9 pre-build checklist.

---

## 3. Verified Capture Method

A single still frame is grabbed via the **`ffmpeg` binary** (subprocess), not OpenCV.
This was tested and works. Pattern:

```bash
ffmpeg -y -rtsp_transport tcp \
  -i "rtsp://${RTSP_USER}:${RTSP_PASS}@${NVR_IP}:554/cam/realmonitor?channel=${CH}&subtype=${SUB}" \
  -frames:v 1 -update 1 -q:v 2 out.jpg
```

- `subtype=0` → **mainstream** (full detail, use for detection)
- `subtype=1` → **substream** (CIF-ish, light, use for the quick viewer)
- `-update 1` is required (ffmpeg 6.1 image2 muxer rejects single-image filenames without it)
- `-q:v 2` = high JPEG quality
- Credentials come from env, never hardcoded (see §8)

**Why ffmpeg and not `cv2.VideoCapture`:** the stream is **HEVC**, and OpenCV's bundled
FFmpeg backend frequently fails to decode H.265 cleanly (returns empty frames). Grabbing
the JPEG with the ffmpeg binary and handing it to the model sidesteps this entirely.

---

## 4. Channel Inventory (Confirmed)

| Channel | Mainstream res | Orientation | Notes |
|--------|----------------|-------------|-------|
| 1 | 960×1080 | Portrait ("corridor" mode) | Downhill street, front of yellow house. **Parking spot A here.** |
| 2 | 960×1080 | Portrait | Driveway view (used as the RTSP test channel). Not a target spot. |
| 3 | 960×576 | Landscape | High-angle bay; large sedan visible. **Parking spot B here.** |
| 4 | — | — | **Offline** (Çevrimdışı) during research. Ignore unless it returns. |
| 5–12 | unknown | — | Recorder reports 12 channels; only 1–3 confirmed live. |

Substream (`subtype=1`) was 352×288 on ch2; assume similar for others but verify per-channel.

---

## 5. The Two Parking Spots (ROI Targets)

Both reduce to a **single fixed-slot occupancy check** (is there a car inside this polygon?),
*not* a gap-measurement problem. One ROI polygon per spot.

**Spot A — Channel 1 (front of yellow house)**
- Location: a single car-length slot against the near curb in front of the ochre building,
  toward the gap between houses. (User marked this region on a frame during research.)
- **Confirmed with a parked car (DMSS live view):** a vehicle in this spot renders **large**
  and sits in the **lower-center of the full 960×1080 ffmpeg frame**, at a clean 3/4 angle
  showing full side + roof. Anchor the ROI polygon there (NOT mid/upper field — the empty
  marked frame looked more distant than the spot actually is, an artifact of DMSS cropping
  the tall frame to 16:9).
- Render size: large → **high detection confidence with YOLOv8n nano** (revised up from
  "moderate"; the parked-car view confirms it's an easy case, like Spot B).
- Lighting: mostly sunlit; a moving shadow line from the building grazes one edge — not a problem (model keys on the car, not background).

**Spot B — Channel 3 (defined bay)**
- Location: the bay where the dark sedan sits in the test frame (left-center of frame).
- Render size: very large (car ≈45% of frame width) → trivial for nano model.
- The easiest of the two; high confidence.

> ROI polygons must be defined in **each frame's own pixel coordinates** (ch1 = 960×1080,
> ch3 = 960×576) — they are NOT interchangeable. Exact polygon coordinates are not yet
> captured; do this as a setup step (view a frame, read off / click polygon vertices, store in config).

---

## 6. Detection Design

- **Model:** YOLOv8n (nano), exported to **ONNX**, run via **onnxruntime** (matches the
  existing fastembed/ONNX setup on this server).
- **Classes of interest:** car, truck, bus, motorcycle (COCO vehicle classes).
- **Occupancy logic:** for each spot, run detection on the frame; if any vehicle bbox
  overlaps the spot's ROI polygon beyond a threshold (e.g. IoU or % of ROI covered) →
  **occupied**, else **free**.
- **Optimization available if needed:** crop the frame to the ROI rectangle, upscale, and
  run inference on the crop. Makes mid-distance cars "bigger" to the model — useful for
  Spot A and especially at night. Keep in reserve; may not be needed for daytime.
- **CPU budget:** one inference per spot, a few times per minute at most. ~0.5–2 s per
  YOLOv8n inference on this CPU. Cap threads (e.g. `OMP_NUM_THREADS=2`). The 65 W-adapter
  power concern does **not** bite for single-shot grabs — it only matters under sustained load.

---

## 7. Architecture

```
                 ┌─────────────────────────┐
                 │   Telegram Bot (single   │
                 │   instance, long-poll)   │
                 └─────────┬───────────────┘
        ┌──────────────────┼──────────────────────┐
   /cameras            /parking              push loop
  (viewer)          (on-demand)         (periodic poll)
        │                  │                      │
        └──────────► Capture Worker ◄─────────────┘
                  (serialized, 1 at a time,
                   ffmpeg snapshot grabs)
                            │
                            ▼
                  Detection (YOLOv8n + ROI)
                  [only for /parking + push]
```

**Capture Worker — important.** Budget XVRs cap simultaneous RTSP connections, and this
box has wedged once already under load. Serialize all captures behind a single lock/queue
so the viewer, /parking, and the push loop never stack 5+ connections at once. Generous
ffmpeg timeouts (the recorder can be slow to first frame).

---

## 8. Feature Specs

### 8.1 `/cameras` — interactive viewer (build first)
- Replies with an inline keyboard: one button per **live** channel + an **"All"** button.
- On tap (callback), grab the chosen channel(s) and send the JPEG(s).
- Use **substream** (`subtype=1`) for speed — this is a quick glance, not detection.
- Channel list should be **config-driven** (e.g. `CAMERAS = {1: "Sokak", 2: "...", 3: "Bahçe"}`),
  so adding/removing channels is a config edit, not a code change.
- **Default decision:** expose only confirmed-live channels (1, 2, 3), not all 12 dead buttons.
  *(Confirm with user if they want placeholders for future cameras.)*

### 8.2 `/parking` — on-demand check
- Grab ch1 + ch3 **mainstream**, run YOLO + each spot's ROI.
- Reply, e.g.: `🅿️ Ön (sarı ev): BOŞ  ·  Channel 3: DOLU`
- Pure request/response, no persistence.

### 8.3 Push alerts
- Poll ch1 + ch3 every **N minutes** (config; start ~3–5 min).
- **State tracking:** store last known status per spot; alert only on a **transition**.
- **Debounce:** require **2–3 consecutive agreeing frames** before flipping state, to absorb
  a passing car, a shadow, or brief IR glare at night.
- **Default alert policy:** notify **once when a spot transitions to FREE**, then stay quiet
  until it changes again. *(Confirm: free-only vs all changes; once-per-transition vs repeating.)*

---

## 9. Pre-Build Checklist (do these first)

- [ ] **Pin the recorder's IP.** Set a static IP on the XVR itself (preferred) or a DHCP
      reservation for MAC `6c:1c:71:1d:41:17`. Put it **outside the router's DHCP pool** to
      avoid collisions, and set **gateway + DNS** so the recorder keeps internet (needed for
      DMSS P2P and NTP). Fixing NTP also corrects the timestamp burned into frames.
- [ ] **Capture ROI coordinates** for Spot A (ch1, 960×1080) and Spot B (ch3, 960×576);
      store polygons in config.
- [ ] **Night-frame test.** Grab ch1 + ch3 *after dark* (same ffmpeg command) and verify a
      parked car is still clearly detectable under IR/grayscale. Cameras have night vision,
      but COCO-trained YOLO is weaker on grayscale night frames. If confidence is shaky:
      lower the night threshold, increase debounce frames, or use the crop-and-upscale trick.
      Treat this as a validation gate before trusting push alerts at night.

---

## 10. Build Phases (one per CLI session; `/clear` between)

- **Phase 0 — Scaffold & capture.** Project skeleton, `.env`, Dockerfile, confirm the
  ffmpeg snapshot works from inside the container against the (now static) IP.
- **Phase 1 — `/cameras` viewer.** Inline keyboard + callbacks + substream grab-and-send.
  This proves the whole capture→Telegram path with no detection. Ship this first.
- **Phase 2 — Detection + `/parking`.** Add YOLOv8n ONNX, ROI polygons, occupancy logic,
  on-demand command.
- **Phase 3 — Push loop.** Periodic poll, per-spot state, debounce, transition alerts.
- **Phase 4 — Night tuning.** Apply night-frame test findings; per-time-of-day thresholds.

Append a short entry to `docs/DECISIONS.md` after each phase (lightweight cross-session memory).

---

## 11. Stack & Dependencies

- **Language:** Python 3.11+
- **Telegram:** `python-telegram-bot` (v21+), single instance, long-polling
- **Inference:** `onnxruntime` + YOLOv8n ONNX weights
- **Image/ROI:** `numpy`, `Pillow` (and/or `opencv-python-headless` for polygon math only —
  NOT for RTSP decoding)
- **Capture:** `ffmpeg` binary via `subprocess`
- **Packaging:** Docker container, managed in Portainer (match the rest of the stack)

---

## 12. Conventions & Gotchas

- **Credentials in `.env`** (gitignored), never in code or committed files. Expected keys:
  `BOT_TOKEN`, `NVR_IP`, `RTSP_USER`, `RTSP_PASS`, `ALLOWED_CHAT_ID`.
- **URL-encode special chars** in the RTSP password if any (`@`→`%40`, `:`→`%3A`, `#`→`%23`,
  `/`→`%2F`) — otherwise the inline `user:pass@host` URL silently looks like an auth failure.
- **One bot instance only.** Duplicate pollers cause Telegram 409 conflicts. If containerizing,
  ensure no old systemd unit is also polling the same token.
- **Serialize captures** behind one worker/lock (see §7) — the XVR is fragile under concurrent load.
- **Generous timeouts** on ffmpeg subprocess calls; the recorder can be slow to first frame.
- **Restrict the bot** to the owner's chat ID (`ALLOWED_CHAT_ID`) — it streams views of the house.

---

## 13. Open Decisions (confirm before/early in build)

1. **Viewer scope:** live channels (1,2,3) only [current default] vs. include placeholders for 12.
2. **Alert policy:** free-only [default] vs. all changes; once-per-transition [default] vs. repeating.
3. **Night coverage:** required? (Pending the night-frame test in §9.)

---

*Research phase complete. Source confirmed, RTSP authenticating, frames decoding, both spots
located and classified, hardware feasibility validated. Remaining unknowns are the three open
decisions above and the night-frame test — all answerable during Phase 0–1.*
