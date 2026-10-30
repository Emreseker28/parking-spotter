# Phase 0 probe results — Spot Watch & Guard

Run 2026-10-30 (~18:10–18:50 local, camera clock UTC+3; host clock is UTC). Production bot
untouched. Probe scripts: `scripts/probes/`. Credentials are read from `.env` and never printed.
Probe frames were kept in a scratch dir only and deleted afterwards.

**Caveat on all timings:** the box was not idle (load 1.4–2.1, qbittorrent, cadvisor, swap full
and actively swapping). Numbers are realistic for "a normal evening", not best case.

## Summary (what changes the plan)

1. **Substream is CIF-only: 352x288, H.265, max 7 fps, 80 kbps CBR.** The plan's 640x360 @ 5 fps
   cannot be configured on this XVR (`ResolutionTypes=CIF`, `FPSMax=7`). Ch1 (portrait 960x1080)
   is squashed into landscape 352x288.
2. **Substream decode is cheap (~5 % of one logical CPU); main-stream decode is not (~31 %).**
   GUARD should run its persistent reader on the substream and fetch main-stream frames only on
   demand (verification crop, plate, evidence).
3. **XVR event push works** (SmartMotionHuman / SmartMotionVehicle / VideoMotion). Gate candidate.
4. **XVR records 24/7 and playback RTSP works** — usable to tune with real past events.
5. **Plain YOLOv8n misses the ch3 car at dusk/IR** (cov 0.00 on a clearly visible car). The existing
   tier-2 (yolov8s on crop, conf 0.15) finds it (cov 0.82). SPOT_WATCH must reuse the existing
   stability + tier-2 (+ VLM) gates or it will send false "free" alerts at night.
6. **OpenVINO: no gain. int8: ~20 % faster but lost detections. Stay fp32 + plain onnxruntime.**
7. **Box is under memory pressure** (swap 4/4 GiB used, ~3.2 GiB available). The 700 MB / no-swap
   budget is only meaningful if the other services stop swapping; worth watching.

## 1. Streams (`probe_streams.py`, XVR HTTP `configManager`)

| Channel | Stream | Codec | Resolution | FPS | Bitrate |
|---|---|---|---|---|---|
| 1 | main | HEVC Main | 960x1080 | 25 | 512 kbps |
| 1 | sub | HEVC Main | 352x288 | 7 | 80 kbps CBR |
| 3 | main | HEVC Main | 960x576 | 25 | 512 kbps |
| 3 | sub | HEVC Main | 352x288 | 7 | 80 kbps CBR |

Firmware `4.001.0000000.14` (2021-03-06). Substream options: H.264/H.265, CIF only, 7 fps max,
bitrate 16–320 kbps, GOP 1–7. ffprobe reports `r_frame_rate=100/1` (timebase, not real fps).
At 352x288 the plate is unreadable; it is legible on the ch1 main stream.

## 2. Decode cost (`decode_cost.sh`, 30 s, pinned to CPUs 0,2, raw BGR to /dev/null)

| Stream | CPU time / 30 s | ≈ one logical CPU | ffmpeg RSS |
|---|---|---|---|
| ch3 sub (352x288 @ 7) | 1.46 s | **~5 %** | 88 MB |
| ch3 main (960x576 @ 25) | 9.24 s | **~31 %** | 112 MB |

Substream fits the "GUARD idle < 25 %" budget with room; main stream does not.
`Could not find ref with POC` at stream start is a harmless HEVC start-up warning.

## 3. YOLOv8n latency (onnxruntime 1.19.2, fp32, taskset to one physical core)

Input = one real ch3 frame letterboxed to model size. p50 / p95 ms, 30 runs.

| Input | 1 thread (cpu 0) | 2 threads (cpus 0,2) | RSS |
|---|---|---|---|
| 640 | 174 / 266 | 158 / 202 | 150 MB |
| 416 | 72 / 97 | 61 / 66 | 115 MB |
| 320 | 41 / 73 | 36 / 41 | 109 MB |

- 2nd hyperthread gains only ~10–15 % but tightens p95. Use **2 threads** if the core is dedicated,
  otherwise 1.
- **Crop benefit needs a smaller-input model.** The shipped models are static 640x640, so a 320
  crop is still letterboxed to 640 (no saving). 320/416 models must be exported (done here with a
  throwaway ultralytics venv, imgsz 320/416, opset 12). A 320 crop costs ~4–5x less than 640.
- **OpenVINO (native runtime, CPU):** 162 / 72 / 39 ms (1 thread) — same as ORT, +60 MB RSS. Reject.
  (The `onnxruntime-openvino` build was not installed; native OpenVINO is the same engine.)
- **int8 (static QDQ, 4-frame calibration):** 416 → 58 ms, 320 → 33 ms (~20 % faster), but 416
  lost all 4 car detections that fp32 found. Calibration set was tiny, so indicative only.
  Reject unless a real calibration set shows no recall loss.
- Raw class counts on this single frame are not an accuracy test (no person in frame).

## 4. Night / IR detection at the spots (dusk, 18:13 local, IR already on)

| Frame | Existing detector (n, full + crop) | Tier-2 (yolov8s crop, conf 0.15) |
|---|---|---|
| ch1 main, owner-spot car | occupied, cov 0.79 | occupied, cov 0.78 (0.43 s) |
| ch3 main, large white car | **free, cov 0.00 (miss)** | occupied, cov 0.82 (0.76 s) |
| ch1 sub 352x288 (scaled ROI) | occupied, cov 0.76 | free, 0.00 |
| ch3 sub 352x288 (scaled ROI) | free, 0.00 | occupied, 0.82 |

Single frames, so anecdotal, but: (a) the known ch3 night miss reproduces already at dusk;
(b) on the substream n and s each miss one spot, so combine them. **Person detection at night was
not tested** (nobody in frame). Do it in Phase 2 with playback clips of the SmartMotionHuman events.

## 5. Licence plate

Legible on ch1 main (960x1080) at the spot; unreadable on the substream. Plate OCR (S5) is
possible only from a main-stream crop, one-shot per arrival. Needs a night check.

## 6. XVR event push (`event_log.py`, 10 min sample, evening)

`eventManager.cgi?action=attach&codes=[All]` with digest auth: **works**, multipart stream.
Seen: `SmartMotionVehicle` (with a `Rect` box, e.g. `[217, 489, 798, 1021]`; coordinate scale not yet verified), `SmartMotionHuman`,
`VideoMotion` (`SmartMotionEnable: true`), plus `RecordDelete` housekeeping.
Event `index` is 0-based channel: 0 = ch1, 2 = ch3, 3 = ch4.

10-minute counts: ch1 SmartMotionVehicle Start ×19, SmartMotionHuman Start ×1, VideoMotion Stop ×28;
ch3 VideoMotion Stop ×8; ch4 vehicle ×4. Ch1 is a busy street (a vehicle event every ~30 s).
Observations: only `Stop` was captured for VideoMotion (no `Start` in this sample) — unexplained,
investigate. **Not measured yet:** event delay vs real time, night reliability, per-zone config
(done in the XVR UI). No `eventManager.getCaps` detail beyond basics; `VideoAnalyseRule` empty.
Verdict: usable as Stage-1 hint; keep the in-host motion gate as an OR fallback until delay and
night behaviour are measured.

## 7. Playback / recording

Continuous recording (`Record[0].TimeSection = 1 00:00:00-24:00:00`, pre-record 4 s). Playback RTSP
`/cam/playback?channel=N&subtype=S&starttime=YYYY_MM_DD_HH_MM_SS&endtime=...` works for both
streams (XVR-local time, i.e. UTC+3), ~1.4x real time on the substream; ~4 days retained.
Gotcha: ffmpeg `rawvideo` output duplicates frames to the 25 fps timebase unless
`-fps_mode passthrough` is given (real substream rate is 7 fps).

## 8. No-disk check

`docker run --read-only --network host --cpuset-cpus 0,2 --memory 700m --memory-swap 700m` on the
existing image: pipe-only ffmpeg grab returned all 5 requested frames (1,520,640 B, exactly 5x
352x288x3); writes to `/app` and `/tmp` both fail with "Read-only file system"; onnxruntime and
OpenCV import fine. **Implication:** the existing `CaptureWorker` uses a temp dir, so the new
modes need a pipe-only grab (`-f image2pipe`/`rawvideo pipe:1`) and a writable state volume only
for SQLite. Compose `tmpfs` for `/tmp` must NOT be added if "no image on disk" is to be literal.

## 9. D6 — which spot

Owner answer: **both ch1 and ch3 count as "in front of the house"; the owner's car is sometimes
parked elsewhere on the street.** Design consequences recorded in DECISIONS.md.

## 10. CPU topology / cpuset

`lscpu -e`: physical core A = CPUs **0,2**, core B = CPUs **1,3** (i3-4130, not 4170).
Proposed: `cpuset: "0,2"` for the new container, Jellyfin/Nextcloud keep core B (they are not
pinned today, so they will still run on core A too; pinning them is out of scope). The host
`Jellyfin` is a systemd service, not a container.
**Not done:** GUARD-latency with vs without a Jellyfin transcode — GUARD does not exist yet and a
synthetic transcode was not run (blocked, and it would risk other ffmpeg processes). Moved to
Phase 2 with the real Jellyfin.

## 11. Duty-cycle proxy (`replay_gate.py`, ch3 substream playback, 2-min windows)

Prototype gate: perimeter crop of the (scaled, dilated +40 %/+25 %) spot polygon → 160x90 gray →
diff vs running-average background (|d| > 25, >0.4 % of pixels) → 10 s tail. Global-change reject
at >40 % never fired.

| XVR-local window | frames | motion frames | active incl. tail |
|---|---|---|---|
| 29 Oct 21:00 | 841 | 133 | 39 % |
| 29 Oct 23:00 | 841 | 45 | 14 % |
| 30 Oct 01:00 | 841 | 54 | 24 % |
| 30 Oct 03:00 | 841 | 0 | 0 % |
| 30 Oct 05:00 | 841 | 0 | 0 % |
| 30 Oct 07:30 | 841 | 0 | 0 % |

6 samples × 2 min is **not** a night profile. Evening/early-night windows are active 14–39 % of
the time with an untuned gate and no ignore mask (street-side motion, trees). At the planned
~4 fps × 36–41 ms (320 crop) that would be a 6–16 % YOLO duty, under the 40 % cap, but the gate
must be tightened with an ignore mask. Ch1 was not replayed.

## Decisions D1–D8 status

| ID | Status |
|---|---|
| D1 snapshots from RAM | **Needs owner confirmation** (default yes) |
| D2 exempt from 30-day store | Keep (default) |
| D3 cloud second stage | Keep No |
| D4 phone presence | Keep Off |
| D5 ambiguous → GUARD | Keep (default) |
| D6 which spot | **Resolved: both ch1 and ch3**; car may be elsewhere on the street |
| D7 alert routing | Keep |
| D8 night definition | Camera clock is UTC+3; existing code treats night as 19–06 local — confirm it is evaluated in Europe/Istanbul, not the container's UTC |

## Open items / not done in Phase 0

- Jellyfin-contention latency test (Phase 2).
- Real overnight duty-cycle measurement (Phase 2/4; leave a logger running).
- Person/crouch detection at night; plate OCR at night.
- XVR event delay and night reliability; the missing `VideoMotion Start`.
- ch1 playback replay.
- Real int8 calibration set (only if int8 is revisited).
