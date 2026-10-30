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

## 2026-10-02 — Improvement plan Phase 6: VLM gate via AI Gate (Groq vision)

**Decision:** route the VLM gate through **AI Gate** (the user's LLM gateway) to a
**Groq vision model** — not a local ollama (6A). The user's words: "this vlm needs
to be configured under ai-gate and all request should go there." (AI Gate is the
user's own service; the source is on the host at /home/likewhosloss/ai-gate.)

**AI Gate changes (separate repo, backward-compatible):**
- `app/api/v1_chat.py`: `ChatMessage.content` `str` → `Union[str, list]` so the
  OpenAI vision array (text + image_url parts) is accepted. The token counter
  already handled the list form; the Groq client forwards it as-is.
- `config/models.yaml`: `qwen/qwen3.8-27b` is the vision model (modality chat →
  multimodal, added `vision` alias). Removed the incorrect `vision` alias from
  `openai/gpt-oss-20b` (a reasoning/text model — it 400'd on images).
- `config/callers.yaml`: added `parking-spotter` caller (50k tokens/$2 per day).

**Verifier (`parkingbot/verifier.py`):** added `verify.backend: ollama | aigate`.
The aigate path POSTs the ROI crop to `/v1/chat/completions` with `X-Caller-ID:
parking-spotter`, model `qwen/qwen3.8-27b`, OpenAI vision format. Same
`VerifyResult` interface; the push loop's GATE/AUDIT logic is unchanged.

**Networking:** the parking-spotter container was on an isolated bridge
(172.23.0.0/16) and could not reach the host's AI Gate. AI Gate uses
`network_mode: host`, so the container now uses `network_mode: host` too
(`aigate_url: http://localhost:8500`).

**Measured (2026-10-02):** car frame → `occupied=True` conf 0.90; empty frame →
`occupied=False`. Latency ~0.4-1.3s. Cost ~$0.002/call (807 tokens). The GATE runs
only on pending free flips (a few/day). The AUDIT is **off** — it would send a crop
to the cloud every poll (144/day); /accuracy uses feedback.jsonl (Phase 1).

**Gate order (end state):** stability (pre-debounce) → debounce → RAM guard →
tier-2 → **VLM (AI Gate)** → alert. Each gate can only suppress a free alert.

**Acceptance:** latency + cost recorded. Residual false-free rate vs Phase 3 is
being measured in the live week. Frames leave the LAN for the ROI crop on pending
free flips only (explicit opt-in).

**Decision: keep** (watch the live false-free rate and the AI Gate budget).

## 2026-10-30 — Spot Watch & Guard plan, Phase 0: recon and probes

Second plan series (SPOT_WATCH / GUARD modes + auto-switching). Phase 0 changed nothing in
the production bot; full numbers in `docs/PROBE_RESULTS.md`, probe scripts in `scripts/probes/`.

**Findings that change the plan:**
- Substream is **CIF-only, 352x288, ≤7 fps** (not 640x360 @ 5 fps). Decode ~5 % of a logical
  CPU vs ~31 % for the main stream → GUARD reads the substream, grabs main-stream frames on demand.
- XVR **event push works** and the XVR **records 24/7 with working playback RTSP**.
- Plain YOLOv8n misses the ch3 car at dusk/IR; tier-2 (yolov8s crop) finds it. SPOT_WATCH must
  reuse the existing stability + tier-2 (+ VLM) gates.
- OpenVINO: no gain (+60 MB). int8: ~20 % faster but lost detections → stay fp32/plain ORT.
  Shipped ONNX models are static 640; 320/416 crop models need export. 2 ORT threads on one
  core give only ~10–15 % over 1.
- Read-only container with pipe-only ffmpeg works; existing `CaptureWorker` (temp dir) is not
  reusable for the new modes. No `tmpfs` /tmp.
- cpuset: physical core pairs are CPUs 0,2 and 1,3 (CPU is an i3-4130). Swap is full on the host.

**D6 (owner, 2026-10-30): both ch1 and ch3 count as "in front of the house", and the owner's car
is sometimes parked elsewhere on the street.** Consequences: SPOT_WATCH watches both spots
independently; GUARD arms only when the owner's car is in one of the two spots (per-spot
channel/zone); no car in either spot ⇒ stay SPOT_WATCH and never send CAR_GONE for a car the
cameras cannot see; auto-switch must say which spot; departure is tracked per spot.

**Not done:** Jellyfin-contention latency (needs GUARD; Phase 2), real overnight duty cycle,
night person detection, XVR event delay, ch1 replay. D1 still awaits owner confirmation.

**Decision: Phase 0 complete pending owner review of D1 and the substream consequences.**

## 2026-10-30 — Spot Watch & Guard plan, Phase 1: mode controller + SPOT_WATCH

Implemented, tested and live-checked; **not yet deployed to the production container**
(owner to approve the restart: `docker compose up -d --build`).

**What was built**
- `parkingbot/spot_tracker.py` — pure per-spot state machine: 2 consecutive free readings,
  alert on the free transition (or "free right now" on entering SPOT_WATCH), reminders every
  10 min x3, "taken again", muted tracking during a pause, persisted state.
- `parkingbot/modes.py` — SQLite store (`state/spotter.db`: mode, auto flag, pause, per-spot
  tracker state; never images) + `ModeController` (UNKNOWN until verified against the camera,
  SPOT_WATCH, PAUSED; GUARD raises "not implemented" until Phase 2).
- `parkingbot/spotwatch.py` — the engine. Both spots, every 60 s; a free reading is re-checked
  ~10 s later on only the affected channel. Camera failures never count as free; 3 in a row
  -> ONE aggregated "unreachable" message, then exponential backoff, "reachable again" later.
  Telegram outage: in-memory outbox, retried each cycle, snapshot dropped after 5 min, message
  after 30 min. `/status` shows mode, spot states, per-camera health/latency, CPU/RSS, free RAM.
- Commands `/watch /guard /auto /pause 10m /resume /status` (`/guard` and `/auto` are stubs
  until Phases 2/3). `spotwatch.enabled: true` supersedes the 10-min push loop (two alerters
  would double up); the old code stays behind `spotwatch.enabled: false`.
- `capture.py`: the ffmpeg grab now writes to a pipe (`-f image2pipe ... pipe:1`), not a temp
  file — required for `read_only: true` and the no-disk rule.
- Compose: `read_only: true`, `cpuset: "0,2"` (one physical core), `mem_limit`/`memswap_limit`
  700m (no swap), no `cpus:` quota, new `/app/state` volume (SQLite only), no tmpfs `/tmp`.
  Dockerfile: `PYTHONDONTWRITEBYTECODE=1`, `TZ=Europe/Istanbul`.
- Timezone (owner): Istanbul. `apply_timezone()` pins the process (so the existing night-hour
  logic and the feedback "hour" now follow the camera clock instead of container UTC).

**Design decisions**
- SPOT_WATCH reads the **main stream** through the existing capture path, not the substream:
  the ROI polygons are in main-stream pixels and Phase 0 showed 352x288 detection is unreliable.
- Every "free" reading passes the existing gates before the tracker sees it: stability gate ->
  tier-2 (yolov8s crop, behind the RAM guard) -> (after 2 reads) the VLM gate on the first
  alert. A gate can only suppress an alert, never create one.
- **Live finding fixed:** on ch3 plain YOLOv8n reads FREE every cycle (the car is there), so
  tier-2 rescued it every minute (2.7 CPU-s/cycle, ~4.5 % of a core) because the stability gate
  never got a reference (it only learns from plain-YOLO occupied readings). The engine now hands
  tier-2's strong occupied result to the stability gate; steady state dropped to 1.7 CPU-s/cycle
  (~2.8 % of a core), stability sims 0.78-0.85 vs threshold 0.45.
- D1 confirmed by owner: alert snapshots are JPEG-encoded in RAM and sent straight to Telegram.
  Spot Watch alerts do NOT carry the feedback buttons (they need a saved crop = an image on disk).

**Measured (live, real XVR, read-only container pinned to CPUs 0,2, 700 MB, no swap)**
- Steady state: 1.6-1.7 CPU-s per 60 s cycle (~2.8 %), 3.4 % average incl. startup and model
  load. RSS ~190-220 MB, cgroup peak 200 MB, swap 0, `nr_throttled = 0`. Cycle wall time ~3.1 s.
- Forced-free on real frames, gates ON: no alert (tier-2 / stability overruled it).
- Forced-free, gates OFF: alert after the confirm re-check with an 87 KB snapshot decoding to
  1080x960, built in RAM.
- XVR unreachable: ONE message naming both cameras, no "free", mode stays UNKNOWN, backoff
  30 -> 60 -> 90 s.
- Latency worst case: freed right after a cycle -> 60 s + 10 s confirm + ~3 s processing = ~73 s
  (inside the ~75 s target but with little slack).

**Verification:** `python -m unittest discover -s tests` -> 133/133 (61 new: tracker, modes,
engine with fakes incl. latency/passing car/unplugged camera/pause/restart/gates/outbox,
pipe grab, command handlers + application wiring with mocked Telegram).

**Known gaps / for the owner**
- Not deployed; Telegram delivery itself was only tested with mocks (a second poller would
  409 against production).
- `/parking` still saves ROI crops to `data/frames` while `feedback.enabled: true` — that is an
  existing feature, not Spot Watch, but it means "no image ever on disk" is not yet literally
  true for the whole container. Say the word to turn it off for the new modes' sake.
- Exit criterion "no new image files on the host during a run" holds for the Spot Watch runs
  (read-only rootfs, tmpfs state); the 48 h soak and a real "spot freed" walk-through remain.
- `UNKNOWN` startup with the camera down sends the unreachable message but nothing else.

## 2026-10-30 — Spot Watch & Guard plan, Phase 2: GUARD core

Implemented, tested (213 unit tests) and live-checked on the real XVR; **not deployed** (the
production container still runs Phase 1 + the rescheduling fix). Owner decisions this phase:
`/guard` asks which spot the car is in; GUARD alerts are Turkish and carry the severity word
(BİLGİ / UYARI / ALARM).

**Modules:** `guard.py` (engine: persistent substream reader with backoff + stall watchdog,
latest-frame-wins slot, evidence ring, dispatcher, ALERT repeat/`/ack`), `guard_gate.py` (host
motion gate + XVR event listener), `guard_detector.py` (YOLOv8n on a crop window, person /
bicycle / motorcycle / car), `guard_tracker.py` (SORT-style, time-based ageing),
`guard_rules.py` (LOITER, CONTACT, MULTI), `guard_limiter.py` (duty-cycle limiter),
`guard_geometry.py` (zones), `night.py` (D8). Commands: `/guard` (inline buttons: which spot),
`/ack`, `/pause` (works from GUARD, keeps tracking muted), `/watch` (disarms), `/status`.

**How it works:** arming = median of 5 main-stream reads of the car in the chosen spot (existing
detector, tier-2 fallback for ch3) -> car zone; perimeter = zone +40 % width / +25 % height
(or a manual polygon), scaled into 352x288 substream pixels (anisotropic). Idle = ffmpeg decode
+ a ~1 ms gate per frame, **zero YOLO**. YOLO (320 crop, ~38 ms) runs only while the motion
gate / an XVR *human* event / a live track is active, capped at 4 fps and by the duty limiter
(>40 % busy over 60 s -> raise motion threshold, then 4 -> 2 -> 1 fps, recover after 5 min calm,
every step logged + sent as a silent INFO). A person standing still is kept alive by their
track, so LOITER still fires. Evidence: raw frames in a RAM ring (8 s); on WARNING/ALERT the
annotated key frame + up to 3 earlier frames go out as a media group (JPEG-encoded in RAM at
that moment), followed by one main-stream snapshot (the substream is tiny). Never persisted.

**Decisions**
- Rules (defaults, night x0.6 via sunset-sunrise for Istanbul +/-30 min, because IR starts
  before sunset): LOITER 10 s -> WARNING; CONTACT person box overlapping the car >=3 s -> ALERT;
  MULTI >=2 people in the perimeter -> WARNING. ALERT repeats every 30 s until `/ack` or the
  "Gördüm" button (capped at 10 repeats so a forgotten alert cannot spam forever); 60 s
  per-rule cooldown; INFO is silent. CROUCH / VEHICLE_STOP / IMPACT / CAR_GONE are Phases 3-4.
- XVR events gate on **`SmartMotionHuman` only**. Live finding: ch1 is a street and
  `SmartMotionVehicle` fires every ~7 s, which kept YOLO active ~50 % of the time for rules
  that are only about people. Codes are configurable (`guard.xvr_codes`).
- Restart safety: a persisted GUARD is never trusted; the bot starts in SPOT_WATCH and re-arms
  only if the camera still shows a car in that spot.
- 320 model lives at `/mnt/storage/parking-spotter/models/yolov8n_320.onnx` (user-writable, no
  sudo) and is mounted read-only at `/app/models-guard`; a 416 export sits next to it.

**Live bug found and fixed:** the first live run showed ~96 fps from the 7 fps substream and
43 % of a core: ffmpeg's rawvideo output duplicates frames up to the stream's 100 fps timebase.
Fixed with `-fps_mode passthrough` (same quirk as the Phase 0 playback probe).

**Measured (real XVR, read-only container pinned to CPUs 0,2, 700 MB, no swap)**
- Idle, ch3: 604 frames/90 s (6.7 fps), **0 YOLO runs, 4.2 % of one core**; ch1 with a short
  motion burst: 6.9 %. Peak cgroup memory 194-350 MB (includes arming), swap 0,
  `nr_throttled = 0`. Arming takes 12-15 s.
- Replay of 3 recorded evening windows (5 min each, ch3 21:00 and 23:00, ch1 20:00) through
  the real pipeline: YOLO 117 / 155 / 0 runs, 3.9-7.1 % CPU, duty <= 5 %, **no alerts, no
  person tracks** (no one was demonstrably in those clips; see gaps).
- Detector decode checked on the standard bus photo (finds the people, ignores the bus).

**Known gaps (be honest):**
- **Person detection on these cameras is unvalidated.** No recorded window contained a known
  person, so LOITER/CONTACT have only been exercised with scripted detections. ch3 is a steep
  top-down IR view, where COCO YOLO is weakest. A live walk-through is needed.
- The XVR search API only returns hourly "Timing" files (no event metadata), so past human
  events cannot be located for replay. Event delay and night reliability remain unmeasured.
- Latency (condition met -> sent) is only unit-tested (about one detection interval); the
  <=3 s end-to-end target is not yet measured live over Telegram.
- Jellyfin-contention test not done (needs a real transcode running). 48 h soak is Phase 4.
- Until Phase 3 (`CAR_GONE`) GUARD does not notice if the car drives away; use `/watch`.
- Evidence is 352x288 (substream); the main-stream follow-up photo is what a human can judge.

## 2026-10-30 — Cross-camera check for "Ön (sarı ev)" (owner: ch3 sees both spots)

Owner input: ch3 contains BOTH spots ("ev önü" and "Ön (sarı ev)", the latter = the nearer of the two
cars on the right of ch3), and ch3 is the ONLY camera that sees "ev önü" (so a ch3 outage blinds it
with no fallback; the bot already reports "durumu bilinmiyor").

**Built:** optional `cross_check` per spot (`Spot.cross`). SPOT_WATCH consults it only when the
primary reading says free and the spot was confirmed occupied: if the other camera still sees a car
the free reading is vetoed (`cross_override`). It can only suppress, never create, an alert; an
unavailable second camera means no opinion; a disagreement lasting `cross_max_veto_minutes` (10)
stops vetoing so a wrong mapping can delay an alert but never swallow it.

**Evidence (30 h of XVR recordings, 40 samples, both cameras):**
- ch1's spot was occupied 40/40 (the same car never left; plain YOLO never missed it, even at night).
- ch3 "car 2" region: crop-based detection finds it by day and misses it at night (15/40); a tier-2 pass
  over the WHOLE frame finds it at night and misses it by day (18/40); neither alone is reliable on
  ch3. The cross-view therefore ORs three opinions (detector, tier-2 crop, tier-2 whole frame).
- "car 3" (far right) was empty 38/40: it is not the ch1 car, consistent with the owner's mapping.
  The history cannot PROVE the mapping because ch1 never freed; it is consistent with it.
- Live method check: car 2 and car 3 read as cars, two clearly empty regions read as empty.

**Honest value:** small. ch1 is already reliable at night, and ch3's view of that car is flaky, so this
mostly guards against a rare ch1 false-free. It costs ~1 s CPU only on a free reading. Tests: 224/224
(8 new). Not deployed (needs a restart); not committed.

## 2026-10-30 — Spot Watch & Guard plan, Phase 3: enrollment, auto-switching, CAR_GONE

Implemented, tested (285 unit tests) and live-checked on the real XVR with a throw-away RAM database;
**not deployed, nothing enrolled in production** (the owner must run `/enroll`).

**Modules:** `appearance.py` (zero-model features), `car_store.py` (SQLite, vectors only), `matcher.py`
(policy), `autoswitch.py` (enroll / arrival / ask / undo / default-GUARD), `carfinder.py` (shared "find
the car in this spot", also used by GUARD arming), `guard_presence.py` (is the car still there),
CAR_GONE / departure / escalation in `guard.py`. Commands: `/enroll [rain]` (asks which spot), `/auto`
(on; `/auto off`), inline Yes/No, undo, [Bendim]/[Ben değilim]. Manual `/watch`/`/guard` turns auto off.

**Model choice (plan 5.3):** the zero-model baseline is enough, no network added. Features = HSV colour
histogram + tone histogram + 32x20 layout thumbnail + blurred edge map (weights .2/.1/.4/.3) plus a
box-geometry prior; stored as a 1445-float vector (checked by test: nothing image-like is stored).
Condition (day / night-IR / rain) comes from the frame itself (IR frames are grayscale), not the clock.

**Evidence (230 real crops, 24 h of both cameras, references excluded within +-2 h):**
- ch3 (van in "ev önü"): owner 0.88-0.97 vs other cars <= 0.53 (gap +0.42), day and night.
- ch1 (the car in "Ön (sarı ev)"): owner 0.86-0.99 vs other cars <= 0.65. Only 2 of 49 owner crops
  were bad and both were junk detector boxes (a 40x199 sliver, two cars merged): a box-sanity rule
  (box area 0.4-1.2x the spot polygon bbox) rejects exactly those; arrivals also use the median of 3.
- **IR query vs daylight-only references scored ~0.70 for the owner and <= 0.49 for others.** That is
  below t_lo, so a naive policy would silently ignore the owner at night. Fixed: with no same-condition
  references the matcher never says "auto", and says "other" only below 0.55 (otherwise it asks).
- Live (night): enrolling 15 samples took ~55 s per spot; fresh reads scored 0.977-0.990 (auto); the
  other spot's car scored 0.14-0.24 (other).

**Policy (conservative on purpose):** >= 0.92 auto-switch to GUARD (+ undo button), 0.75-0.92 ask
"is this your car?" [Evet]/[Hayır] (no answer in 5 min -> GUARD with a notice, D5), < 0.75 ignore silently.
Never auto with < 5 references for the spot; a known "not mine" sample that looks as close pulls the
score down. Every confirmation/correction becomes a labelled sample. Acts only on a confirmed
free->occupied flip while SPOT_WATCH is running, auto is on and alerts are not paused; the car must
settle ~20 s and be seen in >= 2 of 3 reads, otherwise the evaluation is aborted.

**CAR_GONE / departure:** the substream car-zone edge signature is compared every 5 s (no YOLO). Same car
0.69-0.85, empty road <= 0.07 (real frames, day and night), threshold 0.30. Replaying yesterday's dusk
(colour -> IR) gave **no false suspicions** (min similarity 0.62 ch1 / 0.70 ch3 with the 60 s reference
refresh; 0.38 even with a fixed reference for 2.5 h). A suspicion (3 low checks, ~15 s) must then be
confirmed by main-stream reads that ALL miss the car, so exposure shifts or a person in front cannot
alarm. Not paused -> ALARM "ARAÇ GİTTİ" with photos from the RAM ring and [Bendim]/[Ben değilim];
no reply in 2 min or "Ben değilim" -> repeats louder (ACİL) every 15 s. Paused (owner walking to the
car) and absent ~90 s -> quiet departure to SPOT_WATCH; the now-free spot is not announced back to them.
A camera that cannot be read gives no verdict and no alarm.

**Bug found and fixed along the way:** Telegram albums cannot carry inline buttons, so the CAR_GONE
buttons would have been lost on the first alert. The outbox now follows an album with a short
"👆 Yanıtla:" message carrying the controls.

**Known gaps (be honest):**
- **Hard negatives are untested:** the same model/colour parking in the same spot. That is why t_hi is
  high and uncertain cases ask. Real data (your confirmations over days) is needed to loosen it.
- **Nothing is enrolled in production.** The "9 of 10 own-car arrivals" criterion cannot be measured
  without real arrivals; only self-matching and 30 h of crops were measured. Run `/enroll` by day, at
  night and in the rain (target >= 10 per condition) with the car parked in the spot.
- Which car is "yours" was assumed (GUARD was armed on the ch1 spot). `/auto` only reacts to NEW arrivals;
  a car already parked when you turn it on needs a manual `/guard`.
- CAR_GONE and the whole arrival flow are unit-tested with fakes and checked on real frames, but no real
  departure/arrival happened during testing, and Telegram delivery is mocked.
- S4 (phone presence, D4) and S5 (plate OCR) are not implemented; plates are only legible on the ch1 main stream.
- ch3 stays the weak camera for night detection (see the cross-check notes); matching there used the
  van, which is not necessarily the owner's car.

## 2026-10-30 — Phase 4 (start): Prometheus metrics

**Built:** `parkingbot/metrics.py` (prometheus_client): instrumented counters/histograms (alerts by
severity and rule, rule latency, reconnects, frames dropped/skipped, limiter steps, gate overrides, grab
failures, Telegram send failures) + a scrape-time collector for state (mode, spot states, frame age, YOLO
duty, limiter step, camera health, auto-switch outcomes, references, host RAM/swap, container CPU / memory /
**CFS throttling**). The engines take an optional `metrics` (no-op default), so nothing depends on it.
Endpoint `0.0.0.0:9108` (config `metrics:`; never fatal: a taken port only logs). Optional Uptime Kuma push
heartbeat from `HEARTBEAT_URL` (in `.env`, it carries a token). Dashboard `docs/grafana-parking-spotter.json`
(25 panels, generated by `scripts/make_grafana_dashboard.py`), setup + queries in `docs/METRICS.md`.
Follows the OCR service's pattern (host process, scraped as `host.docker.internal:<port>`).

**Verified live** (real XVR, read-only pinned container): a host scrape returned 51 `parking_*` samples
with a live GUARD (frame age 0.04 s, 57 evidence frames, XVR events attached, throttled periods 0).
Every metric the dashboard queries is checked against the exporter (3 container-memory gauges only exist
inside a container). Tests: 302/302 (17 new: instrumentation, scrape, no secrets, one failing section never
breaks the scrape, HTTP serving, port-in-use, heartbeat, presence warm-up).

**A bug the metrics exposed:** `parking_guard_car_zone_similarity` read 0.0 for a parked car. The presence
reference was seeded from the first frames of the HEVC stream, which are garbage ("Could not find ref with
POC"), so every arming produced ~15 s of "car changed" and three needless main-stream confirmations — the
"false alarm: sim=0.00" in the production log at 21:43:31 that I had put down to a person. Now the monitor
warms up for 4 s before taking its reference; live similarity is 0.94 / 0.76. (Test added.)

**Not done / needs the owner:** the host firewall drops Prometheus -> :9108 (verified: scrape from inside the
Prometheus container times out while the endpoint is up), the scrape job is not in `~/monitoring/prometheus.yml`
(not this repo's stack) and the dashboard is not imported. Steps in `docs/METRICS.md`. Still open for Phase 4:
CROUCH / IMPACT / VEHICLE_STOP, night tuning and false-positive review, the 48 h soak.

## 2026-10-30 — GUARD false alarms on street pedestrians (owner report: "margin too wide")

**Owner report:** a person walking in the street was detected as an intruder.
**What the logs said:** both alerts were `CONTACT` (ALARM, held ~2 s), not LOITER — so the perimeter margin was
not the main cause. Replaying the recorded ch1 substream at 22:39 and 22:44 against the production zones
showed the culprits were pedestrians on the lane beside / behind the car, whose boxes overlap the car's
image box by 15-30 % (camera angle) while their CENTRES stay outside it.

**Two defects, one real, one geometric (found by replaying real clips through the rules engine):**
1. *Coasting counted as contact.* One detection followed by ~2 s of the tracker coasting on a stale box was
   treated as "touching the car for 2 s" (`pts=1` in the trace). Also produced 3 spurious CONTACTs in the
   owner's own walk-through. Fix: CONTACT counts only frames where the track was actually seen recently and
   needs >= 3 real detections in the window (scaled to the current YOLO rate, so it still works at 1 fps).
2. *Edge overlap is not contact.* Fix: the person's CENTRE must be inside the car box grown by 6 %
   (`contact_center_margin`), not merely overlap it by 10 %.

**Perimeter narrowed as requested:** +40 %/+25 % -> **+16 %/+10 %** (`perimeter_dilate_w/h`; per-spot
`perimeter_margin` override available).

**Evidence (replay of 3 recorded clips, night thresholds):**
| settings | pedestrian 22:39 | pedestrian 22:44 | owner walk-through 21:43 |
|---|---|---|---|
| before | CONTACT | CONTACT | MULTI, CONTACT x4, LOITER |
| real-detections rule only | CONTACT | CONTACT | MULTI, CONTACT, LOITER |
| + centre test, margin .40/.25 | none | none | MULTI, CONTACT, LOITER |
| + centre test, margin **.16/.10** (chosen) | none | none | MULTI, CONTACT, LOITER |
| + centre test, margin .10/.08 | none | none | MULTI, CONTACT (LOITER lost) |
The centre test is what silences passers-by; narrowing the margin alone changed nothing for CONTACT. .16/.10 is
the narrowest margin that kept the owner's LOITER. Regression tests embed the real recorded detections of both
false alarms (309/309 tests).

**Caveats:** two pedestrian events and one walk-through is a small sample. Someone walking *very* slowly right
along the car's flank could still look like contact, and a thief who only brushes past for < 1.8 s will not alarm
(that is the trade-off). Day thresholds are longer (3 s) so daytime is stricter still. CROUCH/IMPACT will need
the same replay-first treatment. Not deployed yet.
