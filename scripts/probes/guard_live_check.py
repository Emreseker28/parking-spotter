"""Live check of the GUARD engine against the real XVR with a console notifier only
(no Telegram, no production state). Run inside the read-only container, e.g.

  docker run --rm --read-only --tmpfs /state ... --entrypoint python IMAGE \
      /probes/guard_live_check.py --spot 0 --seconds 90
  ... --spot 1 --playback 2026_10_29_21_00_00 --seconds 300     # replay an XVR recording

--playback START   read the SUBSTREAM from XVR playback (XVR-local time, UTC+3) instead of live
                   (zones still come from arming on the live main stream)
"""
import argparse, asyncio, logging, os, sys, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--spot", type=int, default=0)
ap.add_argument("--seconds", type=int, default=90)
ap.add_argument("--playback", default=None)
ap.add_argument("--gate", default=None, help="motion|xvr|both (default from config; playback forces motion)")
ap.add_argument("--no-followup", action="store_true")
ap.add_argument("--metrics-port", type=int, default=0, help="also serve /metrics on this port (0.0.0.0)")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="    %(asctime)s %(name)s: %(message)s")
for n in ("httpx", "httpcore", "hpack"):
    logging.getLogger(n).setLevel(logging.WARNING)

from parkingbot.config import load_settings
from parkingbot.capture import CaptureWorker
from parkingbot.detection import Detector
from parkingbot.modes import ModeController, StateStore
from parkingbot.spotwatch import Outbox
from parkingbot.guard import GuardEngine, GuardConfig
from parkingbot.bot import apply_timezone
from pathlib import Path

s = load_settings()
tz = apply_timezone(s)
cfg = GuardConfig.from_settings(s)
if a.gate:
    cfg.gate = a.gate
if a.playback:
    cfg.gate = "motion"          # XVR events are live-only
if a.no_followup:
    cfg.followup_main = False

class Console:
    def __init__(self): self.n = 0
    async def send_text(self, text, silent=False, **kw):
        print(f"  [TELEGRAM text silent={silent} ack={kw.get('ack', False)}] {text!r}", flush=True)
    async def send_photo(self, jpeg, caption, **kw):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        print(f"  [TELEGRAM photo {len(jpeg)} B {None if img is None else img.shape[:2]} ack={kw.get('ack', False)}] {caption!r}", flush=True)
    async def send_media_group(self, jpegs, caption, **kw):
        sizes = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR).shape[:2] for j in jpegs]
        print(f"  [TELEGRAM media-group {len(jpegs)} images {sizes}] {caption!r}", flush=True)

store = StateStore(Path("/state/spotter.db"))
ctl = ModeController(store, guard_available=True); ctl.verify()
cap = CaptureWorker(s)
outbox = Outbox(Console())
src = None
if a.playback:
    import datetime as dt
    from urllib.parse import quote
    t0 = dt.datetime.strptime(a.playback, "%Y_%m_%d_%H_%M_%S")
    end = (t0 + dt.timedelta(seconds=a.seconds + 30)).strftime("%Y_%m_%d_%H_%M_%S")
    async def src(channel):
        sec = s.secrets
        url = (f"rtsp://{quote(sec.rtsp_user, safe='')}:{quote(sec.rtsp_pass, safe='')}@{sec.nvr_ip}:554/cam/playback"
               f"?channel={channel}&subtype=1&starttime={a.playback}&endtime={end}")
        w, h = cfg.sub_size
        return await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error", "-re", "-rtsp_transport", "tcp", "-i", url, "-an",
            "-fps_mode", "passthrough", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

from parkingbot.metrics import BotMetrics
from types import SimpleNamespace
metrics = BotMetrics()
eng = GuardEngine(s, ctl, cap, Detector(s), outbox, cfg=cfg, tz=tz, source_factory=src, metrics=metrics)
if a.metrics_port:
    from prometheus_client import start_http_server
    fake_bot = SimpleNamespace(controller=ctl, spotwatch=None, guard=eng, auto=None, settings=s,
                               metrics=metrics, _mem_available_mb=lambda: 0.0)
    metrics.attach(fake_bot)
    start_http_server(a.metrics_port, addr="0.0.0.0", registry=metrics.registry)
    print(f"metrics on 0.0.0.0:{a.metrics_port}", flush=True)

def cg(name):
    try: return open(f"/sys/fs/cgroup/{name}").read().split()
    except OSError: return []

async def main():
    t0 = time.perf_counter()
    ok, msg = await eng.arm(a.spot)
    print(f"ARM {'OK' if ok else 'FAILED'} in {time.perf_counter() - t0:.1f}s: {msg}", flush=True)
    if not ok:
        return
    z = eng.zones
    print(f"  zones (substream px): car={tuple(int(v) for v in z.car)} perimeter={tuple(int(v) for v in z.perimeter_bbox)} window={eng.window}")
    ctl.set_manual(__import__('parkingbot.modes', fromlist=['Mode']).Mode.GUARD, s.spots[a.spot].name)
    c0 = time.process_time(); t1 = time.perf_counter()
    import os as _os
    ot = _os.times(); cpu0 = ot.user + ot.system + ot.children_user + ot.children_system
    for i in range(a.seconds // 15):
        await asyncio.sleep(15)
        print(f"  t+{(i + 1) * 15:>3}s frames={eng.frames} dropped={eng.dropped} skipped={eng.skipped} yolo={eng.detections_run} "
              f"duty={eng.limiter.duty(time.time()):.0%} step={eng.limiter.step} active={eng._active(time.time())} "
              f"motion_age={('%.0fs' % (time.time() - eng.last_motion_ts)) if eng.last_motion_ts else 'never'} tracks={len(eng.tracker.tracks)} xvr={eng.xvr.count if eng.xvr else '-'} "
              f"alerts={eng.alerts}", flush=True)
    wall = time.perf_counter() - t1
    ot = _os.times(); cpu1 = ot.user + ot.system + ot.children_user + ot.children_system
    print(f"\nrun {wall:.0f}s: process+ffmpeg CPU {cpu1 - cpu0:.2f}s => {(cpu1 - cpu0) / wall * 100:.1f}% of one core")
    print("--- status ---\n" + eng.status_text())
    st = dict(zip(cg("cpu.stat")[::2], cg("cpu.stat")[1::2]))
    peak = cg("memory.peak")
    print("cgroup nr_throttled =", st.get("nr_throttled"), "| memory.peak MB =", (int(peak[0]) / 1048576) if peak else "n/a",
          "| swap =", cg("memory.swap.current"))
    await eng.disarm("done")
asyncio.run(main())
