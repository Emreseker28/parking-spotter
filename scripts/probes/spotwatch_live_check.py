"""Live check of the SPOT_WATCH engine against the real XVR, console notifier only
(no Telegram, no production state). Run inside the read-only container:

  docker run --rm --read-only --tmpfs /state ... python /probes/spotwatch_live_check.py \
      --cycles 4 [--force-free 1] [--no-gates]

--force-free N   flip spot N (0-based) to "free" after detection (real frame, real gates)
--no-gates       disable stability + tier-2 + VLM so a forced free reaches the alert/snapshot path
"""
import argparse, asyncio, os, sys, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cycles", type=int, default=4)
ap.add_argument("--interval", type=float, default=60.0)
ap.add_argument("--confirm", type=float, default=10.0)
ap.add_argument("--force-free", type=int, default=None)
ap.add_argument("--no-gates", action="store_true")
a = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format="    %(asctime)s %(name)s: %(message)s")
for n in ("httpx", "httpcore", "hpack"):
    logging.getLogger(n).setLevel(logging.WARNING)
from parkingbot.config import load_settings
from parkingbot.capture import CaptureWorker
from parkingbot.detection import Detector
from parkingbot.verifier import Verifier
from parkingbot.stability import StabilityGate
from parkingbot.modes import ModeController, StateStore
from parkingbot.spotwatch import SpotWatch, WatchConfig
from parkingbot.bot import ParkingBot

s = load_settings()
if a.no_gates:
    s.raw["detection_tier2"] = {**s.raw.get("detection_tier2", {}), "enabled": False}
    s.raw["stability"] = {**s.raw.get("stability", {}), "enabled": False}
    s.raw["verify"] = {**s.raw.get("verify", {}), "enabled": False}

class Console:
    async def send_text(self, text, silent=False):
        print(f"  [TELEGRAM text silent={silent}] {text}", flush=True)
    async def send_photo(self, jpeg, caption):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        print(f"  [TELEGRAM photo {len(jpeg)} B, decodes to {None if img is None else img.shape[:2]}] {caption}", flush=True)

store = StateStore(__import__("pathlib").Path("/state/spotter.db"))
ctl = ModeController(store)
cfg = WatchConfig(interval=a.interval, confirm_delay=a.confirm,
                  min_available_mb=float(s.detection_tier2.get("min_available_mb", 2500)))
sw = SpotWatch(s, ctl, store, CaptureWorker(s), Detector(s), Verifier(s), StabilityGate(s),
               Console(), mem_available=ParkingBot._mem_available_mb, cfg=cfg)
if a.force_free is not None:
    target = s.spots[a.force_free].name
    inner = sw._reader
    async def forced(ch):
        rs = await inner(ch)
        for r in rs:
            if r.spot.name == target:
                print(f"  (forcing {target!r}: real detector said occupied={r.occupied}, cov={r.coverage:.2f})")
                r.occupied = False
        return rs
    sw._reader = forced

def cgroup(name):
    try:
        return open(f"/sys/fs/cgroup/{name}").read().split()
    except OSError:
        return []

async def main():
    t0 = time.time(); c0 = sw._cpu_seconds()
    for i in range(a.cycles):
        cc = sw._cpu_seconds()
        d = await sw.run_cycle()
        print(f"cycle {i+1}: cpu {sw._cpu_seconds() - cc:.2f}s kind={sw.last_cycle.kind} {sw.last_cycle.duration_ms:.0f} ms mode={ctl.mode.value} "
              f"spots={ {n: t.confirmed for n, t in sw.trackers.items()} } next in {d:.0f}s", flush=True)
        if i < a.cycles - 1:
            await asyncio.sleep(d)
    wall = time.time() - t0
    print(f"\nwall {wall:.0f}s, process+ffmpeg CPU {sw._cpu_seconds() - c0:.2f}s "
          f"=> {(sw._cpu_seconds() - c0) / wall * 100:.1f}% of one core (incl. startup + model load)")
    print("--- status ---\n" + sw.status_text())
    stat = dict(zip(cgroup("cpu.stat")[::2], cgroup("cpu.stat")[1::2]))
    print("cgroup cpu.stat nr_throttled =", stat.get("nr_throttled"), "| memory.peak MB =",
          (int(cgroup("memory.peak")[0]) / 1048576) if cgroup("memory.peak") else "n/a",
          "| swap.current =", cgroup("memory.swap.current"))
asyncio.run(main())
